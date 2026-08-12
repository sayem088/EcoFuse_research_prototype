"""
EcoFuse: Full Model Assembly
==============================
Integrates all components into a single forward pass:

  Inputs  -->  Encoders  -->  EUGF Fusion  -->  Dual Decoder  -->  DEcoDE ODE
                                                     |
                                                     v
                                              4 Output Maps:
                                              1. AGB (Mg/ha)
                                              2. Degradation Risk
                                              3. Predictive Uncertainty
                                              4. Ecological Params (r,K,d)

Joint Loss:
  L_total = w1*L_Focal + w2*L_RMSE + lambda1*L_EUGF + lambda2*L_ODE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import OpticalEncoder, SAREncoder, GEDIEncoder
from models.eugf import EUGF, eugf_regularization_loss
from models.decode_ode import DEcoDE, ode_consistency_loss


class EcoFuse(nn.Module):
    """
    Full EcoFuse model: multi-modal encoder -> EUGF fusion -> 
    dual decoders (regression + segmentation) -> DEcoDE temporal constraint.
    """
    
    def __init__(self, feat_dim=256, K=10, num_risk_classes=4, climate_dim=3):
        """
        Args:
            feat_dim:         feature dimension for all encoders
            K:                number of Dirichlet pseudo-categories
            num_risk_classes: degradation risk classes (Low/Moderate/High/Critical)
            climate_dim:      ERA5 climate vector size
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.K = K
        self.num_risk_classes = num_risk_classes
        
        # --- Modality Encoders ---
        self.optical_encoder = OpticalEncoder(in_channels=6, feat_dim=feat_dim, K=K)
        self.sar_encoder = SAREncoder(in_channels=3, feat_dim=feat_dim, K=K)
        self.gedi_encoder = GEDIEncoder(in_dim=1, feat_dim=feat_dim, K=K)
        
        # --- EUGF Fusion ---
        self.eugf = EUGF(feat_dim=feat_dim, K=K, num_modalities=3)
        
        # --- Decoder Head 1: AGB Regression ---
        self.agb_decoder = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # AGB in normalized scale [0, 1]
            nn.Sigmoid()
        )
        
        # --- Decoder Head 2: Degradation Risk Segmentation ---
        self.risk_decoder = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_risk_classes)  # logits for 4 risk classes
        )
        
        # --- DEcoDE Temporal ODE ---
        self.decode = DEcoDE(climate_dim=climate_dim, hidden=32, d_max=0.3)
    
    def forward(self, optical, sar, gedi, climate_vec):
        """
        Full forward pass.
        
        Args:
            optical:     (B, 6, H, W) normalized Sentinel-2
            sar:         (B, 3, H, W) normalized Sentinel-1
            gedi:        (B, 1) normalized GEDI AGBD (spatially averaged)
            climate_vec: (B, 3) ERA5 climate vector
        
        Returns:
            outputs: dict with all predictions and intermediate values
        """
        # --- Step 1: Encode each modality ---
        f_opt, alpha_opt, u_opt = self.optical_encoder(optical)
        f_sar, alpha_sar, u_sar = self.sar_encoder(sar)
        f_gedi, alpha_gedi, u_gedi = self.gedi_encoder(gedi)
        
        # --- Step 2: EUGF Fusion ---
        f_fused, alpha_fused, u_fused, weights, deltas = self.eugf(
            features_list=[f_opt, f_sar, f_gedi],
            alphas_list=[alpha_opt, alpha_sar, alpha_gedi],
            uncertainties_list=[u_opt, u_sar, u_gedi]
        )
        
        # --- Step 3: Dual Decoder Heads ---
        agb_pred = self.agb_decoder(f_fused)  # (B, 1) normalized AGB
        risk_logits = self.risk_decoder(f_fused)  # (B, num_risk_classes)
        
        # --- Step 4: DEcoDE ODE Layer ---
        # Use AGB prediction as initial condition, integrate forward
        agb_for_ode = agb_pred * 500.0  # denormalize to Mg/ha for ODE
        B_ode, eco_params = self.decode(agb_for_ode, climate_vec, dt=1.0)
        B_ode_norm = B_ode / 500.0  # re-normalize
        
        outputs = {
            # Primary outputs (4 maps)
            'agb_pred': agb_pred,               # (B, 1) AGB normalized
            'risk_logits': risk_logits,          # (B, 4) risk class logits
            'uncertainty': u_fused,              # (B, 1) predictive uncertainty
            'eco_params': eco_params,            # dict: r, K, d each (B, 1)
            
            # ODE outputs
            'agb_ode': B_ode_norm,               # (B, 1) ODE-predicted AGB
            
            # Fusion internals (for visualization)
            'weights': weights,                  # (B, 3) modality weights
            'deltas': deltas,                    # (B, 3) discounting factors
            'alpha_fused': alpha_fused,          # (B, K) fused evidence
            
            # Per-modality uncertainties
            'u_opt': u_opt,                      # (B, 1)
            'u_sar': u_sar,                      # (B, 1)
            'u_gedi': u_gedi,                    # (B, 1)
        }
        
        return outputs


# ---------------------------------------------------------------------------
# Joint Loss Function
# ---------------------------------------------------------------------------

class EcoFuseLoss(nn.Module):
    """
    L_total = w1 * L_Focal + w2 * L_RMSE + lambda1 * L_EUGF + lambda2 * L_ODE
    
    Default weights from proposal:
      w1 = 1.0   (Focal loss for risk segmentation)
      w2 = 1.0   (RMSE for AGB regression)
      lambda1 = 0.1   (EUGF evidential regularization)
      lambda2 = 0.05  (ODE temporal consistency)
    """
    
    def __init__(self, w1=1.0, w2=1.0, lambda1=0.1, lambda2=0.05,
                 focal_gamma=2.0, K=10):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.focal_gamma = focal_gamma
        self.K = K
    
    def focal_loss(self, logits, targets, gamma=2.0):
        """
        Focal Loss for multi-class segmentation.
        Handles class imbalance by down-weighting easy examples.
        """
        probs = F.softmax(logits, dim=-1)
        targets_oh = F.one_hot(targets, logits.shape[-1]).float()
        pt = (probs * targets_oh).sum(dim=-1)  # probability of true class
        focal_weight = (1.0 - pt) ** gamma
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        return (focal_weight * ce_loss).mean()
    
    def forward(self, outputs, agb_target, risk_target=None):
        """
        Compute joint loss.
        
        Args:
            outputs:     dict from EcoFuse.forward()
            agb_target:  (B, 1) ground truth normalized AGB
            risk_target: (B,) ground truth risk class indices (optional)
        
        Returns:
            total_loss: scalar
            loss_dict:  dict with individual loss components
        """
        # L_RMSE: AGB regression
        l_rmse = torch.sqrt(F.mse_loss(outputs['agb_pred'], agb_target) + 1e-8)
        
        # L_Focal: Risk segmentation (use synthetic labels if none provided)
        if risk_target is not None:
            l_focal = self.focal_loss(outputs['risk_logits'], risk_target,
                                      self.focal_gamma)
        else:
            # Derive risk classes from AGB: Low(>0.6), Moderate(0.3-0.6), 
            # High(0.1-0.3), Critical(<0.1)
            risk_from_agb = torch.zeros(agb_target.shape[0], dtype=torch.long,
                                       device=agb_target.device)
            agb_flat = agb_target.squeeze(-1)
            risk_from_agb[agb_flat >= 0.6] = 0   # Low risk
            risk_from_agb[(agb_flat >= 0.3) & (agb_flat < 0.6)] = 1  # Moderate
            risk_from_agb[(agb_flat >= 0.1) & (agb_flat < 0.3)] = 2  # High
            risk_from_agb[agb_flat < 0.1] = 3   # Critical
            l_focal = self.focal_loss(outputs['risk_logits'], risk_from_agb,
                                      self.focal_gamma)
        
        # L_EUGF: Evidential regularization (KL divergence)
        l_eugf = eugf_regularization_loss(outputs['alpha_fused'], K=self.K)
        
        # L_ODE: Temporal consistency
        l_ode = ode_consistency_loss(outputs['agb_pred'], outputs['agb_ode'])
        
        # Total
        total = (self.w1 * l_focal + self.w2 * l_rmse
                 + self.lambda1 * l_eugf + self.lambda2 * l_ode)
        
        loss_dict = {
            'total': total.item(),
            'focal': l_focal.item(),
            'rmse': l_rmse.item(),
            'eugf': l_eugf.item(),
            'ode': l_ode.item(),
        }
        
        return total, loss_dict
