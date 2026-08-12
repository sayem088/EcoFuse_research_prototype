"""
Evidential Uncertainty-Gated Fusion (EUGF)
============================================
Core novelty of EcoFuse: fuses multi-modal evidence using Dempster-Shafer
theory with learned contextual discounting.

Pipeline:
  1. Receive per-modality (features_m, alpha_m) from encoders
  2. Compute per-modality uncertainty: u_m = K / S_m
  3. Learn contextual discounting: delta_m = MLP(feature_stats, quality)
  4. Discount evidence: alpha_tilde_m = delta_m * (alpha_m - 1) + 1
  5. Dempster-Shafer combination: alpha_fused_k = sum_m(alpha_tilde_m_k - 1) + 1
  6. Reliability-weighted feature aggregation: f_fused = sum(w_m * f_m)
     where w_m = (S_m * delta_m) / sum(S_m' * delta_m')

References:
  - Sensoy et al. (2018) "Evidential Deep Learning"
  - Dempster-Shafer Theory of Evidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualDiscountNet(nn.Module):
    """
    Learns a discounting factor delta in [0, 1] for each modality based
    on input quality signals (feature statistics, uncertainty).
    
    High delta = trust this modality.
    Low delta  = suppress this modality (e.g., cloud-degraded optical).
    """
    def __init__(self, feat_dim=256):
        super().__init__()
        # Input: feature statistics (mean, std, max) + uncertainty
        self.net = nn.Sequential(
            nn.Linear(4, 32),  # 3 stats + 1 uncertainty
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()  # output in [0, 1]
        )
    
    def forward(self, features, uncertainty):
        """
        Args:
            features: (B, feat_dim)
            uncertainty: (B, 1)
        Returns:
            delta: (B, 1) discounting factor
        """
        # Compute feature statistics as quality proxy
        feat_mean = features.mean(dim=-1, keepdim=True)   # (B, 1)
        feat_std = features.std(dim=-1, keepdim=True)     # (B, 1)
        feat_max = features.max(dim=-1, keepdim=True)[0]  # (B, 1)
        
        stats = torch.cat([feat_mean, feat_std, feat_max, uncertainty], dim=-1)  # (B, 4)
        delta = self.net(stats)  # (B, 1)
        return delta


class EUGF(nn.Module):
    """
    Evidential Uncertainty-Gated Fusion.
    
    Fuses M modalities using Dempster-Shafer evidence combination
    with learned per-modality contextual discounting.
    
    Key equations:
      alpha_tilde_m = delta_m * (alpha_m - 1) + 1        (discounted evidence)
      alpha_fused_k = sum_m(alpha_tilde_m_k - 1) + 1     (DS combination)
      w_m = (S_m * delta_m) / sum(S_m' * delta_m')       (reliability weights)
      f_fused = sum_m(w_m * f_m)                          (weighted features)
    """
    
    def __init__(self, feat_dim=256, K=10, num_modalities=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.K = K
        self.num_modalities = num_modalities
        
        # One discounting network per modality
        self.discount_nets = nn.ModuleList([
            ContextualDiscountNet(feat_dim) for _ in range(num_modalities)
        ])
    
    def forward(self, features_list, alphas_list, uncertainties_list):
        """
        Args:
            features_list:     list of M tensors, each (B, feat_dim)
            alphas_list:       list of M tensors, each (B, K)
            uncertainties_list: list of M tensors, each (B, 1)
        
        Returns:
            f_fused:         (B, feat_dim) -- fused feature vector
            alpha_fused:     (B, K) -- fused Dirichlet concentrations
            u_fused:         (B, 1) -- fused uncertainty
            weights:         (B, M) -- per-modality reliability weights
            deltas:          (B, M) -- learned discounting factors
        """
        B = features_list[0].shape[0]
        M = self.num_modalities
        
        # Step 1: Compute discounting factors
        deltas = []
        for m in range(M):
            delta_m = self.discount_nets[m](features_list[m], uncertainties_list[m])
            deltas.append(delta_m)  # (B, 1)
        
        # Step 2: Discount evidence (alpha_tilde)
        alpha_tildes = []
        for m in range(M):
            # alpha_tilde_m = delta_m * (alpha_m - 1) + 1
            alpha_tilde_m = deltas[m] * (alphas_list[m] - 1.0) + 1.0  # (B, K)
            alpha_tildes.append(alpha_tilde_m)
        
        # Step 3: Dempster-Shafer combination
        # alpha_fused_k = sum_m(alpha_tilde_m_k - 1) + 1
        alpha_fused = torch.ones(B, self.K, device=features_list[0].device)
        for m in range(M):
            alpha_fused = alpha_fused + (alpha_tildes[m] - 1.0)
        
        # Fused uncertainty
        S_fused = alpha_fused.sum(dim=-1, keepdim=True)  # (B, 1)
        u_fused = self.K / S_fused  # (B, 1)
        
        # Step 4: Compute reliability weights for feature aggregation
        # w_m = (S_m * delta_m) / sum(S_m' * delta_m')
        reliability_scores = []
        for m in range(M):
            S_m = alphas_list[m].sum(dim=-1, keepdim=True)  # (B, 1)
            score_m = S_m * deltas[m]  # (B, 1)
            reliability_scores.append(score_m)
        
        reliability_stack = torch.cat(reliability_scores, dim=-1)  # (B, M)
        weights = reliability_stack / (reliability_stack.sum(dim=-1, keepdim=True) + 1e-8)  # (B, M)
        
        # Step 5: Weighted feature aggregation
        # f_fused = sum_m(w_m * f_m)
        f_fused = torch.zeros(B, self.feat_dim, device=features_list[0].device)
        for m in range(M):
            f_fused = f_fused + weights[:, m:m+1] * features_list[m]  # (B, feat_dim)
        
        # Stack deltas for output
        deltas_stack = torch.cat(deltas, dim=-1)  # (B, M)
        
        return f_fused, alpha_fused, u_fused, weights, deltas_stack


# ---------------------------------------------------------------------------
# EUGF Evidential Regularization Loss
# ---------------------------------------------------------------------------

def eugf_regularization_loss(alpha, target_class=None, K=10):
    """
    KL divergence between predicted Dirichlet and uniform Dirichlet (alpha=1).
    Prevents evidence collapse by penalizing overconfident predictions
    on incorrect classes.
    
    For regression (no discrete classes), we use a simplified version
    that penalizes very low total evidence (encourages the model to
    actually commit evidence).
    
    Args:
        alpha: (B, K) Dirichlet concentrations
        target_class: optional (B,) class indices for classification
        K: number of pseudo-categories
    
    Returns:
        loss: scalar
    """
    S = alpha.sum(dim=-1, keepdim=True)  # (B, 1)
    
    if target_class is not None:
        # Classification mode: penalize evidence on wrong classes
        one_hot = F.one_hot(target_class, K).float()
        alpha_tilde = alpha - one_hot  # remove evidence from true class
        alpha_tilde = torch.clamp(alpha_tilde, min=1.0)
    else:
        alpha_tilde = alpha
    
    # KL(Dir(alpha_tilde) || Dir(1, ..., 1))
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(K)))
          - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
          + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) 
             - torch.digamma(S_tilde))).sum(dim=-1, keepdim=True))
    
    return kl.mean()
