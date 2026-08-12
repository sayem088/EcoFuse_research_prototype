"""
EcoFuse Training Pipeline
============================
Unified training loop that:
  1. Loads real Sundarbans dataset from .npz
  2. Creates training batches from 64x64 patches
  3. Trains EcoFuse model with joint multi-task loss
  4. Logs per-epoch loss components
  5. Saves checkpoints and returns loss history

Runs entirely on CPU for prototype (Kaggle GPU for scaling).
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ecofuse import EcoFuse, EcoFuseLoss
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_dataset(npz_path):
    """
    Load the Sundarbans dataset from .npz file.
    
    Returns tensors ready for model input:
        optical:     (N, 6, H, W)
        sar:         (N, 3, H, W)
        gedi:        (N, 1)
        climate_vec: (N, 3)
        agb_target:  (N, 1)
    """
    print("[LOAD] Loading dataset: %s" % npz_path)
    data = np.load(npz_path)
    
    # Original format: (H, W, C) -> need (N, C, H, W) for PyTorch
    optical = data['optical']     # (64, 64, 6)
    sar = data['sar']             # (64, 64, 3)
    gedi = data['gedi']           # (64, 64, 1)
    climate = data['climate_vec'] # (3,)
    
    H, W = optical.shape[0], optical.shape[1]
    
    # Create multiple training samples by extracting overlapping sub-patches
    # and augmenting with spatial variations
    patch_size = 32  # sub-patch size for training samples
    stride = 8       # overlap stride
    
    samples_opt = []
    samples_sar = []
    samples_gedi = []
    samples_agb = []
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            opt_patch = optical[i:i+patch_size, j:j+patch_size, :]
            sar_patch = sar[i:i+patch_size, j:j+patch_size, :]
            gedi_patch = gedi[i:i+patch_size, j:j+patch_size, :]
            
            samples_opt.append(opt_patch)
            samples_sar.append(sar_patch)
            # Spatially average GEDI for encoder input (sparse modality)
            samples_gedi.append(gedi_patch.mean())
            # AGB target = GEDI mean for this patch
            samples_agb.append(gedi_patch.mean())
    
    N = len(samples_opt)
    
    # Convert to PyTorch tensors: (N, C, H, W)
    optical_t = torch.tensor(np.array(samples_opt), dtype=torch.float32).permute(0, 3, 1, 2)
    sar_t = torch.tensor(np.array(samples_sar), dtype=torch.float32).permute(0, 3, 1, 2)
    gedi_t = torch.tensor(np.array(samples_gedi), dtype=torch.float32).unsqueeze(-1)
    climate_t = torch.tensor(np.tile(climate, (N, 1)), dtype=torch.float32)
    agb_t = torch.tensor(np.array(samples_agb), dtype=torch.float32).unsqueeze(-1)
    
    print("[LOAD] Created %d training samples (patch=%dx%d, stride=%d)" %
          (N, patch_size, patch_size, stride))
    print("  optical:  %s" % str(optical_t.shape))
    print("  sar:      %s" % str(sar_t.shape))
    print("  gedi:     %s" % str(gedi_t.shape))
    print("  climate:  %s" % str(climate_t.shape))
    print("  agb_tgt:  %s" % str(agb_t.shape))
    
    return optical_t, sar_t, gedi_t, climate_t, agb_t


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_ecofuse(dataset_path, num_epochs=50, batch_size=16, lr=1e-3,
                  checkpoint_dir="outputs/checkpoints",
                  device="cpu"):
    """
    Full training pipeline.
    
    Args:
        dataset_path:   path to sundarbans_dataset.npz
        num_epochs:     training epochs
        batch_size:     mini-batch size
        lr:             learning rate
        checkpoint_dir: where to save model weights
        device:         'cpu' or 'cuda'
    
    Returns:
        model:        trained EcoFuse model
        loss_history: list of dicts with per-epoch losses
        final_outputs: dict with final model predictions
    """
    print("=" * 60)
    print("EcoFuse Training Pipeline")
    print("=" * 60)
    
    # Load data
    optical, sar, gedi, climate, agb_target = load_dataset(dataset_path)
    N = optical.shape[0]
    
    # Initialize model
    model = EcoFuse(feat_dim=256, K=10, num_risk_classes=4, climate_dim=3)
    model = model.to(device)
    model.train()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[MODEL] Total parameters: %s" % format(total_params, ','))
    print("[MODEL] Trainable:        %s" % format(trainable, ','))
    
    # Loss and optimizer
    criterion = EcoFuseLoss(w1=1.0, w2=1.0, lambda1=0.1, lambda2=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    loss_history = []
    best_loss = float('inf')
    
    print("\n[TRAIN] Starting training: %d epochs, batch_size=%d, lr=%.4f" %
          (num_epochs, batch_size, lr))
    print("-" * 60)
    
    for epoch in range(1, num_epochs + 1):
        # Shuffle indices
        perm = torch.randperm(N)
        epoch_losses = {'total': 0, 'focal': 0, 'rmse': 0, 'eugf': 0, 'ode': 0}
        n_batches = 0
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            
            b_opt = optical[idx].to(device)
            b_sar = sar[idx].to(device)
            b_gedi = gedi[idx].to(device)
            b_climate = climate[idx].to(device)
            b_agb = agb_target[idx].to(device)
            
            # Forward
            outputs = model(b_opt, b_sar, b_gedi, b_climate)
            loss, loss_dict = criterion(outputs, b_agb)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]
            n_batches += 1
        
        scheduler.step()
        
        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        
        loss_history.append(epoch_losses)
        
        # Logging
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print("  Epoch %3d/%d | Loss: %.4f | Focal: %.4f | RMSE: %.4f | "
                  "EUGF: %.4f | ODE: %.4f | LR: %.6f" %
                  (epoch, num_epochs, epoch_losses['total'],
                   epoch_losses['focal'], epoch_losses['rmse'],
                   epoch_losses['eugf'], epoch_losses['ode'], lr_now))
        
        # Save best checkpoint
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, "ecofuse_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, ckpt_path)
    
    print("-" * 60)
    print("[TRAIN] Training complete. Best loss: %.4f" % best_loss)
    
    # Final evaluation pass
    model.eval()
    with torch.no_grad():
        final_out = model(optical.to(device), sar.to(device),
                          gedi.to(device), climate.to(device))
    
    # Compute metrics
    agb_pred_np = final_out['agb_pred'].cpu().numpy().flatten()
    agb_true_np = agb_target.numpy().flatten()
    unc_np = final_out['uncertainty'].cpu().numpy().flatten()
    
    metrics = compute_all_metrics(agb_true_np, agb_pred_np, unc_np)
    
    print("\n[METRICS] Final Evaluation:")
    for k, v in metrics.items():
        print("  %-12s: %.4f" % (k, v))
    
    return model, loss_history, final_out, metrics
