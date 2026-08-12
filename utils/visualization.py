"""
EcoFuse Publication-Grade Visualization
=========================================
Generates high-resolution (300 DPI) spatial maps for professor presentation
and eventual journal submission.

Figures:
  1. Multi-Modal Inputs & Biomass Map
  2. Spatial Modality Weight Shift
  3. Temporal Trajectory Consistency (DEcoDE)
  4. Learned Ecological Parameters (r, K, d)

Design:
  - 300 DPI, tight layout
  - High-contrast scientific colormaps (YlGn, magma, Blues/Oranges/Greens)
  - Clear titles, axis labels, and colorbars with units
  - Professional font sizing
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Style Configuration
# ---------------------------------------------------------------------------

def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
    })


# ---------------------------------------------------------------------------
# Figure 1: Multi-Modal Inputs & Biomass Map
# ---------------------------------------------------------------------------

def plot_multimodal_biomass(optical, sar, gedi, agb_pred, uncertainty,
                            output_path="fig1_multimodal_biomass.png"):
    """
    5-panel figure showing all input modalities and model outputs.
    
    Args:
        optical:     (H, W, 6) normalized S2 bands (uses B4,B3,B2 for RGB)
        sar:         (H, W, 3) normalized S1 channels (shows VV)
        gedi:        (H, W, 1) normalized GEDI AGBD
        agb_pred:    (H, W) predicted AGB (normalized 0-1)
        uncertainty: (H, W) EUGF predictive uncertainty
        output_path: save path
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Panel 1: Sentinel-2 RGB (B4=Red, B3=Green, B2=Blue -> indices 2,1,0)
    rgb = np.clip(optical[:, :, [2, 1, 0]] * 2.5, 0, 1)  # brightness boost
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB", fontweight='bold')
    axes[0].set_xlabel("Optical (B4/B3/B2)")
    
    # Panel 2: Sentinel-1 SAR (VV band)
    im_sar = axes[1].imshow(sar[:, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Sentinel-1 SAR (VV)", fontweight='bold')
    axes[1].set_xlabel("Backscatter (norm.)")
    plt.colorbar(im_sar, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel 3: GEDI Ground Truth AGBD
    im_gedi = axes[2].imshow(gedi[:, :, 0], cmap='YlGn', vmin=0, vmax=1)
    axes[2].set_title("GEDI AGBD", fontweight='bold')
    axes[2].set_xlabel("Biomass (norm.)")
    plt.colorbar(im_gedi, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Panel 4: Predicted AGB
    im_agb = axes[3].imshow(agb_pred, cmap='YlGn', vmin=0, vmax=1)
    axes[3].set_title("Predicted AGB", fontweight='bold', color='darkgreen')
    axes[3].set_xlabel("Biomass (norm.)")
    plt.colorbar(im_agb, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Panel 5: EUGF Uncertainty
    im_unc = axes[4].imshow(uncertainty, cmap='magma', vmin=0)
    axes[4].set_title("EUGF Uncertainty", fontweight='bold', color='darkred')
    axes[4].set_xlabel("Uncertainty (u)")
    plt.colorbar(im_unc, ax=axes[4], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle("EcoFuse: Multi-Modal Inputs & Biomass Prediction  |  Sundarbans ROI",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[FIGURE] Saved: %s" % output_path)


# ---------------------------------------------------------------------------
# Figure 2: Spatial Modality Weight Shift
# ---------------------------------------------------------------------------

def plot_modality_weights(w_opt, w_sar, w_gedi,
                          output_path="fig2_modality_weights.png"):
    """
    3-panel figure showing per-pixel reliability weights from EUGF.
    Demonstrates automatic optical suppression under degraded conditions.
    
    Args:
        w_opt:  (H, W) optical weight
        w_sar:  (H, W) SAR weight
        w_gedi: (H, W) GEDI weight
        output_path: save path
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    configs = [
        (w_opt, "Optical Weight ($w_{opt}$)", 'Blues', 'navy'),
        (w_sar, "SAR Weight ($w_{sar}$)", 'Oranges', 'darkorange'),
        (w_gedi, "GEDI Weight ($w_{gedi}$)", 'Greens', 'darkgreen'),
    ]
    
    for ax, (data, title, cmap, color) in zip(axes, configs):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontweight='bold', color=color, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.suptitle("EUGF: Learned Modality Reliability Weights  |  Sundarbans",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[FIGURE] Saved: %s" % output_path)


# ---------------------------------------------------------------------------
# Figure 3: Temporal Trajectory Consistency (DEcoDE)
# ---------------------------------------------------------------------------

def plot_temporal_trajectory(B_spatial_series, B_ode_series, time_labels=None,
                              output_path="fig3_temporal_trajectory.png"):
    """
    Line plot comparing spatial encoder predictions vs. DEcoDE ODE trajectory.
    Shows how the ODE smooths out temporal spikes.
    
    Args:
        B_spatial_series: list/array of spatial AGB predictions over time
        B_ode_series:     list/array of ODE-smoothed AGB predictions
        time_labels:      optional list of time labels (e.g., months)
        output_path: save path
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    t = np.arange(len(B_spatial_series))
    if time_labels is None:
        time_labels = ["T%d" % i for i in t]
    
    ax.plot(t, B_spatial_series, 'o-', color='#E74C3C', linewidth=2,
            markersize=8, label='Spatial Encoder (unconstrained)', alpha=0.8)
    ax.plot(t, B_ode_series, 's-', color='#2ECC71', linewidth=2.5,
            markersize=8, label='DEcoDE ODE (ecologically constrained)', alpha=0.9)
    
    # Shade the gap
    ax.fill_between(t, B_spatial_series, B_ode_series,
                    alpha=0.15, color='gray', label='Temporal inconsistency')
    
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Above-Ground Biomass (Mg/ha)", fontsize=12)
    ax.set_title("DEcoDE: Temporal Trajectory Consistency\n"
                 "ODE-constrained vs. Unconstrained Predictions",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(t)
    ax.set_xticklabels(time_labels)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[FIGURE] Saved: %s" % output_path)


# ---------------------------------------------------------------------------
# Figure 4: Learned Ecological Parameters
# ---------------------------------------------------------------------------

def plot_ecological_params(r_val, K_val, d_val,
                            output_path="fig4_ecological_params.png"):
    """
    Bar chart comparing learned (r, K, d) values against published
    mangrove literature bounds.
    
    Reference bounds:
      r: 0.01-0.15 yr^-1  (Komiyama et al., 2008)
      K: 150-400 Mg/ha     (Simard et al., 2019)
      d: 0.01-0.10 yr^-1   (general mangrove mortality)
    
    Args:
        r_val: learned growth rate
        K_val: learned carrying capacity
        d_val: learned mortality rate
        output_path: save path
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    params = [
        ('Growth Rate $r$\n(yr$^{-1}$)', r_val, 0.01, 0.15, '#3498DB'),
        ('Carrying Capacity $K$\n(Mg/ha)', K_val, 150, 400, '#2ECC71'),
        ('Mortality Rate $d$\n(yr$^{-1}$)', d_val, 0.01, 0.10, '#E74C3C'),
    ]
    
    for ax, (label, val, lo, hi, color) in zip(axes, params):
        # Literature range bar
        ax.barh(['Literature\nRange'], [hi - lo], left=[lo],
                color=color, alpha=0.25, height=0.4,
                edgecolor=color, linewidth=1.5, label='Published range')
        
        # Learned value marker
        ax.plot(val, ['Learned\n(EcoFuse)'], 'D', color=color,
                markersize=14, markeredgecolor='black', markeredgewidth=1.5,
                label='Learned value')
        
        ax.set_xlabel(label, fontsize=11)
        ax.axvline(x=lo, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=hi, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2, axis='x')
    
    fig.suptitle("DEcoDE: Learned Ecological Parameters vs. Literature Bounds\n"
                 "Sundarbans Mangrove  |  Climate-Conditioned MLPs",
                 fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[FIGURE] Saved: %s" % output_path)


# ---------------------------------------------------------------------------
# Training Loss Curve
# ---------------------------------------------------------------------------

def plot_training_loss(loss_history, output_path="fig5_training_loss.png"):
    """
    Training loss curve showing convergence of L_total and components.
    
    Args:
        loss_history: list of dicts with keys 'total','focal','rmse','eugf','ode'
        output_path: save path
    """
    set_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(loss_history) + 1)
    totals = [h['total'] for h in loss_history]
    
    # Left: Total loss
    ax1.plot(epochs, totals, '-', color='#2C3E50', linewidth=2.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Total Loss ($L_{total}$)", fontsize=12)
    ax1.set_title("Training Convergence", fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Component losses
    components = {
        'L_Focal': ([h['focal'] for h in loss_history], '#E74C3C'),
        'L_RMSE': ([h['rmse'] for h in loss_history], '#3498DB'),
        'L_EUGF': ([h['eugf'] for h in loss_history], '#F39C12'),
        'L_ODE': ([h['ode'] for h in loss_history], '#2ECC71'),
    }
    
    for name, (vals, color) in components.items():
        ax2.plot(epochs, vals, '-', color=color, linewidth=2, label=name)
    
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Component Loss", fontsize=12)
    ax2.set_title("Loss Components", fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle("EcoFuse Training  |  Joint Multi-Task Loss",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[FIGURE] Saved: %s" % output_path)
