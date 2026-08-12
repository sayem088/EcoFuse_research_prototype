# EcoFuse_Prototype
<!-- # EcoFuse: Uncertainty-Aware Multi-Modal Fusion with Differentiable Ecological Dynamics

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-API-green.svg)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **EcoFuse**, a physics-informed deep learning architecture designed for coastal biomass monitoring (mangroves, salt marshes, seagrasses) by integrating multi-modal Earth Observation (EO) data with differentiable ecological differential equations.

---

## 📌 Key Novelties

1. **Evidential Uncertainty-Gated Fusion (EUGF)**:
   - Parametrizes each sensor encoder output with Dirichlet distribution parameters ($\boldsymbol{\alpha}_m$).
   - Computes explicit epistemic uncertainty ($u_m = K / S_m$) and dynamically weights inputs based on learned sensor reliability.
   - Gated context-discounting down-weights cloud-contaminated optical signals while preserving high-backscatter SAR and GEDI LiDAR signals.

2. **Differentiable Ecological Dynamics (DEcoDE) Layer**:
   - Embeds a non-linear modified logistic growth differential equation ($\frac{dB}{dt} = r B \left(1 - \frac{B}{K}\right) - d B$) into PyTorch using `torchdiffeq` RK4 ODE solvers.
   - Growth parameters ($r, K, d$) are dynamically predicted by climate-conditioned MLPs driven by ERA5-Land reanalysis vectors (temperature, precipitation, soil moisture).
   - Enforces biological trajectory consistency, preventing unphysical biomass jumps across temporal observations.

---

## 📂 Repository Structure

```
EcoFuse_Prototype/
├── main.py              # Master CLI entrypoint (GEE -> Train -> Figure Generation)
├── train.py             # Training pipeline with mini-batch optimization
├── requirements.txt     # Pinned dependencies
├── .gitignore           # Git ignore file
├── data/
│   ├── __init__.py
│   └── gee_ingest.py    # Multi-modal GEE extraction (Sentinel-1, Sentinel-2, ERA5, GEDI)
├── models/
│   ├── __init__.py
│   ├── encoders.py      # ResNet18 + MLP encoders + Dirichlet evidence heads
│   ├── eugf.py          # Evidential Uncertainty-Gated Fusion module
│   ├── decode_ode.py    # Differentiable Ecological ODE solver
│   └── ecofuse.py       # Full EcoFuse model assembly & joint multi-task loss
├── utils/
│   ├── __init__.py
│   ├── metrics.py       # R², RMSE, MAE, rRMSE, ECE metrics
│   └── visualization.py # 300 DPI publication-grade figure renderers
└── outputs/
    ├── figures/          # Generated evaluation figures (1-5)
    └── checkpoints/      # Saved PyTorch model checkpoints (.pt)
```

---

## 🚀 Quick Start Guide

### 1. Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/EcoFuse_Prototype.git
cd EcoFuse_Prototype

# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine (run once)
earthengine authenticate
```

### 2. Running the Prototype Pipeline

Run the full end-to-end pipeline (Data Ingestion $\rightarrow$ Model Training $\rightarrow$ Figure Generation):

```bash
python main.py --epochs 50
```

To run training without re-downloading GEE data:
```bash
python main.py --skip-gee --epochs 50
```

---

## 💻 Running in Google Colab / Kaggle Notebook

To execute this project on Google Colab or Kaggle:

1. **Upload Code**: Upload `EcoFuse_Prototype.zip` or clone your GitHub repository in a notebook cell:
   ```python
   !git clone https://github.com/YOUR_USERNAME/EcoFuse_Prototype.git
   %cd EcoFuse_Prototype
   ```

2. **Install Dependencies**:
   ```python
   !pip install -r requirements.txt
   ```

3. **Authenticate Earth Engine**:
   ```python
   import ee
   ee.Authenticate()
   ee.Initialize(project='YOUR_GEE_PROJECT_ID')
   ```

4. **Run Master Pipeline**:
   ```python
   !python main.py --epochs 50
   ```

---

## 📊 Outputs & Visualizations

After execution, high-resolution (300 DPI) publication figures are saved to `outputs/figures/`:

- `fig1_multimodal_biomass.png`: Multi-modal satellite inputs (Sentinel-2 RGB, Sentinel-1 SAR VV, GEDI AGBD), predicted biomass map, and EUGF epistemic uncertainty.
- `fig2_modality_weights.png`: Spatially varying modality reliability weights ($w_{\text{opt}}, w_{\text{sar}}, w_{\text{gedi}}$).
- `fig3_temporal_trajectory.png`: Temporal trajectory comparison between unconstrained spatial encoder predictions vs. ecologically constrained DEcoDE trajectories.
- `fig4_ecological_params.png`: Learned ecological growth parameters ($r, K, d$) vs. literature bounds for mangrove ecosystems.
- `fig5_training_loss.png`: Training loss convergence curve across joint multi-task loss components.

---

## 📜 License & Citation

Distributed under the MIT License. See `LICENSE` for more information.

```bibtex
@article{ecofuse2026,
  title={EcoFuse: Uncertainty-Aware Multi-Modal Fusion with Differentiable Ecological Dynamics for Coastal Biomass Monitoring},
  author={EcoFuse Research Team},
  journal={Remote Sensing of Environment (Target)},
  year={2026}
}
``` -->
