"""
EcoFuse Modality-Specific Encoders
====================================
Three separate encoders extract features and Dirichlet evidence parameters
from each input modality:

  1. OpticalEncoder  - Sentinel-2 (6 bands)  -> ResNet-style CNN
  2. SAREncoder      - Sentinel-1 (3 channels) -> ResNet-style CNN
  3. GEDIEncoder     - GEDI AGBD (1 channel)  -> 3-layer MLP

Each encoder outputs:
  - features:   (B, feat_dim)           -- learned feature embedding
  - alpha:      (B, K)                  -- Dirichlet concentration parameters
  - uncertainty: (B, 1)                 -- per-sample uncertainty u = K / S
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with two 3x3 convolutions and skip connection."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class DirichletEvidenceHead(nn.Module):
    """
    Maps feature vector to Dirichlet concentration parameters alpha.
    alpha = Softplus(linear(features)) + 1
    
    Ensures alpha > 1 so the Dirichlet distribution is unimodal,
    and uncertainty u = K / S where S = sum(alpha).
    """
    def __init__(self, feat_dim, K=10):
        super().__init__()
        self.K = K  # number of evidence classes (pseudo-categories)
        self.evidence_layer = nn.Linear(feat_dim, K)
    
    def forward(self, features):
        """
        Args:
            features: (B, feat_dim)
        Returns:
            alpha: (B, K)  -- Dirichlet concentrations
            uncertainty: (B, 1)  -- u = K / S
        """
        evidence = F.softplus(self.evidence_layer(features))  # (B, K), all > 0
        alpha = evidence + 1.0  # ensure alpha > 1
        S = alpha.sum(dim=-1, keepdim=True)  # (B, 1) Dirichlet strength
        uncertainty = self.K / S  # (B, 1) higher = more uncertain
        return alpha, uncertainty


# ---------------------------------------------------------------------------
# Optical Encoder (Sentinel-2, 6 bands)
# ---------------------------------------------------------------------------

class OpticalEncoder(nn.Module):
    """
    ResNet-style encoder for 6-band Sentinel-2 optical imagery.
    Architecture: 6 -> 64 -> 128 -> 256 -> GlobalAvgPool -> feat_dim
    """
    def __init__(self, in_channels=6, feat_dim=256, K=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feat_dim)
        self.evidence_head = DirichletEvidenceHead(feat_dim, K)
    
    def forward(self, x):
        """
        Args:
            x: (B, 6, H, W) optical raster
        Returns:
            features: (B, feat_dim)
            alpha: (B, K)
            uncertainty: (B, 1)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)  # (B, 256)
        features = F.relu(self.fc(x))  # (B, feat_dim)
        alpha, uncertainty = self.evidence_head(features)
        return features, alpha, uncertainty


# ---------------------------------------------------------------------------
# SAR Encoder (Sentinel-1, 3 channels)
# ---------------------------------------------------------------------------

class SAREncoder(nn.Module):
    """
    ResNet-style encoder for 3-channel Sentinel-1 SAR imagery.
    Architecture: 3 -> 64 -> 128 -> 256 -> GlobalAvgPool -> feat_dim
    """
    def __init__(self, in_channels=3, feat_dim=256, K=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, feat_dim)
        self.evidence_head = DirichletEvidenceHead(feat_dim, K)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) SAR raster
        Returns:
            features: (B, feat_dim)
            alpha: (B, K)
            uncertainty: (B, 1)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        features = F.relu(self.fc(x))
        alpha, uncertainty = self.evidence_head(features)
        return features, alpha, uncertainty


# ---------------------------------------------------------------------------
# GEDI Encoder (LiDAR, 1 channel - sparse footprint)
# ---------------------------------------------------------------------------

class GEDIEncoder(nn.Module):
    """
    3-layer MLP encoder for GEDI AGBD values.
    Takes spatially-averaged GEDI input: (B, 1) or (B, 2) if RH95 included.
    Architecture: in_dim -> 64 -> 128 -> feat_dim
    """
    def __init__(self, in_dim=1, feat_dim=256, K=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
        )
        self.evidence_head = DirichletEvidenceHead(feat_dim, K)
    
    def forward(self, x):
        """
        Args:
            x: (B, in_dim) GEDI AGBD scalar(s)
        Returns:
            features: (B, feat_dim)
            alpha: (B, K)
            uncertainty: (B, 1)
        """
        features = self.mlp(x)
        alpha, uncertainty = self.evidence_head(features)
        return features, alpha, uncertainty
