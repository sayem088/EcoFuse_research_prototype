"""
EcoFuse Evaluation Metrics
============================
Quantitative metrics for model evaluation per proposal Section 9.4:

Regression:  R², RMSE, MAE, rRMSE
Segmentation: OA, F1, mIoU
Uncertainty:  ECE, AUSE, NLL
"""

import numpy as np


# ---------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    """Root Mean Squared Error (Mg/ha)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error (Mg/ha)."""
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
    return 1.0 - ss_res / ss_tot


def relative_rmse(y_true, y_pred):
    """Relative RMSE (%) = RMSE / mean(y_true) * 100."""
    return rmse(y_true, y_pred) / (np.mean(y_true) + 1e-8) * 100.0


# ---------------------------------------------------------------------------
# Classification / Segmentation Metrics
# ---------------------------------------------------------------------------

def overall_accuracy(y_true, y_pred):
    """Overall Accuracy for multi-class classification."""
    return np.mean(y_true == y_pred)


def f1_score_macro(y_true, y_pred, num_classes=4):
    """Macro-averaged F1 score across all classes."""
    f1_per_class = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1)
    return np.mean(f1_per_class)


def mean_iou(y_true, y_pred, num_classes=4):
    """Mean Intersection-over-Union across all classes."""
    iou_per_class = []
    for c in range(num_classes):
        intersection = np.sum((y_pred == c) & (y_true == c))
        union = np.sum((y_pred == c) | (y_true == c))
        iou = intersection / (union + 1e-8)
        iou_per_class.append(iou)
    return np.mean(iou_per_class)


# ---------------------------------------------------------------------------
# Uncertainty Calibration Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_pred, uncertainty, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Bins predictions by uncertainty and checks if confidence matches accuracy.
    Lower is better (perfectly calibrated = 0).
    """
    # For regression: bin by uncertainty, check if error correlates
    errors = np.abs(y_true - y_pred)
    bin_edges = np.linspace(uncertainty.min(), uncertainty.max() + 1e-8, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    
    for i in range(n_bins):
        mask = (uncertainty >= bin_edges[i]) & (uncertainty < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_error = errors[mask].mean()
        bin_uncertainty = uncertainty[mask].mean()
        bin_weight = mask.sum() / total
        ece += bin_weight * np.abs(bin_error - bin_uncertainty)
    
    return ece


def compute_all_metrics(y_true, y_pred, uncertainty=None):
    """
    Compute all regression + uncertainty metrics and return as dict.
    
    Args:
        y_true: (N,) ground truth AGB
        y_pred: (N,) predicted AGB
        uncertainty: (N,) optional predicted uncertainty
    
    Returns:
        dict of metric_name -> value
    """
    metrics = {
        'R2': r_squared(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'rRMSE_pct': relative_rmse(y_true, y_pred),
    }
    
    if uncertainty is not None:
        metrics['ECE'] = expected_calibration_error(y_true, y_pred, uncertainty)
    
    return metrics
