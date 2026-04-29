"""
XGBoost Land Use / Land Cover (LULC) Classifier.

CORRECT APPROACH: Cross-temporal classification.
  - Features: temporal statistics from PAST years (mean, std, slope,
    delta of NDVI/NDBI/NDWI across years 2018–2024)
  - Labels:   land cover class from the LATEST year (2025/2026),
              derived from spectral thresholds

This avoids the circular problem of training on the same data used
to generate labels. The model must LEARN temporal patterns to predict
current land cover — a genuine ML task.

Produces all 8 classification metrics:
    Confusion Matrix, F1-Score, Precision, Recall,
    Sensitivity, Accuracy, AUC, Error Rate.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Vegetation", "Urban", "Water", "Barren"]
CLASS_COLORS_RGB = [(46, 125, 50), (230, 81, 0), (21, 101, 192), (121, 85, 72)]
CLASS_COLORS_HEX = ["#2E7D32", "#E65100", "#1565C0", "#795548"]


def _get_xgb():
    """Lazy-import XGBoost; fall back to sklearn GradientBoosting."""
    try:
        from xgboost import XGBClassifier
        logger.info("Using XGBoost classifier")
        return XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed — using sklearn GradientBoosting fallback")
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier


def generate_labels(ndvi: np.ndarray, ndbi: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
    """
    Generate ground-truth labels from spectral index thresholds.
    Same rules as unified_change_detector.py _classify_changes().
    Classes: 0=Vegetation, 1=Urban, 2=Water, 3=Barren
    """
    labels = np.full(ndvi.shape, 3, dtype=np.int32)  # default Barren
    labels[(ndwi > 0.0) & (ndvi < 0.25)] = 2         # Water
    labels[(ndbi > 0.0) & (ndvi < 0.25) & (labels != 2)] = 1  # Urban
    labels[ndvi > 0.3] = 0                            # Vegetation
    return labels


def _compute_temporal_features(
    ndvi_stack: np.ndarray,
    ndbi_stack: np.ndarray,
    ndwi_stack: np.ndarray,
    valid_stack: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel temporal statistics from a multi-year stack.

    Input shapes: [T, H, W] where T = number of past years.
    Output shape: [H*W, num_features]

    Features per pixel (18 total):
      NDVI:  mean, std, min, max, slope, last-first delta
      NDBI:  mean, std, min, max, slope, last-first delta
      NDWI:  mean, std, min, max, slope, last-first delta
    """
    T, H, W = ndvi_stack.shape
    N = H * W

    # Mask invalid pixels with NaN for correct statistics
    ndvi_masked = np.where(valid_stack, ndvi_stack, np.nan)
    ndbi_masked = np.where(valid_stack, ndbi_stack, np.nan)
    ndwi_masked = np.where(valid_stack, ndwi_stack, np.nan)

    features = []

    for stack in [ndvi_masked, ndbi_masked, ndwi_masked]:
        # Reshape to [T, N] for vectorized computation
        flat = stack.reshape(T, N)

        with np.errstate(all='ignore'):
            feat_mean = np.nanmean(flat, axis=0)
            feat_std = np.nanstd(flat, axis=0)
            feat_min = np.nanmin(flat, axis=0)
            feat_max = np.nanmax(flat, axis=0)

            # Slope via least-squares (linear trend over time)
            x = np.arange(T, dtype=np.float32)
            x_mean = x.mean()
            x_var = np.sum((x - x_mean) ** 2)
            if x_var > 0:
                slope = np.nansum((x[:, None] - x_mean) * (flat - feat_mean[None, :]), axis=0) / x_var
            else:
                slope = np.zeros(N, dtype=np.float32)

            # Delta: last year minus first year
            delta = flat[-1] - flat[0]

        # Replace NaN with 0
        for arr in [feat_mean, feat_std, feat_min, feat_max, slope, delta]:
            arr[np.isnan(arr)] = 0.0

        features.extend([feat_mean, feat_std, feat_min, feat_max, slope, delta])

    # Stack: [N, 18]
    return np.column_stack(features).astype(np.float32)


FEATURE_NAMES = [
    "NDVI_mean", "NDVI_std", "NDVI_min", "NDVI_max", "NDVI_slope", "NDVI_delta",
    "NDBI_mean", "NDBI_std", "NDBI_min", "NDBI_max", "NDBI_slope", "NDBI_delta",
    "NDWI_mean", "NDWI_std", "NDWI_min", "NDWI_max", "NDWI_slope", "NDWI_delta",
]


def classify_multiyear(
    yearly_series: list,
) -> Dict:
    """
    Run XGBoost LULC classification using cross-temporal features.

    Args:
        yearly_series: List of YearlyData objects (multiple years).
                       Last entry is used for ground-truth labels.
                       Earlier entries are used to build temporal features.

    Returns:
        dict with classification map, overlay image, and all 8 metrics.
    """
    if len(yearly_series) < 3:
        return {"error": "Need at least 3 years of data for temporal classification"}

    # Split: past years → features, last year → labels
    past = yearly_series[:-1]
    target = yearly_series[-1]

    H, W = target.ndvi.shape

    # ── Build temporal feature stack from past years ──
    ndvi_stack = np.stack([y.ndvi for y in past], axis=0)
    ndbi_stack = np.stack([y.ndbi for y in past], axis=0)
    ndwi_stack = np.stack([y.ndwi for y in past], axis=0)
    valid_stack = np.stack([y.valid_mask for y in past], axis=0)

    logger.info("Building temporal features from %d past years → predicting year %d",
                len(past), target.year)

    X_all = _compute_temporal_features(ndvi_stack, ndbi_stack, ndwi_stack, valid_stack)

    # ── Generate ground-truth labels from the LATEST year ──
    labels_gt = generate_labels(target.ndvi, target.ndbi, target.ndwi)
    y_all = labels_gt.ravel()

    # ── Only use pixels valid in the target year ──
    valid_target = target.valid_mask.ravel()
    # Also require that the pixel had at least 2 valid past observations
    past_valid_count = np.sum(valid_stack.reshape(len(past), -1), axis=0)
    usable = valid_target & (past_valid_count >= 2)

    X_valid = X_all[usable]
    y_valid = y_all[usable]

    if len(X_valid) < 200:
        return {"error": f"Too few valid pixels ({len(X_valid)}) for classification"}

    logger.info("Valid pixels for classification: %d (%.1f%%)",
                len(X_valid), len(X_valid) / (H * W) * 100)

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=0.3, random_state=42, stratify=y_valid
    )

    # ── Train XGBoost ──
    XGBCls = _get_xgb()
    try:
        model = XGBCls(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=42,
            use_label_encoder=False,
        )
    except TypeError:
        # Fallback for sklearn GradientBoosting (different API)
        model = XGBCls(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)
    logger.info("XGBoost training complete — %d estimators, depth %d", 150, 5)

    # ── Predictions ──
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = None

    # ═══════════════════════════════════════════════════════════════
    # Compute all 8 metrics
    # ═══════════════════════════════════════════════════════════════

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    acc = float(accuracy_score(y_test, y_pred))
    error_rate = 1.0 - acc

    prec_per = precision_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    rec_per = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    f1_per = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)

    prec_w = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec_w = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1_w = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    # AUC (One-vs-Rest)
    auc_per = [0.0, 0.0, 0.0, 0.0]
    auc_w = 0.0
    if y_prob is not None and y_prob.shape[1] == 4:
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
        for i in range(4):
            try:
                auc_per[i] = float(roc_auc_score(y_test_bin[:, i], y_prob[:, i]))
            except ValueError:
                auc_per[i] = 0.0
        auc_w = float(np.mean(auc_per))

    # Feature importance
    try:
        importance = model.feature_importances_.tolist()
    except Exception:
        importance = [1.0 / len(FEATURE_NAMES)] * len(FEATURE_NAMES)

    # ── Build classification map (predict ALL pixels) ──
    pred_full = model.predict(X_all).reshape(H, W)

    color_map = np.zeros((H, W, 4), dtype=np.uint8)
    for cls_id, rgb in enumerate(CLASS_COLORS_RGB):
        mask_cls = pred_full == cls_id
        color_map[mask_cls, 0] = rgb[0]
        color_map[mask_cls, 1] = rgb[1]
        color_map[mask_cls, 2] = rgb[2]
        color_map[mask_cls, 3] = 180

    # Encode as base64 PNG
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(color_map, "RGBA").save(buf, format="PNG")
    overlay_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Class distribution
    class_dist = {}
    total_px = float(pred_full.size)
    for i, name in enumerate(CLASS_NAMES):
        count = int(np.sum(pred_full == i))
        class_dist[name] = {
            "pixels": count,
            "percent": round((count / total_px) * 100, 2),
            "color": CLASS_COLORS_HEX[i],
        }

    # Group feature importance by spectral index
    fi_grouped = {}
    for idx, fname in enumerate(FEATURE_NAMES):
        if idx < len(importance):
            base = fname.split("_")[0]  # NDVI, NDBI, NDWI
            fi_grouped[base] = fi_grouped.get(base, 0.0) + importance[idx]

    # Normalize
    fi_total = sum(fi_grouped.values()) or 1.0
    fi_grouped = {k: round(v / fi_total, 4) for k, v in fi_grouped.items()}

    return {
        "status": "success",
        "model": "XGBoost",
        "approach": "Cross-temporal: features from past years → predict current land cover",
        "years_used_for_features": [y.year for y in past],
        "target_year": target.year,
        "overlay": overlay_b64,
        "class_distribution": class_dist,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "feature_importance": fi_grouped,
        "metrics": {
            "accuracy": round(acc, 4),
            "error_rate": round(error_rate, 4),
            "f1_score": {"per_class": [round(v, 4) for v in f1_per], "weighted": round(f1_w, 4)},
            "precision": {"per_class": [round(v, 4) for v in prec_per], "weighted": round(prec_w, 4)},
            "recall": {"per_class": [round(v, 4) for v in rec_per], "weighted": round(rec_w, 4)},
            "sensitivity": {"per_class": [round(v, 4) for v in rec_per], "weighted": round(rec_w, 4)},
            "auc": {"per_class": [round(v, 4) for v in auc_per], "weighted": round(auc_w, 4)},
            "confusion_matrix": cm.tolist(),
        },
        "class_names": CLASS_NAMES,
    }
