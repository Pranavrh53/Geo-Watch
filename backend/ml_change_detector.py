"""
ML Change Detection Pipeline — ChangeFormer (DL) + Spectral Indices
====================================================================

This is the **core ML/DL module** of the project.

Pipeline:
  ┌────────────────────────────────────────────────────────────────┐
  │  Step 1: IMAGE PREPROCESSING                                   │
  │    • Load before/after satellite images                        │
  │    • Resize, normalize to ImageNet stats                       │
  │    • Prepare tensors for GPU inference                         │
  ├────────────────────────────────────────────────────────────────┤
  │  Step 2: DEEP LEARNING — ChangeFormer (WHERE did change occur?)│
  │    • Siamese MiT-b1 Transformer encoder (pretrained ImageNet)  │
  │    • Multi-scale feature extraction at 4 resolutions           │
  │    • Absolute-difference fusion + MLP decoder                  │
  │    • Output: pixel-level change probability map [0, 1]         │
  ├────────────────────────────────────────────────────────────────┤
  │  Step 3: SPECTRAL INDICES (WHAT type of change?)               │
  │    • Fetch NIR (B08) + SWIR (B11) via Sentinel-2 Process API   │
  │    • NDVI = (NIR−Red)/(NIR+Red)   → vegetation index           │
  │    • NDBI = (SWIR−NIR)/(SWIR+NIR) → built-up index             │
  │    • MNDWI ≈ proxy from NDVI+NDBI → water bodies               │
  │    • Compute temporal differences: dNDVI, dNDBI                 │
  ├────────────────────────────────────────────────────────────────┤
  │  Step 4: CLASSIFICATION (Combine DL + Spectral)                │
  │    • ML change mask × spectral indices → classified labels      │
  │    • Categories:                                                │
  │      🔴 Deforestation (forest → built-up)                      │
  │      🟠 New Construction (NDBI rise in changed areas)          │
  │      🔵 Water Body Change (water gain/loss)                    │
  │      🟤 Agricultural Change (crop/farmland shift)              │
  │      ⚫ Road Development (linear built-up patterns)            │
  │      🟢 Vegetation Recovery (NDVI gain)                        │
  ├────────────────────────────────────────────────────────────────┤
  │  Step 5: POST-PROCESSING & VISUALISATION                       │
  │    • Morphological cleanup (remove noise)                      │
  │    • Connected component analysis (individual zones)           │
  │    • Colour-coded overlay generation                           │
  │    • Area statistics (hectares, acres, km²)                    │
  └────────────────────────────────────────────────────────────────┘

Models used:
  • ChangeFormer (MiT-b1 backbone)  — ~13.7M params, pretrained ImageNet
  • Spectral feature engineering     — NDVI, NDBI from Sentinel-2 bands

Why both:
  • DL alone detects change but can't classify WHAT changed
  • Spectral alone classifies type but is noisy at pixel boundaries
  • Combined: DL provides clean change boundaries, spectral classifies type
"""

import io
import os
import sys
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalisation stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ════════════════════════════════════════════════════════════
#  IMAGE PREPROCESSING
# ════════════════════════════════════════════════════════════

def preprocess_for_model(
    image: Image.Image,
    target_size: int = 512,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Preprocess a PIL satellite image for ChangeFormer inference.

    Steps:
      1. Resize to (target_size, target_size)
      2. Convert to float tensor [0, 1]
      3. Normalize with ImageNet mean/std (MiT-b1 was pretrained on ImageNet)
      4. Add batch dimension → [1, 3, H, W]
    """
    img = image.convert("RGB").resize(
        (target_size, target_size), Image.LANCZOS
    )
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3] in [0, 1]

    # Normalize per ImageNet
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    # HWC → CHW → BCHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _pil_to_b64(img_array: np.ndarray, fmt: str = "JPEG") -> str:
    """Numpy RGB array → base64 data URI."""
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format=fmt, quality=92)
    buf.seek(0)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


# ════════════════════════════════════════════════════════════
#  SPECTRAL INDEX FETCHER (reuses spectral_analyzer)
# ════════════════════════════════════════════════════════════

def _fetch_spectral_data(bbox, date, size=512):
    """
    Fetch NDVI/NDBI/SCL from Sentinel-2 via Process API.
    Returns (ndvi, ndbi, valid_mask) or None on failure.
    """
    from backend.spectral_analyzer import get_spectral_detector, GOOD_SCL

    detector = get_spectral_detector()
    tile = detector.fetch_spectral_tile(bbox, date, size=size)
    if tile is None:
        return None

    # Resize if needed
    if tile.shape[0] != size or tile.shape[1] != size:
        tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_NEAREST)

    ndvi = (tile[:, :, 0].astype(np.float32) / 127.5) - 1.0
    ndbi = (tile[:, :, 1].astype(np.float32) / 127.5) - 1.0
    scl  = tile[:, :, 2].astype(np.uint8)
    valid = np.isin(scl, list(GOOD_SCL))

    return ndvi, ndbi, valid


# ════════════════════════════════════════════════════════════
#  COMBINED ML CHANGE DETECTOR
# ════════════════════════════════════════════════════════════

class MLChangeDetector:
    """
    Production change detection pipeline combining:
      • ChangeFormer (Deep Learning) — pixel-level change detection
      • Spectral indices (NDVI/NDBI) — change type classification
    """

    # ── Change categories ────────────────────────────────────
    CAT_DEFORESTATION       = 1   # Forest → built-up
    CAT_NEW_CONSTRUCTION    = 2   # Any → built-up (NDBI rise)
    CAT_WATER_CHANGE        = 3   # Water gained or lost
    CAT_AGRICULTURAL_CHANGE = 4   # Cropland/farmland shift
    CAT_ROAD_DEVELOPMENT    = 5   # Linear built patterns
    CAT_VEGETATION_RECOVERY = 6   # NDVI increase

    CAT_INFO = {
        1: ("Deforestation",        (220,  30,  30)),  # red
        2: ("New Construction",     (255, 140,   0)),  # orange
        3: ("Water Body Change",    ( 30, 120, 255)),  # blue
        4: ("Agricultural Change",  (180, 140,  60)),  # brown
        5: ("Road Development",     (100, 100, 100)),  # gray
        6: ("Vegetation Recovery",  ( 30, 200,  80)),  # green
    }

    # ── Thresholds ───────────────────────────────────────────
    CHANGE_PROB_THRESHOLD   = 0.35  # ChangeFormer probability cutoff
    NDVI_FOREST_THRESH      = 0.35  # "was forest" if NDVI_before > this
    NDVI_VEG_THRESH         = 0.25  # "was vegetated" (crops/grass)
    DNDVI_LOSS_THRESH       = -0.12 # vegetation loss
    DNDBI_RISE_THRESH       = 0.08  # built-up increase
    WATER_NDVI_THRESH       = -0.05 # water has very low/negative NDVI
    WATER_NDBI_THRESH       = -0.10 # water has negative NDBI
    DNDVI_GAIN_THRESH       = 0.15  # vegetation recovery
    MIN_BLOB_AREA           = 100   # min connected component size (px)

    def __init__(self, device: str = None, model_size: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model = None

    # ── Load ChangeFormer ────────────────────────────────────
    def _ensure_model(self):
        """Load ChangeFormer model (lazy, on first use)."""
        if self.model is not None:
            return

        from backend.changeformer import get_changeformer

        weights_dir = Path(__file__).parent.parent / "models" / "weights"
        weights_path = weights_dir / "changeformer_levir.pth"
        wp = str(weights_path) if weights_path.exists() else None

        print(f"\n🧠 Loading ChangeFormer (MiT-b1) on {self.device}…", flush=True)
        self.model = get_changeformer(weights_path=wp, device=self.device)
        print("✓ ChangeFormer ready\n", flush=True)

    # ── Step 2: DL Change Detection ─────────────────────────
    @torch.no_grad()
    def _run_changeformer(
        self, before_img: Image.Image, after_img: Image.Image
    ) -> np.ndarray:
        """
        Run ChangeFormer inference on two images.

        Returns:
            change_prob: np.ndarray [H, W] in [0, 1]
        """
        self._ensure_model()

        before_t = preprocess_for_model(
            before_img, self.model_size, self.device
        )
        after_t = preprocess_for_model(
            after_img, self.model_size, self.device
        )

        print("  🧠 Running ChangeFormer inference…", flush=True)
        change_prob = self.model(before_t, after_t)  # [1, 1, H, W]
        prob_map = change_prob.squeeze().cpu().numpy()  # [H, W]

        print(f"  ✓ Change probability — "
              f"mean={prob_map.mean():.3f}, max={prob_map.max():.3f}, "
              f">{self.CHANGE_PROB_THRESHOLD}: "
              f"{(prob_map > self.CHANGE_PROB_THRESHOLD).sum()} px",
              flush=True)

        return prob_map

    # ── Step 4: Classification ───────────────────────────────
    def _classify_changes(
        self,
        change_mask: np.ndarray,
        ndvi_b: np.ndarray,
        ndbi_b: np.ndarray,
        ndvi_a: np.ndarray,
        ndbi_a: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """
        Classify changed pixels into categories using spectral indices.

        Logic:
          Within the DL-detected change mask:
            1. Was forest (NDVI>0.35) AND NDBI rose → Deforestation
            2. NDBI rose significantly → New Construction
            3. Water signature appeared/disappeared → Water Change
            4. Was vegetated (NDVI 0.25-0.35) AND NDVI dropped → Agricultural
            5. Linear high-NDBI patterns → Road Development
            6. NDVI increased → Vegetation Recovery
        """
        h, w = change_mask.shape
        cat = np.zeros((h, w), dtype=np.uint8)

        dndvi = ndvi_a - ndvi_b
        dndbi = ndbi_a - ndbi_b

        changed = (change_mask > 0) & valid

        # ── 1. Deforestation ─────────────────────────────
        was_forest = ndvi_b > self.NDVI_FOREST_THRESH
        ndbi_rose  = dndbi > self.DNDBI_RISE_THRESH
        ndvi_fell  = dndvi < self.DNDVI_LOSS_THRESH
        cat[changed & was_forest & ndvi_fell & ndbi_rose] = self.CAT_DEFORESTATION

        # ── 2. New Construction (non-forest → built-up) ──
        cat[changed & ~was_forest & ndbi_rose & (cat == 0)] = self.CAT_NEW_CONSTRUCTION

        # ── 3. Water Body Change ─────────────────────────
        was_water = (ndvi_b < self.WATER_NDVI_THRESH) & (ndbi_b < self.WATER_NDBI_THRESH)
        is_water  = (ndvi_a < self.WATER_NDVI_THRESH) & (ndbi_a < self.WATER_NDBI_THRESH)
        water_changed = changed & (was_water != is_water) & (cat == 0)
        cat[water_changed] = self.CAT_WATER_CHANGE

        # ── 4. Agricultural Change ───────────────────────
        was_agri = (ndvi_b > self.NDVI_VEG_THRESH) & (ndvi_b <= self.NDVI_FOREST_THRESH)
        cat[changed & was_agri & ndvi_fell & (cat == 0)] = self.CAT_AGRICULTURAL_CHANGE

        # ── 5. Road Development (linear NDBI patterns) ───
        # Detect elongated shapes in the remaining NDBI-rise pixels
        remaining_built = changed & ndbi_rose & (cat == 0)
        if np.sum(remaining_built) > 0:
            road_mask = self._detect_linear_features(
                remaining_built.astype(np.uint8)
            )
            cat[road_mask > 0] = self.CAT_ROAD_DEVELOPMENT
            # Non-linear remaining built pixels → New Construction
            cat[remaining_built & (cat == 0)] = self.CAT_NEW_CONSTRUCTION

        # ── 6. Vegetation Recovery ───────────────────────
        cat[changed & (dndvi > self.DNDVI_GAIN_THRESH) & (cat == 0)] = self.CAT_VEGETATION_RECOVERY

        return cat

    @staticmethod
    def _detect_linear_features(mask: np.ndarray) -> np.ndarray:
        """
        Detect elongated (road-like) shapes using morphological analysis.
        Roads are long & thin; buildings are compact.
        """
        # Find contours and check aspect ratio
        road_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 5:
                continue
            # Fit minimum bounding rectangle
            rect = cv2.minAreaRect(cnt)
            (_, (rw, rh), _) = rect
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            area = cv2.contourArea(cnt)
            # Roads: aspect ratio > 4 and area > 200
            if aspect > 4.0 and area > 200:
                cv2.drawContours(road_mask, [cnt], -1, 1, -1)
        return road_mask

    # ── Morphological cleanup ────────────────────────────────
    def _cleanup_mask(self, cat: np.ndarray) -> np.ndarray:
        """Clean each category with morphological ops + blob filtering."""
        kernel = np.ones((5, 5), np.uint8)
        result = np.zeros_like(cat)

        for c in range(1, 7):
            mask_c = (cat == c).astype(np.uint8)
            if np.sum(mask_c) == 0:
                continue

            # Open (remove noise) then Close (fill gaps)
            mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, kernel)
            mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, kernel)

            # Remove tiny blobs
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_c, connectivity=8
            )
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] < self.MIN_BLOB_AREA:
                    mask_c[labels == i] = 0

            result[mask_c > 0] = c

        return result

    # ════════════════════════════════════════════════════════
    #  MAIN PIPELINE
    # ════════════════════════════════════════════════════════
    def analyze_changes(
        self,
        before_img: Image.Image,
        after_img: Image.Image,
        bbox: Dict[str, float],
        before_date: str,
        after_date: str,
        detect_types: List[str] = None,
        pixel_resolution: float = 10.0,
    ) -> Dict:
        """
        Full ML change detection pipeline.

        Parameters
        ----------
        before_img, after_img : PIL Images (true-colour satellite)
        bbox : bounding box {west, south, east, north}
        before_date, after_date : ISO date strings
        detect_types : list of categories to detect (None = all)
            Options: "deforestation", "construction", "water",
                     "agricultural", "roads", "vegetation_recovery"
        pixel_resolution : ground sampling distance in metres

        Returns
        -------
        dict with:
            - category-wise statistics (hectares, pixels, %)
            - overlay images (base64)
            - change probability heatmap
            - explanation text
        """
        if detect_types is None:
            detect_types = [
                "deforestation", "construction", "water",
                "agricultural", "roads", "vegetation_recovery",
            ]

        print(f"\n{'='*60}", flush=True)
        print("  ML CHANGE DETECTION PIPELINE", flush=True)
        print(f"  Model: ChangeFormer (MiT-b1 Transformer)", flush=True)
        print(f"  + Spectral: NDVI + NDBI (Sentinel-2 L2A)", flush=True)
        print(f"  Dates: {before_date} → {after_date}", flush=True)
        print(f"  Device: {self.device}", flush=True)
        print(f"  Detect: {detect_types}", flush=True)
        print(f"{'='*60}\n", flush=True)

        sz = self.model_size

        # ── Step 1: Preprocess images ────────────────────
        print("Step 1/5: Preprocessing images…", flush=True)
        before_resized = before_img.convert("RGB").resize((sz, sz), Image.LANCZOS)
        after_resized  = after_img.convert("RGB").resize((sz, sz), Image.LANCZOS)
        before_arr = np.array(before_resized)
        after_arr  = np.array(after_resized)
        print(f"  Images resized to {sz}×{sz}", flush=True)

        # ── Step 2: ChangeFormer (DL) ────────────────────
        print("\nStep 2/5: Deep Learning change detection…", flush=True)
        change_prob = self._run_changeformer(before_resized, after_resized)

        # Resize prob map to model_size if different
        if change_prob.shape != (sz, sz):
            change_prob = cv2.resize(change_prob, (sz, sz),
                                     interpolation=cv2.INTER_LINEAR)

        # Binary change mask from DL model
        dl_change_mask = (change_prob > self.CHANGE_PROB_THRESHOLD).astype(np.uint8)
        dl_change_pct = float(np.sum(dl_change_mask)) / dl_change_mask.size * 100
        print(f"  DL detected change in {dl_change_pct:.1f}% of pixels", flush=True)

        # ── Step 3: Spectral data ────────────────────────
        print("\nStep 3/5: Fetching spectral indices (NDVI + NDBI)…", flush=True)
        spectral_before = _fetch_spectral_data(bbox, before_date, size=sz)
        spectral_after  = _fetch_spectral_data(bbox, after_date, size=sz)

        if spectral_before is not None and spectral_after is not None:
            ndvi_b, ndbi_b, valid_b = spectral_before
            ndvi_a, ndbi_a, valid_a = spectral_after
            valid = valid_b & valid_a
            has_spectral = True
            cloud_pct = round((1 - np.sum(valid) / valid.size) * 100, 1)
            print(f"  ✓ Spectral data loaded (clouds: {cloud_pct}%)", flush=True)
            print(f"  NDVI before: mean={np.mean(ndvi_b[valid]):.3f}", flush=True)
            print(f"  NDVI after:  mean={np.mean(ndvi_a[valid]):.3f}", flush=True)
        else:
            # Fallback: estimate spectral from RGB (less accurate)
            print("  ⚠ Spectral fetch failed — estimating from RGB…", flush=True)
            ndvi_b, ndbi_b = self._estimate_spectral_from_rgb(before_arr)
            ndvi_a, ndbi_a = self._estimate_spectral_from_rgb(after_arr)
            valid = np.ones((sz, sz), dtype=bool)
            has_spectral = False
            cloud_pct = 0.0

        # Apply cloud mask to DL change mask
        dl_change_mask = dl_change_mask & valid.astype(np.uint8)

        # ── Step 4: Classification ───────────────────────
        print("\nStep 4/5: Classifying change types…", flush=True)
        cat = self._classify_changes(
            dl_change_mask, ndvi_b, ndbi_b, ndvi_a, ndbi_a, valid
        )

        # Filter by requested detect_types
        type_map = {
            "deforestation": self.CAT_DEFORESTATION,
            "construction": self.CAT_NEW_CONSTRUCTION,
            "water": self.CAT_WATER_CHANGE,
            "agricultural": self.CAT_AGRICULTURAL_CHANGE,
            "roads": self.CAT_ROAD_DEVELOPMENT,
            "vegetation_recovery": self.CAT_VEGETATION_RECOVERY,
        }
        allowed = {type_map[t] for t in detect_types if t in type_map}
        cat[~np.isin(cat, list(allowed))] = 0

        # Morphological cleanup
        cat = self._cleanup_mask(cat)

        # ── Step 5: Visualisation & stats ────────────────
        print("\nStep 5/5: Building visualisations…", flush=True)

        pixel_area_sqm = pixel_resolution ** 2
        total_valid = int(np.sum(valid))

        # Category statistics
        categories = {}
        for c, (label, color) in self.CAT_INFO.items():
            n = int(np.sum(cat == c))
            if n > 0:
                ha = round(n * pixel_area_sqm / 10_000, 2)
                n_labels, _, _, _ = cv2.connectedComponentsWithStats(
                    (cat == c).astype(np.uint8), connectivity=8
                )
                categories[label] = {
                    "pixels": n,
                    "hectares": ha,
                    "acres": round(ha * 2.47105, 2),
                    "sqkm": round(ha / 100, 4),
                    "percent": round(n / max(1, total_valid) * 100, 2),
                    "zones": n_labels - 1,
                    "color": f"rgb({color[0]},{color[1]},{color[2]})",
                }

        total_changed = int(np.sum(cat > 0))
        total_pct = round(total_changed / max(1, total_valid) * 100, 2)
        total_ha = round(total_changed * pixel_area_sqm / 10_000, 2)

        # ── Build overlay images ─────────────────────────

        # 1) After overlay — colour-coded changes on after image
        after_overlay = after_arr.copy()
        for c, (label, color) in self.CAT_INFO.items():
            mask_c = (cat == c).astype(np.uint8)
            if np.sum(mask_c) == 0:
                continue
            fill = after_overlay.copy()
            fill[mask_c > 0] = color
            after_overlay = cv2.addWeighted(after_overlay, 0.60, fill, 0.40, 0)
            cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(after_overlay, cnts, -1, color, 2)

        # 2) Before overlay — green tint on forest, red on areas-to-be-lost
        before_overlay = before_arr.copy()
        forest_mask = (ndvi_b > self.NDVI_FOREST_THRESH).astype(np.uint8)
        fill_g = before_overlay.copy()
        fill_g[forest_mask > 0] = (30, 200, 60)
        before_overlay = cv2.addWeighted(before_overlay, 0.75, fill_g, 0.25, 0)
        loss_mask = (cat > 0).astype(np.uint8) & ((cat != self.CAT_VEGETATION_RECOVERY) & (cat != 0)).astype(np.uint8)
        if np.sum(loss_mask) > 0:
            fill_r = before_overlay.copy()
            fill_r[loss_mask > 0] = (220, 30, 10)
            before_overlay = cv2.addWeighted(before_overlay, 0.55, fill_r, 0.45, 0)
            cnts, _ = cv2.findContours(loss_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(before_overlay, cnts, -1, (255, 220, 0), 2)

        # 3) Change probability heatmap (DL model raw output)
        prob_u8 = (np.clip(change_prob, 0, 1) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_blend = cv2.addWeighted(after_arr, 0.30, heatmap, 0.70, 0)

        # 4) NDVI difference heatmap
        dndvi = ndvi_a - ndvi_b
        dndvi_clip = np.clip(dndvi, -0.5, 0.5)
        dndvi_u8 = ((dndvi_clip + 0.5) * 255).astype(np.uint8)
        ndvi_hmap = cv2.applyColorMap(255 - dndvi_u8, cv2.COLORMAP_JET)
        ndvi_hmap = cv2.cvtColor(ndvi_hmap, cv2.COLOR_BGR2RGB)
        ndvi_blend = cv2.addWeighted(after_arr, 0.30, ndvi_hmap, 0.70, 0)

        # 5) Spotlight (dim everything except changed zones)
        dimmed = (before_arr * 0.15).astype(np.uint8)
        spotlight = dimmed.copy()
        any_change = (cat > 0).astype(np.uint8)
        if np.sum(any_change) > 0:
            spotlight[any_change > 0] = after_arr[any_change > 0]
            for c, (label, color) in self.CAT_INFO.items():
                mask_c = (cat == c).astype(np.uint8)
                if np.sum(mask_c) == 0:
                    continue
                cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(spotlight, cnts, -1, color, 3)

        # ── Vegetation stats ─────────────────────────────
        veg_before_pct = round(
            float(np.sum((ndvi_b > self.NDVI_VEG_THRESH) & valid))
            / max(1, total_valid) * 100, 2
        )
        veg_after_pct = round(
            float(np.sum((ndvi_a > self.NDVI_VEG_THRESH) & valid))
            / max(1, total_valid) * 100, 2
        )
        net_veg_loss = round(max(0, veg_before_pct - veg_after_pct), 2)

        # ── Explanation ──────────────────────────────────
        explanation = self._build_explanation(
            categories, veg_before_pct, veg_after_pct, net_veg_loss,
            cloud_pct, dl_change_pct, has_spectral
        )

        # ── Print summary ────────────────────────────────
        print(f"\n{'='*50}", flush=True)
        print("  RESULTS:", flush=True)
        print(f"  Total change: {total_pct}% ({total_ha} ha)", flush=True)
        for label, info in categories.items():
            print(f"    {label}: {info['hectares']} ha "
                  f"({info['zones']} zones)", flush=True)
        print(f"  Vegetation: {veg_before_pct}% → {veg_after_pct}% "
              f"(loss: {net_veg_loss}%)", flush=True)
        print(f"{'='*50}\n", flush=True)

        return {
            "status": "success",
            "method": "ChangeFormer (MiT-b1 Transformer) + Spectral Indices (NDVI+NDBI)",
            "models": {
                "dl_model": "ChangeFormer — Siamese MiT-b1 (13.7M params)",
                "spectral": "NDVI (B08−B04) + NDBI (B11−B08) from Sentinel-2 L2A",
                "device": self.device,
            },
            "data_source": "Sentinel-2 L2A (B04, B08, B11, SCL)" if has_spectral else "RGB estimation",
            "before": {
                "overlay_image": _pil_to_b64(before_overlay),
                "veg_percent": veg_before_pct,
                "ndvi_mean": round(float(np.mean(ndvi_b[valid])), 3) if has_spectral else None,
            },
            "after": {
                "overlay_image": _pil_to_b64(after_overlay),
                "veg_percent": veg_after_pct,
                "ndvi_mean": round(float(np.mean(ndvi_a[valid])), 3) if has_spectral else None,
            },
            "change": {
                "total_change_percent": total_pct,
                "total_change_hectares": total_ha,
                "net_vegetation_loss_percent": net_veg_loss,
                "dl_change_percent": round(dl_change_pct, 2),
                "cloud_masked_percent": cloud_pct,
            },
            "categories": categories,
            "explanation": explanation,
            "overlays": {
                "classified": _pil_to_b64(after_overlay),
                "dl_heatmap": _pil_to_b64(heatmap_blend),
                "ndvi_heatmap": _pil_to_b64(ndvi_blend),
                "spotlight": _pil_to_b64(spotlight),
            },
        }

    # ── Fallback: estimate spectral from RGB ─────────────────
    @staticmethod
    def _estimate_spectral_from_rgb(rgb: np.ndarray):
        """
        Rough NDVI/NDBI estimation from true-colour RGB when spectral
        bands are unavailable. Uses Excess Green Index as NDVI proxy
        and inverse brightness as NDBI proxy.

        This is less accurate than real NIR/SWIR but still usable.
        """
        r = rgb[:, :, 0].astype(np.float32) / 255.0
        g = rgb[:, :, 1].astype(np.float32) / 255.0
        b = rgb[:, :, 2].astype(np.float32) / 255.0

        # ExG as NDVI proxy: (2G - R - B) / (R + G + B)
        total = r + g + b + 1e-6
        exg = (2.0 * g - r - b) / total
        ndvi_proxy = np.clip(exg * 2.0, -1, 1)  # scale to NDVI range

        # Built-up proxy: low saturation + high brightness
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        ndbi_proxy = np.clip((1.0 - s) * v - 0.3, -1, 1)

        return ndvi_proxy, ndbi_proxy

    # ── Explanation builder ──────────────────────────────────
    @staticmethod
    def _build_explanation(
        categories, veg_before, veg_after, net_loss,
        cloud_pct, dl_pct, has_spectral
    ) -> str:
        cat_lines = ""
        icons = {
            "Deforestation": "🔴",
            "New Construction": "🟠",
            "Water Body Change": "🔵",
            "Agricultural Change": "🟤",
            "Road Development": "⚫",
            "Vegetation Recovery": "🟢",
        }
        for label, info in categories.items():
            icon = icons.get(label, "•")
            cat_lines += (f"  {icon} {label}: {info['hectares']} ha "
                          f"({info['zones']} zones, {info['percent']}%)\n")
        if not cat_lines:
            cat_lines = "  No significant changes detected\n"

        spectral_note = (
            "  Data: Sentinel-2 L2A spectral bands (B04, B08, B11)"
            if has_spectral else
            "  Data: RGB-estimated spectral indices (less accurate)"
        )

        return (
            "ML CHANGE DETECTION PIPELINE\n"
            "═══════════════════════════════\n\n"
            "STEP 1 — DEEP LEARNING (ChangeFormer)\n"
            "  Model: Siamese MiT-b1 Transformer (~13.7M parameters)\n"
            "  Pretrained: ImageNet-1k (via HuggingFace nvidia/mit-b1)\n"
            "  Task: Pixel-level change detection between two images\n"
            "  How: Shared-weight encoder extracts features from both\n"
            "       images, difference module computes |F_after - F_before|\n"
            "       at 4 scales, MLP decoder predicts change probability.\n"
            f"  Result: {dl_pct:.1f}% of pixels flagged as changed\n\n"
            "STEP 2 — SPECTRAL CLASSIFICATION (NDVI + NDBI)\n"
            f"  {spectral_note}\n"
            "  NDVI = (NIR − Red) / (NIR + Red) → vegetation health\n"
            "  NDBI = (SWIR − NIR) / (SWIR + NIR) → built-up surfaces\n"
            "  Uses temporal difference of indices to classify WHAT changed.\n\n"
            "CLASSIFICATION LOGIC:\n"
            "  🔴 Forest + NDVI drop + NDBI rise  → Deforestation\n"
            "  🟠 NDBI rise (non-forest)           → New Construction\n"
            "  🔵 Water signature appeared/lost     → Water Body Change\n"
            "  🟤 Cropland + NDVI drop              → Agricultural Change\n"
            "  ⚫ Linear high-NDBI shapes            → Road Development\n"
            "  🟢 NDVI increase                     → Vegetation Recovery\n\n"
            f"THIS RUN:\n"
            f"  Vegetation before: {veg_before}%\n"
            f"  Vegetation after:  {veg_after}%\n"
            f"  Net vegetation loss: {net_loss}%\n"
            f"  Clouds masked: {cloud_pct}%\n\n"
            f"DETECTED CHANGES:\n{cat_lines}\n"
            "FOR BEST RESULTS:\n"
            "  ✓  Cloud-free images for both dates\n"
            "  ✓  Same SEASON for both (Jun vs Jun, not Jun vs Dec)\n"
            "  ✓  3–10 years apart for meaningful change\n"
            "  ✓  Sentinel-2 at 10 m resolution\n"
            "  ✗  Avoid monsoon vs dry season (gives false positives)\n"
        )


# ════════════════════════════════════════════════════════════
#  SINGLETON
# ════════════════════════════════════════════════════════════

_ml_detector = None


def get_ml_detector(device: str = None) -> MLChangeDetector:
    """Get or create singleton MLChangeDetector."""
    global _ml_detector
    if _ml_detector is None:
        _ml_detector = MLChangeDetector(device=device)
    return _ml_detector
