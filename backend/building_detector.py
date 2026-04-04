"""
Step 2: Building Detection via Semantic Segmentation
=====================================================

Deep-learning–based building mask generator for Sentinel-2 imagery.

Pipeline
--------
1. load_image()      – GeoTIFF (rasterio) OR NumPy array input
2. preprocess()      – Normalize, convert to RGB (B04/B03/B02), resize to 512×512
3. run_model()       – SegFormer-b2 (HF Transformers) with DeepLabV3 fallback
4. postprocess()     – Threshold → morphological cleanup → binary mask
5. detect_buildings() – End-to-end convenience wrapper

Outputs
-------
* before_building_mask  (H, W)  uint8  {0, 1}
* after_building_mask   (H, W)  uint8  {0, 1}
* Optional PNG saves and matplotlib visualization
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_INPUT_SIZE = 512          # H × W fed to the segmentation model
BUILDING_THRESHOLD = 0.50       # Sigmoid probability above which a pixel is "building"

# SegFormer model identifiers (Hugging Face)
_SEGFORMER_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"

# Morphological cleanup parameters
_MORPH_CLOSE_K = 7             # kernel size for gap-filling
_MORPH_OPEN_K  = 3             # kernel size for noise removal
_MIN_BUILDING_AREA = 50        # minimum connected-component area (pixels)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(
    source: Union[str, Path, np.ndarray],
    band_order: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Load a Sentinel-2 scene and return a float32 array shaped (H, W, C).

    Parameters
    ----------
    source : str | Path | np.ndarray
        * Path ending in .tif/.tiff → read with rasterio
        * Path ending in .npy       → load NumPy array
        * np.ndarray               → used as-is (must be (H,W,C) or (C,H,W))
    band_order : tuple of 3 ints, optional
        Zero-based channel indices for (Red, Green, Blue) in the loaded array.
        Defaults to (2, 1, 0) assuming bands are ordered B02, B03, B04, B08
        (so Red=B04=idx2, Green=B03=idx1, Blue=B02=idx0).

    Returns
    -------
    np.ndarray   float32 (H, W, C)  — raw pixel values, not yet normalised
    """
    if band_order is None:
        band_order = (2, 1, 0)  # B04 (Red), B03 (Green), B02 (Blue)

    if isinstance(source, np.ndarray):
        arr = source.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
            # Looks like (C, H, W) – transpose
            arr = np.transpose(arr, (1, 2, 0))
        logger.debug("Loaded image from ndarray: shape=%s", arr.shape)
        return arr

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Image path does not exist: {source}")

    suffix = source.suffix.lower()

    if suffix in (".tif", ".tiff"):
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError("rasterio is required to load GeoTIFF files.") from exc

        with rasterio.open(source) as ds:
            data = ds.read().astype(np.float32)   # (C, H, W)
        arr = np.transpose(data, (1, 2, 0))        # → (H, W, C)
        logger.info("Loaded GeoTIFF %s: shape=%s", source.name, arr.shape)

    elif suffix == ".npy":
        arr = np.load(source).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0))
        logger.info("Loaded NumPy array %s: shape=%s", source.name, arr.shape)

    elif suffix in (".png", ".jpg", ".jpeg"):
        img = Image.open(source).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        logger.info("Loaded image %s: shape=%s", source.name, arr.shape)

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    image: np.ndarray,
    band_order: Tuple[int, int, int] = (2, 1, 0),
    target_size: int = MODEL_INPUT_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw Sentinel-2 data to a model-ready RGB tensor.

    Steps
    -----
    1. Select R/G/B channels (B04, B03, B02 by default)
    2. Clip extreme reflectance values (0–10 000 DN range for Sentinel-2)
    3. Normalise to [0, 1]
    4. Resize to (target_size, target_size)
    5. Return both the resized float32 array AND a uint8 BGR version for OpenCV ops

    Parameters
    ----------
    image      : (H, W, C) float32 raw pixel values
    band_order : (r_idx, g_idx, b_idx) into the channel axis
    target_size: model spatial resolution

    Returns
    -------
    rgb_norm   : (H, W, 3) float32  in [0,1]  — to feed the model
    rgb_uint8  : (H, W, 3) uint8            — for visualization
    """
    h_in, w_in, n_ch = image.shape

    # ── Select RGB bands ──────────────────────────────────────────────────────
    max_idx = max(band_order)
    if n_ch <= max_idx:
        warnings.warn(
            f"Image has only {n_ch} channels but band_order asks for index {max_idx}. "
            "Falling back to the first 3 channels (or replicating if <3).",
            stacklevel=2,
        )
        if n_ch >= 3:
            rgb = image[:, :, :3]
        elif n_ch == 1:
            rgb = np.repeat(image, 3, axis=2)
        else:
            rgb = np.concatenate(
                [image, np.zeros((h_in, w_in, 3 - n_ch), dtype=np.float32)], axis=2
            )
    else:
        rgb = image[:, :, list(band_order)]

    # ── Normalize ─────────────────────────────────────────────────────────────
    # Sentinel-2 DN values typically range 0–10000;
    # plain RGB images are 0–255.
    max_val = rgb.max()
    if max_val > 255:
        # Sentinel-2 reflectance units (clamp at p98 to reduce hot-pixel effect)
        p98 = np.percentile(rgb, 98)
        rgb = np.clip(rgb, 0.0, p98 if p98 > 0 else max_val)
        rgb = rgb / max(p98, 1.0)
    elif max_val > 1.0:
        rgb = rgb / 255.0

    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    # ── Resize ────────────────────────────────────────────────────────────────
    if h_in != target_size or w_in != target_size:
        rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return rgb, rgb_uint8


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model loading (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

class _ModelRegistry:
    """Lazy singleton that loads the segmentation model exactly once."""

    _instance: Optional["_ModelRegistry"] = None

    def __init__(self) -> None:
        self._segformer_processor = None
        self._segformer_model = None
        self._deeplab_model = None
        self._backend: Optional[str] = None
        self._device: str = "cpu"

    @classmethod
    def get(cls) -> "_ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── SegFormer ─────────────────────────────────────────────────────────────
    def _try_load_segformer(self) -> bool:
        """
        Load SegFormer-b2 (ADE-20K weights).

        We use ADE-20K weights because:
        * They include "building" as class 1 (wall) and 48 (skyscraper/building)
        * The model was trained on 512×512 images, matching our pipeline
        * It generalises well to overhead views with fine-tuning or zero-shot
        """
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            import torch

            logger.info("Loading SegFormer model: %s", _SEGFORMER_MODEL_ID)
            self._segformer_processor = SegformerImageProcessor.from_pretrained(_SEGFORMER_MODEL_ID)
            self._segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                _SEGFORMER_MODEL_ID
            )
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._segformer_model = self._segformer_model.to(self._device).eval()
            self._backend = "segformer"
            logger.info("SegFormer loaded on %s", self._device)
            return True

        except Exception as exc:
            logger.warning("SegFormer load failed: %s – trying DeepLabV3 fallback.", exc)
            return False

    # ── DeepLabV3 ─────────────────────────────────────────────────────────────
    def _try_load_deeplab(self) -> bool:
        try:
            import torch
            import torchvision.models.segmentation as seg_models

            logger.info("Loading DeepLabV3+ ResNet-101 (COCO pretrained)")
            weights = seg_models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            self._deeplab_model = seg_models.deeplabv3_resnet101(weights=weights)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._deeplab_model = self._deeplab_model.to(self._device).eval()
            self._backend = "deeplab"
            logger.info("DeepLabV3 loaded on %s", self._device)
            return True

        except Exception as exc:
            logger.error("DeepLabV3 load also failed: %s", exc)
            return False

    def ensure_loaded(self, preferred: str = "segformer") -> str:
        """Load the chosen backend if not already loaded. Returns backend name."""
        if self._backend is not None:
            return self._backend

        if preferred == "segformer":
            ok = self._try_load_segformer()
            if not ok:
                ok = self._try_load_deeplab()
        else:
            ok = self._try_load_deeplab()
            if not ok:
                ok = self._try_load_segformer()

        if not ok:
            raise RuntimeError(
                "Neither SegFormer nor DeepLabV3 could be loaded. "
                "Check your torch / transformers installation."
            )
        return self._backend  # type: ignore[return-value]

    # ── ADE-20K building class IDs ─────────────────────────────────────────
    # Classes that correspond to man-made structures:
    # 1=wall, 2=building/skyscraper, 25=house, 48=building, 84=shelter,
    # 31=bridge, 46=column, 15=road (excluded intentionally)
    _ADE_BUILDING_IDS = {1, 2, 25, 48, 84}

    def run_segformer(self, rgb_norm: np.ndarray) -> np.ndarray:
        """Run SegFormer and return a (H, W) float32 building probability map."""
        import torch
        from PIL import Image as PilImage

        pil = PilImage.fromarray((rgb_norm * 255).astype(np.uint8))
        inputs = self._segformer_processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._segformer_model(**inputs)
            # logits: (1, num_classes, H//4, W//4)
            logits = outputs.logits  # shape: [1, C, Hd, Wd]

        # ── Build binary building-class logit map ──────────────────────────
        num_classes = logits.shape[1]
        building_ids = [c for c in self._ADE_BUILDING_IDS if c < num_classes]

        if building_ids:
            # Probability = max softmax over building classes vs background
            softmax = torch.softmax(logits[0], dim=0)         # (C, Hd, Wd)
            building_prob = softmax[building_ids].sum(dim=0)   # (Hd, Wd)
        else:
            # Fallback: treat class 1 (wall) as proxy
            building_prob = torch.softmax(logits[0], dim=0)[1]

        prob_np = building_prob.cpu().float().numpy()          # (Hd, Wd)

        # Upsample to model input size (512×512)
        target = rgb_norm.shape[0]
        prob_np = cv2.resize(prob_np, (target, target), interpolation=cv2.INTER_LINEAR)
        return prob_np.astype(np.float32)

    # ── VOC class 15 = person, class 8 = bicycle                            ──
    # VOC does NOT include "building" as a primary class.
    # We proxy via class 12 (tvmonitor) – NOT useful.
    # Instead we use the "aeroplane" background signal inverted,
    # or better: we reuse the full-image feature activations as a proxy.
    # A cleaner approach: treat high activation on ANY non-background class
    # and low vegetation signal as "urban area ~ building".
    _DEEPLAB_NON_NATURAL_CLASSES = {0, 1, 2, 3, 4, 6, 7, 14, 15, 16, 17, 18, 19}

    def run_deeplab(self, rgb_norm: np.ndarray) -> np.ndarray:
        """Run DeepLabV3 and return a (H, W) float32 building proxy probability."""
        import torch
        import torchvision.transforms.functional as TF

        # ImageNet normalisation expected by DeepLabV3
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])

        tensor = torch.from_numpy(np.transpose(rgb_norm, (2, 0, 1)))  # (3, H, W)
        tensor = TF.normalize(tensor, mean=mean, std=std)
        tensor = tensor.unsqueeze(0).to(self._device)                  # (1, 3, H, W)

        with torch.no_grad():
            out = self._deeplab_model(tensor)["out"]  # (1, 21, H, W)

        softmax = torch.softmax(out[0], dim=0)  # (21, H, W)

        # VOC classes: 0=background, 9=chair, 11=diningtable, 12=dog ...
        # No explicit "building" class in COCO-VOC, so we use a heuristic:
        # "building probability" ~ 1 − vegetation − sky − water
        # We define non-urban classes and subtract their probability.
        NATURAL      = {0, 9, 10, 15}  # background, chair, cow, person (less urban)
        NON_BUILDING = {c for c in range(21) if c in NATURAL}
        non_bld = softmax[list(NON_BUILDING)].sum(dim=0)
        bld_proxy = 1.0 - non_bld

        prob_np = bld_proxy.cpu().float().numpy()
        target = rgb_norm.shape[0]
        prob_np = cv2.resize(prob_np, (target, target), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob_np, 0.0, 1.0).astype(np.float32)


_registry = _ModelRegistry.get()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model inference
# ─────────────────────────────────────────────────────────────────────────────

def run_model(
    rgb_norm: np.ndarray,
    preferred_backend: str = "segformer",
) -> np.ndarray:
    """
    Run the segmentation model on a pre-processed RGB image.

    Parameters
    ----------
    rgb_norm          : (H, W, 3) float32 in [0, 1]
    preferred_backend : "segformer" (default) or "deeplab"

    Returns
    -------
    prob_map : (H, W) float32 in [0, 1] — building probability per pixel
    """
    backend = _registry.ensure_loaded(preferred=preferred_backend)

    if backend == "segformer":
        return _registry.run_segformer(rgb_norm)
    else:
        return _registry.run_deeplab(rgb_norm)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess(
    prob_map: np.ndarray,
    threshold: float = BUILDING_THRESHOLD,
    close_kernel: int = _MORPH_CLOSE_K,
    open_kernel: int = _MORPH_OPEN_K,
    min_area: int = _MIN_BUILDING_AREA,
) -> np.ndarray:
    """
    Convert a raw probability map to a clean binary building mask.

    Steps
    -----
    1. Threshold at `threshold` → raw binary mask
    2. Morphological closing  → fill small holes inside buildings
    3. Morphological opening  → remove salt-and-pepper noise
    4. Remove connected components smaller than `min_area` pixels

    Parameters
    ----------
    prob_map     : (H, W) float32 in [0, 1]
    threshold    : decision boundary (default 0.50)
    close_kernel : kernel size for MORPH_CLOSE (gap filling)
    open_kernel  : kernel size for MORPH_OPEN  (noise removal)
    min_area     : minimum blob size to keep (pixels)

    Returns
    -------
    mask : (H, W) uint8 with values {0, 1}
    """
    # ── Threshold ─────────────────────────────────────────────────────────────
    binary = (prob_map > threshold).astype(np.uint8)

    # ── Close gaps inside buildings ───────────────────────────────────────────
    k_close = np.ones((close_kernel, close_kernel), dtype=np.uint8)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # ── Remove noise / isolated pixels ───────────────────────────────────────
    k_open = np.ones((open_kernel, open_kernel), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # ── Drop small components ─────────────────────────────────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1

    return clean.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. End-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

def detect_buildings(
    before_source: Union[str, Path, np.ndarray],
    after_source: Union[str, Path, np.ndarray],
    band_order: Tuple[int, int, int] = (2, 1, 0),
    target_size: int = MODEL_INPUT_SIZE,
    threshold: float = BUILDING_THRESHOLD,
    preferred_backend: str = "segformer",
    save_dir: Optional[Union[str, Path]] = None,
    visualize: bool = True,
) -> Dict:
    """
    Full building detection pipeline for a before/after Sentinel-2 pair.

    Parameters
    ----------
    before_source    : Path or ndarray for the "before" scene
    after_source     : Path or ndarray for the "after" scene
    band_order       : (red_idx, green_idx, blue_idx) in source channel order
    target_size      : model spatial dimension (default 512)
    threshold        : building probability threshold (default 0.50)
    preferred_backend: "segformer" or "deeplab"
    save_dir         : if set, masks and visualisation are saved here
    visualize        : generate and save a matplotlib figure

    Returns
    -------
    dict with keys:
        before_building_mask  : (H, W) uint8
        after_building_mask   : (H, W) uint8
        before_prob_map       : (H, W) float32      (raw model output)
        after_prob_map        : (H, W) float32
        before_rgb_uint8      : (H, W, 3) uint8
        after_rgb_uint8       : (H, W, 3) uint8
        new_buildings_mask    : (H, W) uint8        (appeared after - before)
        removed_buildings_mask: (H, W) uint8        (disappeared before - after)
        stats                 : dict
        figure_path           : str | None
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading before-image …")
    before_raw = load_image(before_source, band_order)

    logger.info("Loading after-image …")
    after_raw  = load_image(after_source, band_order)

    # ── Preprocess ────────────────────────────────────────────────────────────
    before_norm, before_u8 = preprocess(before_raw, band_order, target_size)
    after_norm,  after_u8  = preprocess(after_raw,  band_order, target_size)

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running segmentation on before-image …")
    before_prob = run_model(before_norm, preferred_backend)

    logger.info("Running segmentation on after-image …")
    after_prob  = run_model(after_norm, preferred_backend)

    # ── Postprocess ───────────────────────────────────────────────────────────
    before_mask = postprocess(before_prob, threshold=threshold)
    after_mask  = postprocess(after_prob,  threshold=threshold)

    # ── Change masks ──────────────────────────────────────────────────────────
    new_buildings     = np.clip(after_mask.astype(int) - before_mask.astype(int), 0, 1).astype(np.uint8)
    removed_buildings = np.clip(before_mask.astype(int) - after_mask.astype(int), 0, 1).astype(np.uint8)

    # ── Statistics ────────────────────────────────────────────────────────────
    total_px = float(before_mask.size)
    stats = {
        "before_building_pixels": int(before_mask.sum()),
        "after_building_pixels":  int(after_mask.sum()),
        "new_building_pixels":    int(new_buildings.sum()),
        "removed_building_pixels":int(removed_buildings.sum()),
        "before_building_pct":  round(before_mask.sum() / total_px * 100, 3),
        "after_building_pct":   round(after_mask.sum()  / total_px * 100, 3),
        "new_building_pct":     round(new_buildings.sum()  / total_px * 100, 3),
        "model_backend":        _registry._backend,
        "threshold":            threshold,
        "image_size":           target_size,
    }

    logger.info(
        "Detection complete — before: %d px (%.2f%%) | after: %d px (%.2f%%) | new: %d px",
        stats["before_building_pixels"], stats["before_building_pct"],
        stats["after_building_pixels"],  stats["after_building_pct"],
        stats["new_building_pixels"],
    )

    # ── Save masks ────────────────────────────────────────────────────────────
    figure_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        _save_mask_png(before_mask, save_dir / "before_building_mask.png")
        _save_mask_png(after_mask,  save_dir / "after_building_mask.png")
        _save_mask_png(new_buildings,     save_dir / "new_buildings.png")
        _save_mask_png(removed_buildings, save_dir / "removed_buildings.png")
        logger.info("Masks saved to %s", save_dir)

    if visualize:
        fig_path = (Path(save_dir) / "building_detection_overview.png") if save_dir else None
        figure_path = _visualize(
            before_u8, after_u8,
            before_mask, after_mask,
            new_buildings, removed_buildings,
            before_prob, after_prob,
            save_path=fig_path,
        )

    return {
        "before_building_mask":   before_mask,
        "after_building_mask":    after_mask,
        "before_prob_map":        before_prob,
        "after_prob_map":         after_prob,
        "before_rgb_uint8":       before_u8,
        "after_rgb_uint8":        after_u8,
        "new_buildings_mask":     new_buildings,
        "removed_buildings_mask": removed_buildings,
        "stats":                  stats,
        "figure_path":            str(figure_path) if figure_path else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Overlay helper
# ─────────────────────────────────────────────────────────────────────────────

def overlay_mask_on_image(
    rgb_uint8: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 165, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a binary building mask onto an RGB image.

    Parameters
    ----------
    rgb_uint8 : (H, W, 3) uint8
    mask      : (H, W) uint8  with values {0, 1}
    color     : RGB fill color for masked pixels (default orange)
    alpha     : opacity of the mask overlay (0=transparent, 1=opaque)

    Returns
    -------
    blended : (H, W, 3) uint8
    """
    overlay = rgb_uint8.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(rgb_uint8, 1 - alpha, overlay, alpha, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(str(path))


def _visualize(
    before_u8: np.ndarray,
    after_u8: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    new_bld: np.ndarray,
    removed_bld: np.ndarray,
    before_prob: np.ndarray,
    after_prob: np.ndarray,
    save_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate a 3×3 matplotlib figure summarising the detection results."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.patch.set_facecolor("#0d1117")

    titles = [
        "Before Image (RGB)",          "After Image (RGB)",            "New Buildings",
        "Before Prob Map",              "After Prob Map",               "Removed Buildings",
        "Before Mask Overlay",          "After Mask Overlay",           "Change Summary",
    ]
    for ax in axes.ravel():
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    def _titl(ax, t):
        ax.set_title(t, color="#c9d1d9", fontsize=10, pad=6, fontweight="bold")
        ax.axis("off")

    # Row 0
    axes[0, 0].imshow(before_u8)
    _titl(axes[0, 0], titles[0])

    axes[0, 1].imshow(after_u8)
    _titl(axes[0, 1], titles[1])

    new_vis = np.zeros((*new_bld.shape, 3), dtype=np.uint8)
    new_vis[new_bld > 0] = [50, 205, 50]   # lime-green
    axes[0, 2].imshow(new_vis)
    _titl(axes[0, 2], titles[2])

    # Row 1
    axes[1, 0].imshow(before_prob, cmap="hot", vmin=0, vmax=1)
    _titl(axes[1, 0], titles[3])

    axes[1, 1].imshow(after_prob, cmap="hot", vmin=0, vmax=1)
    _titl(axes[1, 1], titles[4])

    rem_vis = np.zeros((*removed_bld.shape, 3), dtype=np.uint8)
    rem_vis[removed_bld > 0] = [220, 50, 50]  # red
    axes[1, 2].imshow(rem_vis)
    _titl(axes[1, 2], titles[5])

    # Row 2
    axes[2, 0].imshow(overlay_mask_on_image(before_u8, before_mask, (255, 165, 0), 0.45))
    _titl(axes[2, 0], titles[6])

    axes[2, 1].imshow(overlay_mask_on_image(after_u8, after_mask, (255, 165, 0), 0.45))
    _titl(axes[2, 1], titles[7])

    # Change summary panel
    ax_sum = axes[2, 2]
    ax_sum.set_facecolor("#161b22")
    ax_sum.axis("off")
    lines = [
        ("Before buildings", f"{before_mask.sum():,} px", "#58a6ff"),
        ("After buildings",  f"{after_mask.sum():,} px",  "#58a6ff"),
        ("New buildings",    f"{new_bld.sum():,} px",     "#3fb950"),
        ("Removed bldgs",    f"{removed_bld.sum():,} px", "#f85149"),
    ]
    y = 0.85
    for label, val, clr in lines:
        ax_sum.text(0.05, y, label, color="#8b949e", fontsize=11, transform=ax_sum.transAxes)
        ax_sum.text(0.65, y, val,  color=clr,        fontsize=11, transform=ax_sum.transAxes, fontweight="bold")
        y -= 0.18
    _titl(ax_sum, titles[8])

    plt.suptitle("Building Detection — Sentinel-2 Change Analysis",
                 color="#c9d1d9", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(pad=1.5)

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Visualisation saved → %s", save_path)
        return save_path

    plt.savefig("/tmp/building_detection_overview.png", dpi=100,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return Path("/tmp/building_detection_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Demo / synthetic data helper (for testing without real imagery)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_sentinel2(
    size: int = 512,
    seed: int = 42,
    new_buildings: bool = False,
) -> np.ndarray:
    """
    Generate a synthetic Sentinel-2–like array (H, W, 4) for B02/B03/B04/B08.

    Useful for unit-testing the pipeline without real satellite data.

    Parameters
    ----------
    size         : spatial dimension (square)
    seed         : random seed for reproducibility
    new_buildings: if True, add extra "built-up" blocks to simulate change

    Returns
    -------
    np.ndarray (H, W, 4) with DN values in range [0, 10000]
    """
    rng = np.random.RandomState(seed)
    h = w = size

    # Base vegetation / bare soil scene
    nir  = rng.uniform(2000, 6000, (h, w)).astype(np.float32)   # B08
    red  = rng.uniform(300,  2000, (h, w)).astype(np.float32)   # B04
    green= rng.uniform(400,  2500, (h, w)).astype(np.float32)   # B03
    blue = rng.uniform(200,  2000, (h, w)).astype(np.float32)   # B02

    # Simulate some high-reflectance built-up areas
    for _ in range(4):
        x, y = rng.randint(50, w - 100), rng.randint(50, h - 100)
        bw, bh = rng.randint(30, 80), rng.randint(30, 80)
        # Buildings: high red + blue, lower NIR (lower NDVI)
        red  [y:y+bh, x:x+bw]  = rng.uniform(4000, 7000, (bh, bw))
        blue [y:y+bh, x:x+bw]  = rng.uniform(3000, 6000, (bh, bw))
        green[y:y+bh, x:x+bw]  = rng.uniform(3000, 5500, (bh, bw))
        nir  [y:y+bh, x:x+bw]  = rng.uniform(1000, 3000, (bh, bw))

    if new_buildings:
        # Add additional buildings that weren't in the "before" scene
        for _ in range(3):
            x, y = rng.randint(100, w - 120), rng.randint(100, h - 120)
            bw, bh = rng.randint(40, 100), rng.randint(40, 100)
            red  [y:y+bh, x:x+bw] = rng.uniform(5000, 8000, (bh, bw))
            blue [y:y+bh, x:x+bw] = rng.uniform(4000, 7000, (bh, bw))
            green[y:y+bh, x:x+bw] = rng.uniform(4000, 6500, (bh, bw))
            nir  [y:y+bh, x:x+bw] = rng.uniform(800,  2500, (bh, bw))

    # Stack as (H, W, 4) → B02, B03, B04, B08
    return np.stack([blue, green, red, nir], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI / __main__ demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Building detection from Sentinel-2 imagery"
    )
    parser.add_argument("--before", default=None,
                        help="Path to before GeoTIFF / .npy  (omit for synthetic demo)")
    parser.add_argument("--after",  default=None,
                        help="Path to after GeoTIFF / .npy   (omit for synthetic demo)")
    parser.add_argument("--save-dir",  default="output/building_masks",
                        help="Directory to save output masks and visualisation")
    parser.add_argument("--backend",   default="segformer",
                        choices=["segformer", "deeplab"],
                        help="Segmentation backend (default: segformer)")
    parser.add_argument("--threshold", default=0.50, type=float,
                        help="Building probability threshold (default: 0.50)")
    parser.add_argument("--no-vis",  action="store_true",
                        help="Skip matplotlib visualisation")
    args = parser.parse_args()

    # ── Use synthetic data when no real images supplied ─────────────────────
    if args.before is None or args.after is None:
        logger.info("No real images supplied — running synthetic demo.")
        before_data = make_synthetic_sentinel2(seed=42,  new_buildings=False)
        after_data  = make_synthetic_sentinel2(seed=42,  new_buildings=True)
    else:
        before_data = args.before
        after_data  = args.after

    results = detect_buildings(
        before_source    = before_data,
        after_source     = after_data,
        band_order       = (2, 1, 0),       # B04=R, B03=G, B02=B
        target_size      = MODEL_INPUT_SIZE,
        threshold        = args.threshold,
        preferred_backend= args.backend,
        save_dir         = args.save_dir,
        visualize        = not args.no_vis,
    )

    print("\n" + "=" * 55)
    print("  BUILDING DETECTION RESULTS")
    print("=" * 55)
    print(f"  before_mask shape : {results['before_building_mask'].shape}")
    print(f"  after_mask  shape : {results['after_building_mask'].shape}")
    print(f"  Model backend     : {results['stats']['model_backend']}")
    print(f"  Before bldg px    : {results['stats']['before_building_pixels']:,}  "
          f"({results['stats']['before_building_pct']:.2f}%)")
    print(f"  After  bldg px    : {results['stats']['after_building_pixels']:,}  "
          f"({results['stats']['after_building_pct']:.2f}%)")
    print(f"  New buildings     : {results['stats']['new_building_pixels']:,}  "
          f"({results['stats']['new_building_pct']:.2f}%)")
    if results["figure_path"]:
        print(f"  Visualisation     : {results['figure_path']}")
    print("=" * 55 + "\n")
