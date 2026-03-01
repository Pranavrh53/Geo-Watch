"""
Satellite Change Detection -- Vegetation Loss -> New Construction
===============================================================

Goal: Find places where GREEN VEGETATION existed before but has been
      REPLACED by buildings/concrete/roads (deforestation + urbanisation).

Algorithm (direct, accurate):
  1. Compute Excess Green Index (ExG) for BEFORE image
     ExG = (2xG - R - B) / (R+G+B)  ->  high = vegetation, low = concrete
  2. Compute ExG for AFTER image
  3. Vegetation Loss Mask = was_green_before AND not_green_after
  4. Cloud masking to remove satellite artifacts
  5. Morphological cleanup to remove noise

IDEAL INPUT FOR ACCURATE RESULTS:
  OK  Cloud-free images (no white patches over the area)
  OK  Same season for both dates (e.g. both in winter, or both in summer)
      -- avoids seasonal greenness changes being mistaken for construction
  OK  At least 3-10 years apart (to ensure real land-use change)
  OK  Same geographic area / same zoom level
  OK  Sentinel-2 TRUE COLOR (RGB) at 10 m resolution
  NO  Avoid images with heavy haze, fog, or smoke
  NO  Avoid comparing monsoon season vs dry season (green vs brown = false positive)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict
import logging
import io
import base64

logger = logging.getLogger(__name__)


# ============================================================
#  SPECTRAL UTILITIES
# ============================================================

def compute_exg(img):
    """
    Compute Excess Green Index (ExG) per pixel.
    ExG = (2*Green - Red - Blue) / (Red + Green + Blue)
    Interpretation:
      ExG > 0.05  ->  vegetation (trees, crops, grass)
      ExG < 0.02  ->  bare soil, concrete, buildings, roads, water
    Returns: float32 array [H, W]
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total = r + g + b + 1e-6
    return ((2.0 * g - r - b) / total).astype(np.float32)


def detect_cloud_mask(img, brightness_thresh=180, saturation_thresh=35):
    """
    Detect cloud/bright haze pixels.
    Clouds are BRIGHT (high V) and DESATURATED (low S) in HSV.
    Returns binary mask: 1 = cloud/haze, 0 = clear.
    """
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    bright = hsv[:, :, 2] > brightness_thresh
    desat = hsv[:, :, 1] < saturation_thresh
    cloud = (bright & desat).astype(np.uint8)
    kernel = np.ones((21, 21), np.uint8)
    return cv2.dilate(cloud, kernel, iterations=1)


def cleanup_mask(mask, kernel_size=5, min_blob_area=300):
    """
    Morphological cleanup:
      OPEN (erode->dilate) removes isolated noise pixels.
      Remove blobs smaller than min_blob_area pixels.
    Does NOT use CLOSE -- avoids merging separate change zones.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_blob_area:
            cleaned[labels == i] = 0
    return cleaned


def auto_brighten(arr: np.ndarray) -> np.ndarray:
    """
    Auto-brighten a dark satellite image for display using CLAHE
    (Contrast Limited Adaptive Histogram Equalization) per channel.

    This only affects the VISUAL output -- the raw pixel values used
    for ExG analysis are never modified.
    """
    # Check if the image is actually dark (mean brightness < 60)
    mean_brightness = np.mean(arr)
    if mean_brightness >= 60:
        return arr  # already bright enough, no change

    result = np.zeros_like(arr)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    for c in range(3):
        result[:, :, c] = clahe.apply(arr[:, :, c])
    return result


def image_to_base64(image, fmt="JPEG"):
    """Convert PIL Image to base64 data URI."""
    if fmt == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=92)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


# ============================================================
#  VEGETATION-LOSS DETECTOR
# ============================================================

class BuildingDetector:
    """
    Detects areas where vegetation was LOST between two satellite images,
    indicating deforestation / new construction.

    Method: Direct ExG (Excess Green Index) comparison.
      was_vegetated  = ExG_before > VEG_THRESH
      lost_vegation  = ExG_after  < LOSS_THRESH
      change_mask    = was_vegetated AND lost_vegetation AND no_cloud
    """

    VEG_THRESH = 0.06   # ExG_before must exceed this -> "was green"
    LOSS_THRESH = 0.03  # ExG_after must be below this -> "lost green"
    MIN_DROP = 0.04     # Minimum ExG drop to filter noise

    def detect_new_buildings(self, before_img, after_img, pixel_resolution=10.0):
        """Detect vegetation loss -> new construction."""
        print("Detecting vegetation loss (ExG comparison)...", flush=True)

        before_arr = np.array(before_img.convert("RGB"))
        after_arr = np.array(after_img.convert("RGB"))
        h, w = before_arr.shape[:2]

        # Step 1: Cloud masking
        cloud_b = detect_cloud_mask(before_img)
        cloud_a = detect_cloud_mask(after_img)
        cloud_b = cv2.resize(cloud_b, (w, h), interpolation=cv2.INTER_NEAREST)
        cloud_a = cv2.resize(cloud_a, (w, h), interpolation=cv2.INTER_NEAREST)
        cloud = ((cloud_b > 0) | (cloud_a > 0)).astype(np.uint8)
        cloud_pct = float(np.sum(cloud) / cloud.size * 100)
        print(f"Clouds/haze masked: {cloud_pct:.1f}%", flush=True)

        # Step 2: ExG for both images
        exg_before = compute_exg(before_img)
        exg_after = compute_exg(after_img)
        exg_before = cv2.resize(exg_before, (w, h), interpolation=cv2.INTER_LINEAR)
        exg_after = cv2.resize(exg_after, (w, h), interpolation=cv2.INTER_LINEAR)
        exg_drop = exg_before - exg_after

        veg_before_pct = float(np.sum(exg_before > self.VEG_THRESH) / exg_before.size * 100)
        veg_after_pct = float(np.sum(exg_after > self.VEG_THRESH) / exg_after.size * 100)
        print(f"Vegetation BEFORE: {veg_before_pct:.1f}%  AFTER: {veg_after_pct:.1f}%", flush=True)

        # Step 3: Vegetation-loss mask
        was_vegetated = (exg_before > self.VEG_THRESH).astype(np.uint8)
        lost_vegetation = (exg_after < self.LOSS_THRESH).astype(np.uint8)
        significant_drop = (exg_drop > self.MIN_DROP).astype(np.uint8)
        veg_loss = (
            (was_vegetated > 0) &
            (lost_vegetation > 0) &
            (significant_drop > 0) &
            (cloud == 0)
        ).astype(np.uint8)

        # Step 4: Morphological cleanup
        change_mask = cleanup_mask(veg_loss, kernel_size=5, min_blob_area=300)

        # Step 5: Statistics
        total_pixels = change_mask.size
        changed_pixels = int(np.sum(change_mask > 0))
        change_pct = changed_pixels / total_pixels * 100.0
        pixel_area_sqm = pixel_resolution ** 2
        change_ha = (changed_pixels * pixel_area_sqm) / 10_000
        num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
            change_mask, connectivity=8
        )
        num_clusters = num_labels - 1
        print(f"Vegetation loss zones: {num_clusters}", flush=True)
        print(f"Area: {change_ha:.1f} ha  ({change_pct:.1f}%)", flush=True)

        # Step 6: Overlays
        # Auto-brighten the after image for all visual outputs
        # (the ANALYSIS above already ran on raw values -- this is display only)
        after_display = auto_brighten(after_arr)
        print(f"After image mean brightness: raw={np.mean(after_arr):.0f}  display={np.mean(after_display):.0f}", flush=True)

        # --- BEFORE overlay ---
        # Show original before image with green tint on vegetated areas
        # AND red hatching on zones that will be lost ("was here, now gone")
        before_result = before_arr.copy()
        veg_vis = (exg_before > self.VEG_THRESH).astype(np.uint8)
        green_fill = before_result.copy()
        green_fill[veg_vis > 0] = [30, 200, 60]
        before_result = cv2.addWeighted(before_result, 0.75, green_fill, 0.25, 0)
        # Draw cleared zones on BEFORE image -- clearly shows what was lost
        if changed_pixels > 0:
            contours, _ = cv2.findContours(
                change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Solid red fill over cleared zones on before image
            cleared_fill = before_result.copy()
            cleared_fill[change_mask > 0] = [220, 30, 10]
            before_result = cv2.addWeighted(before_result, 0.55, cleared_fill, 0.45, 0)
            cv2.drawContours(before_result, contours, -1, (255, 220, 0), 2)
        before_overlay = Image.fromarray(before_result)

        # --- AFTER overlay ---
        # Brightened after image with red zones showing cleared vegetation
        after_result = after_display.copy()
        if changed_pixels > 0:
            # Solid color fill -- does not depend on underlying pixel brightness
            after_result[change_mask > 0] = [220, 40, 20]
            contours, _ = cv2.findContours(
                change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(after_result, contours, -1, (255, 220, 0), 2)
        after_overlay = Image.fromarray(after_result)

        # --- DIFFERENCE (spotlight) overlay ---
        # Dim the BEFORE image (it's well-exposed), spotlight cleared zones
        dimmed = (before_arr * 0.2).astype(np.uint8)
        diff_img = dimmed.copy()
        if changed_pixels > 0:
            diff_img[change_mask > 0] = before_arr[change_mask > 0]
            contours, _ = cv2.findContours(
                change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(diff_img, contours, -1, (255, 220, 0), 2)
        diff_overlay = Image.fromarray(diff_img)

        # --- HEATMAP of ExG drop intensity ---
        drop_clipped = np.clip(exg_drop, 0, 0.4)
        drop_u8 = (drop_clipped / 0.4 * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(drop_u8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        # Blend with brightened after for context
        heatmap_blend = cv2.addWeighted(after_display, 0.35, heatmap_color, 0.65, 0)
        heatmap_overlay = Image.fromarray(heatmap_blend)

        # Step 7: Explanation
        net_veg_loss = max(0.0, veg_before_pct - veg_after_pct)
        explanation = (
            "HOW THIS DETECTION WORKS\n"
            "-------------------------------\n"
            "Uses the Excess Green Index (ExG), a standard spectral formula\n"
            "used in satellite remote sensing:\n\n"
            "  ExG = (2 x Green - Red - Blue) / (R + G + B)\n\n"
            "  Vegetation (trees/grass): ExG > 0.06\n"
            "  Concrete/buildings/roads: ExG < 0.03\n\n"
            "LOGIC:\n"
            "  Was the pixel GREEN in the BEFORE image? (ExG_before > 0.06)\n"
            "  Is the pixel NO LONGER GREEN in the AFTER image? (ExG_after < 0.03)\n"
            "  If BOTH are true -> vegetation was cleared (shown in RED)\n\n"
            f"THIS RUN:\n"
            f"  Vegetation BEFORE: {veg_before_pct:.1f}%\n"
            f"  Vegetation AFTER:  {veg_after_pct:.1f}%\n"
            f"  Net vegetation loss: {net_veg_loss:.1f}%\n"
            f"  Clouds masked: {cloud_pct:.1f}%\n"
            f"  Zones detected: {num_clusters}\n"
            f"  Area cleared: {change_ha:.1f} ha ({change_pct:.1f}%)\n\n"
            "FOR BEST RESULTS:\n"
            "  OK  Cloud-free images for both dates\n"
            "  OK  Same SEASON for both (Jan vs Jan, not Jan vs July)\n"
            "      Comparing monsoon vs dry season gives false positives!\n"
            "  OK  3-10 years apart for meaningful change\n"
            "  OK  Same area, same zoom level\n"
            "  OK  Sentinel-2 True Color at 10m resolution\n"
            "  NO  Haze, smoke, heavy shadows in either image\n"
        )

        return {
            "status": "success",
            "feature": "new_buildings",
            "method": "Vegetation Loss Detection (ExG Spectral Index)",
            "before": {"overlay_image": image_to_base64(before_overlay)},
            "after": {"overlay_image": image_to_base64(after_overlay)},
            "change": {
                "percent_change": round(change_pct, 2),
                "new_buildup_hectares": round(change_ha, 2),
                "new_buildup_acres": round(change_ha * 2.47105, 2),
                "net_change_hectares": round(change_ha, 2),
                "new_clusters": num_clusters,
                "new_pixels": changed_pixels,
            },
            "explanation": explanation,
            "overlays": {
                "new_buildings": image_to_base64(after_overlay),
                "difference": image_to_base64(diff_overlay),
                "heatmap": image_to_base64(heatmap_overlay),
            },
        }


# ============================================================
# SINGLETON ACCESSOR
# ============================================================

_building_detector = None


def get_building_detector():
    global _building_detector
    if _building_detector is None:
        _building_detector = BuildingDetector()
    return _building_detector