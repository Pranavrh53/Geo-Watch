"""
Modular Feature Detectors for Satellite Change Detection

Each detector identifies a specific land cover type using colour-space
analysis designed for Sentinel-2 TRUE-COLOR composites (Red-Green-Blue).

Channel mapping for TRUE-COLOR composite (layers=TRUE_COLOR):
  Channel 0 = Red   band (B04, 10 m)
  Channel 1 = Green band (B03, 10 m)
  Channel 2 = Blue  band (B02, 10 m)

Key detection indices used (all visible-band):
  ExG   = 2·G − R − B          (Excess Green → vegetation)
  HSV-V = brightness            (built-up = brighter)
  HSV-S = saturation            (built-up = less saturated / gray)
  Texture / edge density        (buildings create strong edges)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
import logging
import io
import base64

logger = logging.getLogger(__name__)


class BaseDetector:
    """Base class for all feature detectors"""

    def detect(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """
        Detect features in image.
        Returns (binary_mask, percentage_coverage)
        """
        raise NotImplementedError

    @staticmethod
    def cleanup_mask(mask: np.ndarray, kernel_size: int = 5,
                     min_blob_area: int = 100) -> np.ndarray:
        """Remove noise and tiny blobs from a binary mask"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Remove small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_blob_area:
                cleaned[labels == i] = 0

        return cleaned

    @staticmethod
    def create_overlay(base_image: Image.Image, mask: np.ndarray,
                       fill_color: Tuple[int, int, int],
                       contour_color: Tuple[int, int, int],
                       fill_alpha: float = 0.25,
                       contour_thickness: int = 2) -> Image.Image:
        """
        Create a clean overlay: light transparent fill + bold contour outlines.
        The base image remains clearly visible.
        """
        base = np.array(base_image.convert("RGB"))

        # Resize mask if needed
        if mask.shape[:2] != base.shape[:2]:
            mask = cv2.resize(mask, (base.shape[1], base.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        result = base.copy()

        if np.any(mask > 0):
            # Semi-transparent fill
            fill = base.copy()
            fill[mask > 0] = fill_color
            result = cv2.addWeighted(base, 1 - fill_alpha, fill, fill_alpha, 0)

            # Bold contour outlines
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, contour_color, contour_thickness)

        return Image.fromarray(result)

    @staticmethod
    def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
        """Convert PIL Image to base64 data URI"""
        buf = io.BytesIO()
        image.save(buf, format=fmt, quality=92)
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        mime = "image/jpeg" if fmt == "JPEG" else "image/png"
        return f"data:{mime};base64,{b64}"


# ============================================================
# BUILDING / BUILT-UP AREA DETECTOR  (TRUE-COLOR)
# ============================================================

class BuildingDetector(BaseDetector):
    """
    Detects built-up areas (buildings, roads, constructed surfaces) in
    Sentinel-2 **TRUE-COLOR** imagery (R-G-B) using visible-band indices.

    Approach — **confidence scoring** (not simple thresholding)
    ------------------------------------------------------------
    Each non-vegetation pixel accumulates a confidence score from
    multiple independent signals.  Only pixels whose aggregate score
    exceeds a threshold are labelled "built-up".

    Signals used:
      1. **ExG** (Excess Green Index = 2G − R − B)
         Low / negative ExG → NOT vegetation → +20 pts base.
      2. **HSV Value (brightness)**
         Concrete, asphalt, metal roofs reflect more light.
      3. **HSV Saturation (grayness)**
         Man-made surfaces are distinctly less saturated than soil or
         vegetation.
      4. **Local texture (σ in 11×11 window)**
         Buildings at 10 m create brightness variation (roof edges,
         shadow gaps) that bare soil / water lack.
      5. **Canny edge density (21×21 window)**
         Dense, structured edges → buildings and roads.

    At Sentinel-2's 10 m resolution individual buildings aren't visible,
    but *clusters* of buildings / construction zones are clearly
    detectable through these combined signals.
    """

    # ---------- tuneable thresholds ----------
    EXG_VEG_THRESH  = 8      # ExG > this  → vegetation
    MIN_BRIGHTNESS  = 25     # V < this → shadow / water → exclude
    SCORE_THRESH    = 40     # minimum confidence to call "built-up"

    def detect(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """Detect built-up areas in a TRUE-COLOR image.

        Uses a multi-signal confidence score per pixel.
        Returns (binary_mask uint8, percentage_coverage float).
        """
        rgb = np.array(image.convert("RGB"))
        r, g, b = (rgb[:, :, 0].astype(np.float32),
                    rgb[:, :, 1].astype(np.float32),
                    rgb[:, :, 2].astype(np.float32))

        # ---- 1. Vegetation & shadow exclusion ----
        exg = 2.0 * g - r - b
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v_chan = hsv[:, :, 2].astype(np.float32)
        s_chan = hsv[:, :, 1].astype(np.float32)

        is_vegetation = exg > self.EXG_VEG_THRESH
        is_shadow     = v_chan < self.MIN_BRIGHTNESS
        candidate     = (~is_vegetation) & (~is_shadow)

        # ---- 2. Local texture: std-dev in 11×11 window ----
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu    = cv2.blur(gray, (11, 11))
        mu_sq = cv2.blur(gray ** 2, (11, 11))
        local_std = np.sqrt(np.maximum(mu_sq - mu ** 2, 0))

        # ---- 3. Edge density in 21×21 window ----
        edges = cv2.Canny(rgb, 50, 130).astype(np.float32) / 255.0
        edge_density = cv2.blur(edges, (21, 21))

        # ---- 4. Confidence score (0-100) ----
        score = np.zeros(gray.shape, dtype=np.float32)

        # 4a  Brightness contribution (0-20 pts)
        #     V in [50, 180] maps linearly to 0-20
        score += np.clip((v_chan - 50) / 130.0, 0, 1) * 20

        # 4b  Grayness / low-saturation contribution (0-20 pts)
        #     S in [130, 40] maps linearly to 0-20
        #     Widened to include Indian brick / clay construction (S~100-120)
        score += np.clip((130 - s_chan) / 90.0, 0, 1) * 20

        # 4c  Low ExG contribution (0-20 pts)
        #     ExG in [8, -20] maps linearly to 0-20
        score += np.clip((self.EXG_VEG_THRESH - exg) / 28.0, 0, 1) * 20

        # 4d  Texture contribution (0-20 pts)
        #     local_std in [5, 25] maps linearly to 0-20
        score += np.clip((local_std - 5) / 20.0, 0, 1) * 20

        # 4e  Edge density contribution (0-20 pts)
        #     edge_density in [0.03, 0.20] maps linearly to 0-20
        score += np.clip((edge_density - 0.03) / 0.17, 0, 1) * 20

        # ---- 5. Apply candidate mask and threshold ----
        score[~candidate] = 0
        buildup_mask = (score >= self.SCORE_THRESH).astype(np.uint8)

        # ---- 6. Morphological cleanup ----
        buildup_mask = self.cleanup_mask(
            buildup_mask, kernel_size=5, min_blob_area=150
        )

        total = buildup_mask.size
        built = int(np.sum(buildup_mask > 0))
        pct = (built / total) * 100.0

        logger.info(f"BuildingDetector: {pct:.1f}% built-up ({built}/{total} px)")
        return buildup_mask, pct

    def detect_new_buildings(
        self,
        before_img: Image.Image,
        after_img: Image.Image,
        pixel_resolution: float = 10.0
    ) -> Dict:
        """
        Detect NEW built-up areas that appeared between before and after images.

        Returns a dict with:
        - before/after coverage stats
        - overlay images showing new construction
        - area calculations in hectares/acres
        """
        print("🏗️  BUILDING DETECTION: Analyzing before image...", flush=True)
        before_mask, before_pct = self.detect(before_img)

        print("🏗️  BUILDING DETECTION: Analyzing after image...", flush=True)
        after_mask, after_pct = self.detect(after_img)

        # Ensure same size
        if before_mask.shape != after_mask.shape:
            after_mask = cv2.resize(
                after_mask, (before_mask.shape[1], before_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # ---- Find NEW buildings (in after but NOT in before) ----
        new_buildings = ((after_mask > 0) & (before_mask == 0)).astype(np.uint8)
        new_buildings = self.cleanup_mask(new_buildings, kernel_size=7, min_blob_area=200)

        # ---- Find demolished/removed buildings ----
        removed_buildings = ((before_mask > 0) & (after_mask == 0)).astype(np.uint8)
        removed_buildings = self.cleanup_mask(removed_buildings, kernel_size=7, min_blob_area=200)

        # ---- Area calculations ----
        pixel_area_sqm = pixel_resolution ** 2
        new_pixels = int(np.sum(new_buildings))
        removed_pixels = int(np.sum(removed_buildings))

        new_hectares = (new_pixels * pixel_area_sqm) / 10000
        removed_hectares = (removed_pixels * pixel_area_sqm) / 10000
        before_hectares = (int(np.sum(before_mask)) * pixel_area_sqm) / 10000
        after_hectares = (int(np.sum(after_mask)) * pixel_area_sqm) / 10000

        change_pct = after_pct - before_pct

        # ---- Count distinct new building clusters ----
        num_new_clusters, _, new_stats, _ = cv2.connectedComponentsWithStats(
            new_buildings, connectivity=8
        )
        num_new_clusters -= 1  # Subtract background

        print(f"✅ New built-up zones: {num_new_clusters}", flush=True)
        print(f"✅ New area: {new_hectares:.1f} ha | Removed: {removed_hectares:.1f} ha", flush=True)
        print(f"✅ Before: {before_pct:.1f}% | After: {after_pct:.1f}% | Change: {change_pct:+.1f}%", flush=True)

        # ---- Create overlay images ----

        # Before: show existing buildings in blue
        before_overlay = self.create_overlay(
            before_img, before_mask,
            fill_color=(0, 120, 255), contour_color=(0, 80, 220),
            fill_alpha=0.2, contour_thickness=2
        )

        # After: show all buildings in light gray, NEW ones in bright orange
        after_arr = np.array(after_img.convert("RGB"))
        result = after_arr.copy()

        # Existing buildings (were already there): subtle blue
        existing = ((after_mask > 0) & (before_mask > 0)).astype(np.uint8)
        if existing.shape[:2] != result.shape[:2]:
            existing = cv2.resize(existing, (result.shape[1], result.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        if np.any(existing > 0):
            fill = result.copy()
            fill[existing > 0] = [100, 160, 220]
            result = cv2.addWeighted(result, 0.85, fill, 0.15, 0)

        # NEW buildings: bold orange fill + thick contours + numbered clusters
        if new_buildings.shape[:2] != result.shape[:2]:
            new_buildings_resized = cv2.resize(
                new_buildings, (result.shape[1], result.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            new_buildings_resized = new_buildings

        if np.any(new_buildings_resized > 0):
            fill = result.copy()
            fill[new_buildings_resized > 0] = [255, 140, 0]  # Orange
            result = cv2.addWeighted(result, 0.65, fill, 0.35, 0)
            contours, _ = cv2.findContours(
                new_buildings_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, (255, 100, 0), 3)

            # Number each new-construction cluster
            for i in range(1, num_new_clusters + 1):
                cx = int(new_stats[i, cv2.CC_STAT_LEFT] + new_stats[i, cv2.CC_STAT_WIDTH] / 2)
                cy = int(new_stats[i, cv2.CC_STAT_TOP] + new_stats[i, cv2.CC_STAT_HEIGHT] / 2)
                if new_buildings.shape != new_buildings_resized.shape:
                    sx = result.shape[1] / new_buildings.shape[1]
                    sy = result.shape[0] / new_buildings.shape[0]
                    cx, cy = int(cx * sx), int(cy * sy)
                cv2.putText(result, str(i), (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(result, str(i), (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        after_overlay = Image.fromarray(result)

        # ---- Difference view: dark background + bright change zones + numbers ----
        diff_img = np.zeros_like(after_arr)   # black canvas

        # Faintly show the after image so spatial context is visible
        diff_img = cv2.addWeighted(diff_img, 1.0, after_arr, 0.25, 0)

        # Draw existing buildings in dim blue
        if np.any(existing > 0):
            diff_img[existing > 0] = [40, 80, 140]

        # Draw new buildings in bright orange with thick contours
        if np.any(new_buildings_resized > 0):
            diff_img[new_buildings_resized > 0] = [255, 160, 30]
            cnts, _ = cv2.findContours(
                new_buildings_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(diff_img, cnts, -1, (255, 255, 255), 2)

            # Number each cluster on the diff view
            for i in range(1, num_new_clusters + 1):
                cx = int(new_stats[i, cv2.CC_STAT_LEFT] + new_stats[i, cv2.CC_STAT_WIDTH] / 2)
                cy = int(new_stats[i, cv2.CC_STAT_TOP] + new_stats[i, cv2.CC_STAT_HEIGHT] / 2)
                if new_buildings.shape != new_buildings_resized.shape:
                    sx = result.shape[1] / new_buildings.shape[1]
                    sy = result.shape[0] / new_buildings.shape[0]
                    cx, cy = int(cx * sx), int(cy * sy)
                # White text with black outline for readability
                cv2.putText(diff_img, str(i), (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(diff_img, str(i), (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Draw removed buildings in green (if any)
        if removed_pixels > 0:
            rm = removed_buildings
            if rm.shape[:2] != diff_img.shape[:2]:
                rm = cv2.resize(rm, (diff_img.shape[1], diff_img.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            diff_img[rm > 0] = [0, 200, 80]

        diff_overlay = Image.fromarray(diff_img)

        # Removed buildings overlay (on before image)
        if removed_pixels > 0:
            removed_overlay_resized = removed_buildings
            if removed_buildings.shape[:2] != np.array(before_img.convert("RGB")).shape[:2]:
                removed_overlay_resized = cv2.resize(
                    removed_buildings,
                    (np.array(before_img.convert("RGB")).shape[1],
                     np.array(before_img.convert("RGB")).shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            removed_vis = self.create_overlay(
                before_img, removed_overlay_resized,
                fill_color=(0, 200, 80), contour_color=(0, 160, 60),
                fill_alpha=0.3, contour_thickness=2
            )
        else:
            removed_vis = before_overlay

        return {
            "status": "success",
            "feature": "new_buildings",
            "before": {
                "buildup_percent": round(before_pct, 2),
                "buildup_hectares": round(before_hectares, 2),
                "buildup_acres": round(before_hectares * 2.47105, 2),
                "overlay_image": self.image_to_base64(before_overlay),
            },
            "after": {
                "buildup_percent": round(after_pct, 2),
                "buildup_hectares": round(after_hectares, 2),
                "buildup_acres": round(after_hectares * 2.47105, 2),
                "overlay_image": self.image_to_base64(after_overlay),
            },
            "change": {
                "percent_change": round(change_pct, 2),
                "new_buildup_hectares": round(new_hectares, 2),
                "new_buildup_acres": round(new_hectares * 2.47105, 2),
                "removed_hectares": round(removed_hectares, 2),
                "net_change_hectares": round(new_hectares - removed_hectares, 2),
                "new_clusters": num_new_clusters,
                "new_pixels": new_pixels,
                "removed_pixels": removed_pixels,
            },
            "overlays": {
                "new_buildings": self.image_to_base64(after_overlay),
                "removed_buildings": self.image_to_base64(removed_vis),
                "difference": self.image_to_base64(diff_overlay),
            }
        }


# ============================================================
# SINGLETON ACCESSORS
# ============================================================

_building_detector = None


def get_building_detector() -> BuildingDetector:
    global _building_detector
    if _building_detector is None:
        _building_detector = BuildingDetector()
    return _building_detector
