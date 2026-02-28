"""
Image processing and enhancement for satellite imagery
Handles image quality improvement and change detection
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process and enhance satellite images for better ML analysis"""
    
    @staticmethod
    def enhance_image(image: Image.Image, sharpen: bool = True, denoise: bool = True) -> Image.Image:
        """
        Enhance image quality using OpenCV
        
        Args:
            image: PIL Image to enhance
            sharpen: Apply sharpening filter
            denoise: Apply denoising
            
        Returns:
            Enhanced PIL Image
        """
        # Convert PIL to OpenCV
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Denoise
        if denoise:
            img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
        
        # Sharpen
        if sharpen:
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            img_cv = cv2.filter2D(img_cv, -1, kernel)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    
    # ----------------------------------------------------------------
    # Land-cover labels used by per-pixel classification
    # ----------------------------------------------------------------
    CATEGORY_VEGETATION   = 1
    CATEGORY_BUILT_UP     = 2
    CATEGORY_BARE_SOIL    = 3
    CATEGORY_WATER        = 4
    CATEGORY_OTHER        = 5

    # Colours (RGB) for each change category in the overlay
    CHANGE_COLORS = {
        'new_construction':   (255, 100,   0),   # orange
        'vegetation_loss':    (220, 180,  30),   # yellow
        'new_vegetation':     (  0, 200,  80),   # green
        'water_change':       ( 30, 120, 255),   # blue
        'demolition':         (180,  60, 220),   # purple
        'other_change':       (200, 200, 200),   # light gray
    }

    # ----------------------------------------------------------------
    # Per-pixel land cover classifier (TRUE-COLOR Sentinel-2 imagery)
    # ----------------------------------------------------------------
    @staticmethod
    def _classify_pixel_landcover(rgb: np.ndarray) -> np.ndarray:
        """
        Classify every pixel as vegetation / built-up / bare soil / water / other.

        Works on TRUE-COLOR RGB imagery using:
          * Excess Green Index (ExG = 2G - R - B)  -> vegetation
          * HSV brightness (V) and saturation (S)  -> built-up vs water

        Returns an int array the same shape as the first two dims of rgb.
        """
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        exg = 2 * g - r - b

        cat = np.full(rgb.shape[:2], ImageProcessor.CATEGORY_OTHER, dtype=np.uint8)

        # Vegetation: high excess green
        cat[exg > 10] = ImageProcessor.CATEGORY_VEGETATION

        # Water: very dark + bluish hue
        cat[(v < 30) | ((v < 55) & (s > 50) & (h > 90) & (h < 130))] = ImageProcessor.CATEGORY_WATER

        # Built-up: non-vegetation, bright, grayish (low saturation)
        built_mask = (exg <= 10) & (v > 55) & (s < 115) & (v > 20)
        cat[built_mask] = ImageProcessor.CATEGORY_BUILT_UP

        # Bare soil: non-vegetation, moderate brightness, warmer hue (brownish/tan)
        bare_mask = (exg <= 10) & (v > 25) & (s >= 60) & (h < 35)
        # Exclude anything already classified as built-up
        bare_mask = bare_mask & (cat != ImageProcessor.CATEGORY_BUILT_UP)
        cat[bare_mask] = ImageProcessor.CATEGORY_BARE_SOIL

        return cat

    # ----------------------------------------------------------------
    # Main change detection  (colour-coded, multi-category)
    # ----------------------------------------------------------------
    @staticmethod
    def detect_changes(
        before_img: Image.Image,
        after_img: Image.Image,
        sensitivity: float = 30.0,
        pixel_resolution: float = 10.0,
    ) -> Dict:
        """
        Detect and classify structural changes between two Sentinel-2
        TRUE-COLOR satellite images.

        Produces a colour-coded change map:
          Orange  - New construction / urbanisation
          Yellow  - Vegetation loss (cleared land)
          Green   - New vegetation / re-greening
          Blue    - Water body change
          Purple  - Demolition / urban removal
          Gray    - Other / unclassified change

        Returns dict with change metrics, PIL overlay images, category breakdown,
        and base64-encoded overlay images.
        """
        import io, base64

        # ---- 0. Preparation ----
        if before_img.size != after_img.size:
            after_img = after_img.resize(before_img.size, Image.LANCZOS)

        before = np.array(before_img.convert("RGB")).astype(np.float32)
        after  = np.array(after_img.convert("RGB")).astype(np.float32)

        # ---- 1. Histogram-match after -> before (per channel) ----
        after_norm = np.empty_like(after)
        for c in range(3):
            bm, bs = before[:, :, c].mean(), before[:, :, c].std() + 1e-6
            am, as_ = after[:, :, c].mean(), after[:, :, c].std() + 1e-6
            after_norm[:, :, c] = (after[:, :, c] - am) * (bs / as_) + bm
        after_norm = np.clip(after_norm, 0, 255)

        before_u8 = before.astype(np.uint8)
        after_u8  = after_norm.astype(np.uint8)

        # ---- 2. Change magnitude ----
        # 2a  Per-channel absolute difference (max across R,G,B)
        ch_diff = np.max(np.abs(before - after_norm), axis=2)

        # 2b  Structural (edge) difference
        bg = cv2.equalizeHist(cv2.cvtColor(before_u8, cv2.COLOR_RGB2GRAY))
        ag = cv2.equalizeHist(cv2.cvtColor(after_u8,  cv2.COLOR_RGB2GRAY))
        be = cv2.dilate(cv2.Canny(bg, 50, 150), np.ones((3, 3), np.uint8))
        ae = cv2.dilate(cv2.Canny(ag, 50, 150), np.ones((3, 3), np.uint8))
        edge_diff = cv2.absdiff(be, ae).astype(np.float32) / 255.0

        # 2c  Smoothed pixel diff (suppress speckle)
        px_diff = gaussian_filter(ch_diff, sigma=2.0)

        # ---- 3. Land-cover transition approach ----
        # Classify both images into land-cover categories
        lc_before = ImageProcessor._classify_pixel_landcover(before_u8)
        lc_after  = ImageProcessor._classify_pixel_landcover(after.astype(np.uint8))

        # Primary signal: pixels where land-cover category changed
        lc_changed = (lc_before != lc_after).astype(np.uint8) * 255

        # Clean the LC-change mask: median filter removes scattered noise
        lc_clean = cv2.medianBlur(lc_changed, 5)
        # Close small gaps
        lc_clean = cv2.morphologyEx(lc_clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Secondary signal: high pixel diff areas even if LC didn't change
        high_diff = (px_diff > 50).astype(np.uint8) * 255

        # Use LC-clean as the change mask base
        change_mask = np.maximum(lc_clean, high_diff)

        # Remove tiny blobs (< 200 px)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(change_mask, 8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < 200:
                change_mask[labels == i] = 0

        # ---- 4. Classify changed pixels ----
        # Only meaningful categories shown; "other" is suppressed

        V = ImageProcessor.CATEGORY_VEGETATION
        B = ImageProcessor.CATEGORY_BUILT_UP
        S = ImageProcessor.CATEGORY_BARE_SOIL
        W = ImageProcessor.CATEGORY_WATER

        # Category map (same shape, 0 = no change)
        category_map = np.zeros(change_mask.shape, dtype=np.uint8)
        changed = change_mask > 0

        # New construction: (veg / bare / water) -> built-up
        category_map[changed & (lc_before != B) & (lc_after == B)] = 1
        # Vegetation loss: vegetation -> (bare / other) but NOT built-up
        category_map[changed & (lc_before == V) & (lc_after != V) & (lc_after != B)] = 2
        # New vegetation: non-veg -> vegetation
        category_map[changed & (lc_before != V) & (lc_after == V)] = 3
        # Water change: gain / loss of water
        category_map[changed & ((lc_before == W) != (lc_after == W))] = 4
        # Demolition: built-up -> vegetation / bare
        category_map[changed & (lc_before == B) & (lc_after != B)] = 5
        # Remaining changed pixels — mark as 'other' for stats but remove from overlay
        category_map[changed & (category_map == 0)] = 6

        # Remove "other" from the change mask — only show clearly classified changes
        change_mask[category_map == 6] = 0

        CAT_LABELS = {
            1: 'new_construction',
            2: 'vegetation_loss',
            3: 'new_vegetation',
            4: 'water_change',
            5: 'demolition',
        }
        CAT_DISPLAY = {
            1: 'New Construction',
            2: 'Vegetation Loss',
            3: 'New Vegetation',
            4: 'Water Change',
            5: 'Demolition',
        }

        # ---- 5. Build colour-coded overlay ----
        after_orig = np.array(after_img.convert("RGB"))
        overlay_img = after_orig.copy()

        for cat_id, cat_name in CAT_LABELS.items():
            mask_cat = (category_map == cat_id).astype(np.uint8)
            if np.sum(mask_cat) == 0:
                continue
            color = ImageProcessor.CHANGE_COLORS[cat_name]
            # Semi-transparent fill (30%)
            fill = overlay_img.copy()
            fill[mask_cat > 0] = color
            overlay_img = cv2.addWeighted(overlay_img, 0.70, fill, 0.30, 0)
            # Category contour
            cnts, _ = cv2.findContours(mask_cat, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, cnts, -1, color, 2)

        # Also overlay on the before image for comparison
        before_overlay = np.array(before_img.convert("RGB")).copy()
        for cat_id, cat_name in CAT_LABELS.items():
            mask_cat = (category_map == cat_id).astype(np.uint8)
            if np.sum(mask_cat) == 0:
                continue
            color = ImageProcessor.CHANGE_COLORS[cat_name]
            fill = before_overlay.copy()
            fill[mask_cat > 0] = color
            before_overlay = cv2.addWeighted(before_overlay, 0.70, fill, 0.30, 0)
            cnts, _ = cv2.findContours(mask_cat, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(before_overlay, cnts, -1, color, 2)

        # ---- 6. Statistics ----
        total_pixels   = int(change_mask.size)
        changed_pixels = int(np.count_nonzero(change_mask))
        change_pct     = round((changed_pixels / total_pixels) * 100, 2)
        pixel_area_ha  = (pixel_resolution ** 2) / 10_000
        changed_hectares = round(changed_pixels * pixel_area_ha, 2)

        category_stats = {}
        for cat_id, cat_name in CAT_LABELS.items():
            n = int(np.sum(category_map == cat_id))
            if n > 0:
                category_stats[cat_name] = {
                    'pixels': n,
                    'percent': round((n / total_pixels) * 100, 2),
                    'hectares': round(n * pixel_area_ha, 2),
                    'label': CAT_DISPLAY[cat_id],
                    'color': 'rgb({},{},{})'.format(*ImageProcessor.CHANGE_COLORS[cat_name]),
                }

        if change_pct < 2:
            severity = 'Minimal'
        elif change_pct < 8:
            severity = 'Low'
        elif change_pct < 20:
            severity = 'Medium'
        elif change_pct < 40:
            severity = 'High'
        else:
            severity = 'Very High'

        # Dominant change
        dominant = 'No significant change'
        if category_stats:
            top = max(category_stats.values(), key=lambda x: x['pixels'])
            dominant = top['label']

        # ---- 7. Base64 encode overlays ----
        def pil_to_b64(img_array):
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format='JPEG', quality=92)
            buf.seek(0)
            return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

        overlay_pil = Image.fromarray(overlay_img)
        before_overlay_pil = Image.fromarray(before_overlay)

        changes_detected = {
            'total_pixels':         total_pixels,
            'changed_pixels':       changed_pixels,
            'change_percentage':    change_pct,
            'change_area_hectares': changed_hectares,
            'overlay_image':        overlay_pil,
            'before_overlay_image': before_overlay_pil,
            'change_mask':          Image.fromarray(change_mask),
            'severity':             severity,
            'change_type':          dominant,
            'confidence':           'High' if change_pct > 5 else 'Medium',
            'categories':           category_stats,
            'after_overlay_b64':    pil_to_b64(overlay_img),
            'before_overlay_b64':   pil_to_b64(before_overlay),
        }

        logger.info(
            f"Change detection: {change_pct:.1f}% changed, "
            f"severity={severity}, dominant={dominant}, "
            f"categories={list(category_stats.keys())}"
        )

        return changes_detected

    @staticmethod
    def calculate_ndvi(image: Image.Image, nir_band: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Calculate NDVI (Normalized Difference Vegetation Index)
        Note: Requires NIR band data, not available in true color images
        
        Returns:
            NDVI array or None if NIR not available
        """
        if nir_band is None:
            logger.warning("NDVI calculation requires NIR band, not available in true color")
            return None
        
        img_array = np.array(image)
        red = img_array[:, :, 0].astype(float)
        
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = (nir_band - red) / (nir_band + red + 1e-8)
        
        return ndvi
    
    @staticmethod
    def super_resolution(image: Image.Image, scale: int = 2) -> Image.Image:
        """
        Apply super-resolution using OpenCV DNN
        
        Args:
            image: Input PIL image
            scale: Upscaling factor (2 or 4)
            
        Returns:
            Super-resolved PIL image
        """
        try:
            # Convert to OpenCV
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Simple bicubic upscaling (fallback method)
            # For production, use ESRGAN or Real-ESRGAN models
            h, w = img_cv.shape[:2]
            img_cv = cv2.resize(img_cv, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            
            # Convert back to PIL
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
            
        except Exception as e:
            logger.error(f"Super-resolution failed: {e}")
            return image
    
    @staticmethod
    def check_image_quality(image: Image.Image) -> Dict:
        """
        Check if image is usable for analysis
        
        Returns:
            Quality metrics dictionary
        """
        img_array = np.array(image)
        
        # Calculate statistics
        std_dev = float(np.std(img_array))
        mean_intensity = float(np.mean(img_array))
        
        # Check if image is completely blank (very strict threshold)
        # Only reject if std_dev < 1 (essentially uniform color)
        if std_dev < 1:
            return {
                'is_valid': False,
                'reason': 'Image is completely blank (uniform color)',
                'std_dev': std_dev,
                'mean_intensity': mean_intensity
            }
        
        # Check for reasonable variance
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate blur score using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_blurry = bool(blur_score < 100)  # Convert numpy.bool_ to Python bool
        
        return {
            'is_valid': True,
            'blur_score': float(blur_score),
            'is_blurry': is_blurry,
            'std_dev': std_dev,
            'mean_intensity': mean_intensity,
            'quality': 'Good' if blur_score > 300 else 'Acceptable' if blur_score > 100 else 'Poor'
        }
