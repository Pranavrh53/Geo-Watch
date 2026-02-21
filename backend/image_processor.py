"""
Image processing and enhancement for satellite imagery
Handles image quality improvement and change detection
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional
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
    
    @staticmethod
    def detect_changes(
        before_img: Image.Image,
        after_img: Image.Image,
        sensitivity: float = 30.0
    ) -> Dict:
        """
        Detect changes between two images
        
        Args:
            before_img: Before image (PIL)
            after_img: After image (PIL)
            sensitivity: Change detection threshold (0-255)
            
        Returns:
            Dictionary with change metrics and overlay image
        """
        # Ensure same size
        if before_img.size != after_img.size:
            after_img = after_img.resize(before_img.size)
        
        # Convert to numpy arrays
        before = np.array(before_img)
        after = np.array(after_img)
        
        # Convert to grayscale for comparison
        before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(before_gray, after_gray)
        
        # Threshold to get binary change mask
        _, change_mask = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate change statistics
        total_pixels = change_mask.size
        changed_pixels = np.count_nonzero(change_mask)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Create colored overlay (red for changes)
        overlay = after.copy()
        overlay[change_mask > 0] = [255, 0, 0]  # Red for changes
        
        # Blend with original
        alpha = 0.4
        result = cv2.addWeighted(after, 1 - alpha, overlay, alpha, 0)
        
        # Detect change types using color analysis
        changes_detected = {
            'total_pixels': int(total_pixels),
            'changed_pixels': int(changed_pixels),
            'change_percentage': round(change_percentage, 2),
            'change_area_hectares': round(change_percentage * 0.01, 4),  # Rough estimate
            'overlay_image': Image.fromarray(result),
            'change_mask': Image.fromarray(change_mask),
            'severity': 'High' if change_percentage > 10 else 'Medium' if change_percentage > 5 else 'Low'
        }
        
        # Classify change types
        change_types = ImageProcessor._classify_changes(before, after, change_mask)
        changes_detected.update(change_types)
        
        return changes_detected
    
    @staticmethod
    def _classify_changes(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Classify types of changes (construction, deforestation, etc.)
        
        Returns:
            Dictionary with classified change types
        """
        # Get only changed regions
        changed_before = before[mask > 0]
        changed_after = after[mask > 0]
        
        if len(changed_before) == 0:
            return {'change_type': 'No significant change'}
        
        # Calculate average colors in changed regions
        before_avg = np.mean(changed_before.reshape(-1, 3), axis=0)
        after_avg = np.mean(changed_after.reshape(-1, 3), axis=0)
        
        # Simple heuristic classification
        # Green (vegetation): high G, low R and B
        # Gray (urban/construction): balanced RGB, high intensity
        # Blue (water): high B, low R and G
        # Brown (bare land): R > G > B
        
        before_green = (before_avg[1] > before_avg[0] * 1.1) and (before_avg[1] > before_avg[2] * 1.1)
        after_green = (after_avg[1] > after_avg[0] * 1.1) and (after_avg[1] > after_avg[2] * 1.1)
        
        before_gray = max(before_avg) - min(before_avg) < 30 and np.mean(before_avg) > 100
        after_gray = max(after_avg) - min(after_avg) < 30 and np.mean(after_avg) > 100
        
        # Determine change type
        change_type = "General change detected"
        confidence = "Low"
        
        if before_green and after_gray:
            change_type = "Possible Construction (Vegetation → Urban)"
            confidence = "Medium"
        elif before_green and not after_green:
            change_type = "Possible Deforestation (Vegetation Loss)"
            confidence = "Medium"
        elif not before_gray and after_gray:
            change_type = "Possible Urban Development"
            confidence = "Medium"
        elif after_green and not before_green:
            change_type = "Possible Greening (Vegetation Increase)"
            confidence = "Medium"
        
        return {
            'change_type': change_type,
            'confidence': confidence,
            'before_avg_color': before_avg.tolist(),
            'after_avg_color': after_avg.tolist()
        }
    
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
