"""
AI-Based Urban Detection Module
Uses advanced color analysis for satellite false-color imagery
Optimized for Sentinel-2 false-color composites (NIR-Red-Green)
"""
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import io
import base64
import cv2

logger = logging.getLogger(__name__)


class UrbanDetector:
    """
    Urban area detection using SegFormer pre-trained model
    Model: nvidia/segformer-b0-finetuned-ade-512-512
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        # ADE20K urban-related class IDs (0-indexed, 150 classes total)
        # Based on ADE20K official class mapping
        # Source: https://github.com/CSAILVision/ADE20K
        self.urban_classes = [
            1,   # building
            6,   # road
            11,  # sidewalk
            49,  # house  
            74,  # skyscraper
            60,  # bridge
            0,   # wall (often appears in urban areas)
        ]
        
        logger.info(f"UrbanDetector initialized on device: {self.device}")
    
    def load_model(self):
        """Load SegFormer model from Hugging Face"""
        try:
            logger.info("Loading SegFormer model...")
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✓ SegFormer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Image loaded: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def detect_urban_areas(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """
        Advanced urban detection for FALSE-COLOR satellite imagery
        Uses multi-spectral analysis optimized for Sentinel-2 composites
        
        For Sentinel-2 False-Color (NIR-Red-Green):
        - Channel 0 (Red in display) = NIR band (high for vegetation)
        - Channel 1 (Green in display) = Red band
        - Channel 2 (Blue in display) = Green band
        
        Urban characteristics in false-color:
        - LOW NIR (appears dark/gray, not red)
        - Uniform texture
        - Medium brightness (not too dark like water, not bright like vegetation)
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            print(f"🖼️  Image shape: {img_array.shape}", flush=True)
            print(f"🎨 Image dtype: {img_array.dtype}", flush=True)
            
            if len(img_array.shape) != 3 or img_array.shape[2] < 3:
                raise ValueError("Image must be RGB/false-color composite")
            
            # Extract channels (assuming NIR-R-G false-color composite)
            nir = img_array[:, :, 0].astype(float)  # Red channel = NIR
            red = img_array[:, :, 1].astype(float)  # Green channel = Red
            green = img_array[:, :, 2].astype(float)  # Blue channel = Green
            
            print(f"📊 NIR range: {nir.min():.1f} to {nir.max():.1f}, mean: {nir.mean():.1f}", flush=True)
            print(f"📊 Red range: {red.min():.1f} to {red.max():.1f}, mean: {red.mean():.1f}", flush=True)
            print(f"📊 Green range: {green.min():.1f} to {green.max():.1f}, mean: {green.mean():.1f}", flush=True)
            
            # Method 1: NDVI-like index (vegetation indicator)
            # High vegetation = high NIR, low in urban areas
            # NDVI = (NIR - Red) / (NIR + Red)
            epsilon = 1e-8
            ndvi = (nir - red) / (nir + red + epsilon)
            
            # Urban areas have LOW or negative NDVI
            # Vegetation has high positive NDVI (> 0.2)
            # Urban typically: -0.1 to 0.2
            # Water: negative NDVI
            urban_from_ndvi = (ndvi < 0.25) & (ndvi > -0.3)  # Exclude water
            
            print(f"📊 NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}", flush=True)
            print(f"🔍 Low NDVI pixels (potential urban): {np.sum(urban_from_ndvi)}", flush=True)
            
            # Method 2: NIR threshold
            # Vegetation appears BRIGHT RED (high NIR > 100)
            # Urban appears DARK/GRAY (low NIR < 100)
            nir_threshold = np.percentile(nir, 40)  # Adapt to image
            urban_from_nir = nir < nir_threshold
            
            print(f"🔍 NIR threshold (40th percentile): {nir_threshold:.1f}", flush=True)
            print(f"🔍 Low NIR pixels: {np.sum(urban_from_nir)}", flush=True)
            
            # Method 3: Brightness check (exclude very dark water/shadows)
            grayscale = np.mean(img_array, axis=2)
            brightness_mask = (grayscale > 30) & (grayscale < 150)  # Urban range
            
            print(f"🔍 Medium brightness pixels: {np.sum(brightness_mask)}", flush=True)
            
            # Method 4: Texture analysis (urban areas have more edges)
            gray_uint8 = grayscale.astype(np.uint8)
            edges = cv2.Canny(gray_uint8, 30, 100)
            edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            texture_mask = edges_dilated > 0
            
            print(f"🔍 High texture pixels (edges): {np.sum(texture_mask)}", flush=True)
            
            # COMBINE ALL METHODS with weighted voting
            # Core urban detection: Low NDVI + Low NIR + Medium Brightness
            core_urban = urban_from_ndvi & urban_from_nir & brightness_mask
            
            # Expand with texture (buildings have sharp edges)
            urban_mask = core_urban | (urban_from_nir & brightness_mask & texture_mask)
            
            # Clean up noise
            kernel = np.ones((3,3), np.uint8)
            urban_mask_clean = cv2.morphologyEx(urban_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            urban_mask_clean = cv2.morphologyEx(urban_mask_clean, cv2.MORPH_OPEN, kernel)
            
            # Calculate urban percentage
            total_pixels = urban_mask_clean.size
            urban_pixels = np.sum(urban_mask_clean)
            urban_percentage = (urban_pixels / total_pixels) * 100
            
            print(f"✓ Urban detection: {urban_percentage:.2f}% ({urban_pixels}/{total_pixels} pixels)", flush=True)
            print(f"✓ Methods used: NDVI + NIR + Brightness + Texture", flush=True)
            
            logger.info(f"Urban detection complete: {urban_percentage:.2f}% urban (multi-spectral)")
            
            return urban_mask_clean, urban_percentage
            
        except Exception as e:
            print(f"❌ Urban detection failed: {e}", flush=True)
            logger.error(f"Urban detection failed: {e}")
            raise
    
    def create_overlay_image(self, original_image: Image.Image, mask: np.ndarray, 
                            color=(255, 0, 0), alpha=0.5) -> Image.Image:
        """
        Create an overlay image with urban areas highlighted
        """
        # Convert mask to RGB overlay
        overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
        overlay[mask == 1] = color  # Red for urban areas
        
        # Convert to PIL Image
        overlay_img = Image.fromarray(overlay, mode='RGB')
        
        # Resize overlay to match original image
        overlay_img = overlay_img.resize(original_image.size, Image.BILINEAR)
        
        # Blend with original image
        blended = Image.blend(original_image, overlay_img, alpha)
        
        return blended
    
    def mask_to_base64(self, mask: np.ndarray) -> str:
        """Convert numpy mask to base64 encoded PNG"""
        # Convert binary mask to image (0 = black, 255 = white)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        
        # Save to bytes
        buffer = io.BytesIO()
        mask_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def overlay_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 encoded JPEG"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    
    def analyze_urban_change(self, before_image_path: str, after_image_path: str,
                            pixel_resolution: float = 10.0) -> Dict:
        """
        Analyze urban change between two satellite images
        
        Args:
            before_image_path: Path to before image
            after_image_path: Path to after image
            pixel_resolution: Resolution in meters per pixel (default: 10m for Sentinel-2)
        
        Returns:
            Dictionary with analysis results
        """
        print(f"\n🔬 STARTING URBAN CHANGE ANALYSIS", flush=True)
        print(f"Before: {before_image_path}", flush=True)
        print(f"After: {after_image_path}", flush=True)
        
        logger.info("Starting urban change analysis...")
        
        # Load images
        print("📸 Loading images...", flush=True)
        before_img = self.preprocess_image(before_image_path)
        after_img = self.preprocess_image(after_image_path)
        print(f"✓ Images loaded: {before_img.size}", flush=True)
        
        # Detect urban areas
        before_mask, before_percent = self.detect_urban_areas(before_img)
        after_mask, after_percent = self.detect_urban_areas(after_img)
        
        # Calculate change
        change_percent = after_percent - before_percent
        
        # Calculate area in different units
        total_pixels = before_mask.size
        pixel_area_sqm = pixel_resolution ** 2
        
        # Use int64 to prevent overflow
        before_urban_pixels = int(np.sum(before_mask))
        after_urban_pixels = int(np.sum(after_mask))
        change_pixels = after_urban_pixels - before_urban_pixels
        
        before_area_sqm = before_urban_pixels * pixel_area_sqm
        after_area_sqm = after_urban_pixels * pixel_area_sqm
        change_area_sqm = change_pixels * pixel_area_sqm
        
        # Convert to hectares
        before_hectares = before_area_sqm / 10000
        after_hectares = after_area_sqm / 10000
        change_hectares = change_area_sqm / 10000
        
        # Convert to acres
        before_acres = before_hectares * 2.47105
        after_acres = after_hectares * 2.47105
        change_acres = change_hectares * 2.47105
        
        # Create overlay images
        before_overlay = self.create_overlay_image(before_img, before_mask, color=(255, 0, 0), alpha=0.4)
        after_overlay = self.create_overlay_image(after_img, after_mask, color=(255, 0, 0), alpha=0.4)
        
        # Create change visualization (new urban areas)
        new_urban_mask = (after_mask > before_mask).astype(np.uint8)
        change_overlay = self.create_overlay_image(after_img, new_urban_mask, color=(255, 165, 0), alpha=0.5)
        
        # Convert to base64 for web display
        results = {
            "status": "success",
            "before": {
                "urban_percent": round(before_percent, 2),
                "urban_hectares": round(before_hectares, 2),
                "urban_acres": round(before_acres, 2),
                "overlay_image": self.overlay_to_base64(before_overlay)
            },
            "after": {
                "urban_percent": round(after_percent, 2),
                "urban_hectares": round(after_hectares, 2),
                "urban_acres": round(after_acres, 2),
                "overlay_image": self.overlay_to_base64(after_overlay)
            },
            "change": {
                "percent_change": round(change_percent, 2),
                "hectares_change": round(change_hectares, 2),
                "acres_change": round(change_acres, 2),
                "growth_rate": round((change_percent / before_percent * 100) if before_percent > 0 else 0, 2),
                "change_overlay": self.overlay_to_base64(change_overlay),
                "new_urban_pixels": int(np.sum(new_urban_mask))
            }
        }
        
        logger.info(f"Urban analysis complete: {change_percent:+.2f}% change")
        
        return results


# Singleton instance
_urban_detector = None

def get_urban_detector() -> UrbanDetector:
    """Get or create singleton UrbanDetector instance"""
    global _urban_detector
    if _urban_detector is None:
        _urban_detector = UrbanDetector()
    return _urban_detector
