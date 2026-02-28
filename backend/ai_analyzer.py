"""
AI-Based Urban Detection Module
Uses SegFormer pre-trained model for semantic segmentation
Model: nvidia/segformer-b0-finetuned-ade-512-512 (ADE20K, 150 classes)
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
    Urban area detection using SegFormer semantic segmentation.
    Uses nvidia/segformer-b0-finetuned-ade-512-512 trained on ADE20K (150 classes).
    Identifies buildings, roads, sidewalks, houses etc. as 'urban'.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        # ADE20K urban-related class IDs (0-indexed)
        # Full list: https://github.com/CSAILVision/ADE20K
        self.urban_classes = [
            0,   # wall
            1,   # building
            6,   # road
            11,  # sidewalk, pavement
            13,  # path
            28,  # house
            48,  # skyscraper
            52,  # bridge
            53,  # tower
            54,  # awning
            64,  # fence
            84,  # truck
            90,  # car
            102, # bus
            127, # parking
        ]
        
        logger.info(f"UrbanDetector initialized (device: {self.device})")
    
    def load_model(self):
        """Load SegFormer model from Hugging Face"""
        try:
            logger.info(f"Loading SegFormer model: {self.model_name}...")
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✓ SegFormer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SegFormer model: {e}")
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
    
    def _false_color_to_rgb(self, image: Image.Image) -> Image.Image:
        """
        Convert Sentinel-2 false-color composite to pseudo-true-color for SegFormer.
        
        False-color (NIR-Red-Green displayed as R-G-B):
          Channel 0 (displayed Red) = NIR band
          Channel 1 (displayed Green) = Red band
          Channel 2 (displayed Blue) = Green band
        
        Pseudo true-color:
          New R = channel 1 (actual Red)
          New G = channel 2 (actual Green)  
          New B = (channel 2 * 0.85) (approximate Blue)
        """
        img_array = np.array(image)
        
        # Check if this looks like false-color (Red channel has high vegetation signal)
        r_mean = img_array[:, :, 0].mean()
        g_mean = img_array[:, :, 1].mean()
        b_mean = img_array[:, :, 2].mean()
        
        # In false-color, red channel (NIR) and green channel (Red) tend to be 
        # significantly different from blue channel (Green)
        is_false_color = (r_mean > b_mean * 1.3) or (g_mean > b_mean * 1.3)
        
        if is_false_color:
            pseudo_rgb = np.stack([
                img_array[:, :, 1],  # Actual Red band
                img_array[:, :, 2],  # Actual Green band
                np.clip(img_array[:, :, 2].astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
            ], axis=2)
            logger.info("Converted false-color → pseudo-true-color for SegFormer")
            return Image.fromarray(pseudo_rgb)
        else:
            logger.info("Image appears to be true-color, using as-is")
            return image
    
    def detect_urban_areas(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """
        Detect urban areas using SegFormer semantic segmentation.
        
        The model segments the image into 150 ADE20K classes, then we
        combine urban-related classes (building, road, sidewalk, etc.)
        into a single urban mask.
        
        Returns:
            (urban_mask, urban_percentage) - binary mask and % of urban pixels
        """
        try:
            # Load model on first call
            if self.model is None:
                self.load_model()
            
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            print(f"🖼️  Image shape: {img_array.shape}", flush=True)
            
            # Convert false-color to pseudo-true-color for better model performance
            model_input = self._false_color_to_rgb(image)
            
            # Run SegFormer inference
            print("🧠 Running SegFormer semantic segmentation...", flush=True)
            inputs = self.processor(images=model_input, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Upsample logits to original image size
            logits = outputs.logits  # shape: (1, num_classes, H/4, W/4)
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )
            
            # Get per-pixel class predictions
            seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
            
            # Create urban mask: 1 where any urban class is detected
            urban_mask = np.isin(seg_map, self.urban_classes).astype(np.uint8)
            
            # Morphological cleanup
            kernel = np.ones((5, 5), np.uint8)
            urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_CLOSE, kernel)
            urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate urban percentage
            total_pixels = urban_mask.size
            urban_pixels = int(np.sum(urban_mask))
            urban_percentage = (urban_pixels / total_pixels) * 100
            
            # Log class distribution for debugging
            unique, counts = np.unique(seg_map, return_counts=True)
            top_classes = sorted(zip(unique, counts), key=lambda x: -x[1])[:5]
            print(f"📊 Top 5 detected classes: {top_classes}", flush=True)
            print(f"✅ Urban detection: {urban_percentage:.2f}% "
                  f"({urban_pixels}/{total_pixels} pixels)", flush=True)
            
            return urban_mask, urban_percentage
            
        except Exception as e:
            print(f"❌ SegFormer failed: {e}", flush=True)
            logger.error(f"SegFormer urban detection failed: {e}")
            # Fallback to improved heuristic method
            print("⚠️ Falling back to heuristic method...", flush=True)
            return self._detect_urban_heuristic(image)
    
    def _detect_urban_heuristic(self, image: Image.Image) -> Tuple[np.ndarray, float]:
        """
        Fallback heuristic urban detection when SegFormer is unavailable.
        Fixed version: proper brightness handling, consistent thresholds.
        """
        img_array = np.array(image)
        nir = img_array[:, :, 0].astype(float)
        red = img_array[:, :, 1].astype(float)
        green = img_array[:, :, 2].astype(float)
        
        # NDVI: vegetation has high NDVI, urban has low
        epsilon = 1e-8
        ndvi = (nir - red) / (nir + red + epsilon)
        
        # Urban = low NDVI (not vegetation) but not water (very negative)
        urban_from_ndvi = (ndvi < 0.2) & (ndvi > -0.2)
        
        # Brightness: urban areas can be BRIGHT (concrete, roofs) or medium
        # Key fix: DON'T cap at 150 — bright urban areas exist!
        grayscale = np.mean(img_array[:, :, :3], axis=2)
        brightness_mask = grayscale > 25  # Just exclude very dark (water/shadow)
        
        # Low NIR ratio: vegetation has HIGH NIR, urban has LOW relative NIR
        nir_ratio = nir / (grayscale + epsilon)
        low_nir = nir_ratio < 1.3  # Urban: NIR not much higher than average
        
        # Combine
        urban_mask = (urban_from_ndvi & brightness_mask & low_nir).astype(np.uint8)
        
        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_CLOSE, kernel)
        urban_mask = cv2.morphologyEx(urban_mask, cv2.MORPH_OPEN, kernel)
        
        urban_percentage = (np.sum(urban_mask) / urban_mask.size) * 100
        print(f"✅ Heuristic urban detection: {urban_percentage:.2f}%", flush=True)
        
        return urban_mask, urban_percentage
    
    def create_overlay_image(self, original_image: Image.Image, mask: np.ndarray,
                            color=(255, 0, 0), alpha=0.3) -> Image.Image:
        """
        Create a clean overlay with semi-transparent fill + contour outlines.
        Much cleaner than solid color blocks.
        """
        orig_array = np.array(original_image)
        if orig_array.shape[2] == 4:
            orig_array = orig_array[:, :, :3]  # Drop alpha
        
        # Resize mask to match image if needed
        if mask.shape[:2] != orig_array.shape[:2]:
            mask = cv2.resize(mask, (orig_array.shape[1], orig_array.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        
        result = orig_array.copy()
        
        # Semi-transparent fill
        fill = orig_array.copy()
        fill[mask > 0] = color
        result = cv2.addWeighted(orig_array, 1 - alpha, fill, alpha, 0)
        
        # Add contour outlines for clarity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        
        return Image.fromarray(result)
    
    def mask_to_base64(self, mask: np.ndarray) -> str:
        """Convert numpy mask to base64 encoded PNG"""
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        buffer = io.BytesIO()
        mask_img.save(buffer, format='PNG')
        buffer.seek(0)
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
        Analyze urban change between two satellite images using SegFormer.
        
        Returns results with:
        - Before/after urban percentages and areas
        - Clean overlay images (semi-transparent, contour-outlined)
        - Change map highlighting only NEW urban growth
        """
        print(f"\n🔬 STARTING URBAN CHANGE ANALYSIS (SegFormer)", flush=True)
        print(f"Before: {before_image_path}", flush=True)
        print(f"After: {after_image_path}", flush=True)
        
        # Load images
        before_img = self.preprocess_image(before_image_path)
        after_img = self.preprocess_image(after_image_path)
        
        # Detect urban areas in both images
        print("\n📊 Analyzing BEFORE image...", flush=True)
        before_mask, before_percent = self.detect_urban_areas(before_img)
        
        print("\n📊 Analyzing AFTER image...", flush=True)
        after_mask, after_percent = self.detect_urban_areas(after_img)
        
        # Ensure masks are same size
        if before_mask.shape != after_mask.shape:
            after_mask = cv2.resize(after_mask, (before_mask.shape[1], before_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        
        # Calculate change
        change_percent = after_percent - before_percent
        
        # Area calculations
        pixel_area_sqm = pixel_resolution ** 2
        before_urban_pixels = int(np.sum(before_mask))
        after_urban_pixels = int(np.sum(after_mask))
        
        before_hectares = (before_urban_pixels * pixel_area_sqm) / 10000
        after_hectares = (after_urban_pixels * pixel_area_sqm) / 10000
        change_hectares = after_hectares - before_hectares
        
        before_acres = before_hectares * 2.47105
        after_acres = after_hectares * 2.47105
        change_acres = change_hectares * 2.47105
        
        # Create clean overlay images
        # Before: urban areas in BLUE (30% opacity + contours)
        before_overlay = self.create_overlay_image(
            before_img, before_mask, color=(0, 120, 255), alpha=0.3
        )
        # After: urban areas in RED (30% opacity + contours)
        after_overlay = self.create_overlay_image(
            after_img, after_mask, color=(255, 60, 60), alpha=0.3
        )
        
        # Change map: highlight only NEW urban areas (growth) in orange,
        # and lost urban areas (demolition) in green
        new_urban = ((after_mask > 0) & (before_mask == 0)).astype(np.uint8)
        lost_urban = ((before_mask > 0) & (after_mask == 0)).astype(np.uint8)
        
        # Create a combined change visualization on the after image
        after_array = np.array(after_img)
        if after_array.shape[2] == 4:
            after_array = after_array[:, :, :3]
        
        if new_urban.shape[:2] != after_array.shape[:2]:
            new_urban = cv2.resize(new_urban, (after_array.shape[1], after_array.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            lost_urban = cv2.resize(lost_urban, (after_array.shape[1], after_array.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        
        change_vis = after_array.copy()
        # New urban → Orange highlight
        change_fill = change_vis.copy()
        change_fill[new_urban > 0] = [255, 140, 0]   # Orange = new construction
        change_fill[lost_urban > 0] = [0, 200, 80]    # Green = vegetation recovery
        change_vis = cv2.addWeighted(after_array, 0.6, change_fill, 0.4, 0)
        
        # Add contours
        contours_new, _ = cv2.findContours(new_urban, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_lost, _ = cv2.findContours(lost_urban, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(change_vis, contours_new, -1, (255, 100, 0), 2)
        cv2.drawContours(change_vis, contours_lost, -1, (0, 180, 60), 2)
        
        change_overlay = Image.fromarray(change_vis)
        
        growth_rate = ((change_percent / before_percent) * 100) if before_percent > 0 else 0
        
        print(f"\n{'='*50}", flush=True)
        print(f"📊 RESULTS:", flush=True)
        print(f"  Before urban: {before_percent:.2f}%", flush=True)
        print(f"  After urban:  {after_percent:.2f}%", flush=True)
        print(f"  Change:       {change_percent:+.2f}%", flush=True)
        print(f"  Growth rate:  {growth_rate:+.2f}%", flush=True)
        print(f"{'='*50}\n", flush=True)
        
        results = {
            "status": "success",
            "model": "SegFormer (nvidia/segformer-b0-finetuned-ade-512-512)",
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
                "growth_rate": round(growth_rate, 2),
                "change_overlay": self.overlay_to_base64(change_overlay),
                "new_urban_pixels": int(np.sum(new_urban))
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
