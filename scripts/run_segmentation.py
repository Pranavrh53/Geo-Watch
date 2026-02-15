"""
ML Segmentation Module
Runs semantic segmentation on satellite tiles using SegFormer and DeepLabV3
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torchvision import models
import click
import logging
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR_PROCESSED, MODELS_DIR,
    MODELS, TILE_SIZE, USE_GPU,
    LAND_COVER_CLASSES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LandCoverSegmenter:
    """Semantic segmentation for land cover classification"""
    
    def __init__(self, model_name='segformer', use_gpu=USE_GPU):
        self.model_name = model_name
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained segmentation model"""
        logger.info(f"Loading {self.model_name} model...")
        
        if self.model_name == 'segformer':
            model_id = MODELS['segformer']['name']
            self.processor = SegformerImageProcessor.from_pretrained(model_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
            
        elif self.model_name == 'deeplabv3':
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"✅ Model loaded successfully")
    
    def preprocess_tile(self, tile):
        """
        Preprocess tile for model input
        
        Args:
            tile: numpy array (bands, height, width)
        
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if needed
        if tile.shape[0] >= 3:
            # Use first 3 bands as RGB
            rgb = tile[:3]
        else:
            # Duplicate single channel to RGB
            rgb = np.repeat(tile[:1], 3, axis=0)
        
        # Transpose to (height, width, channels) for processor
        rgb = np.transpose(rgb, (1, 2, 0))
        
        # Scale to 0-255 range
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        if self.model_name == 'segformer':
            # Use SegFormer processor
            inputs = self.processor(images=rgb, return_tensors="pt")
            return inputs.pixel_values.to(self.device)
        
        else:
            # Convert to tensor for DeepLabV3
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)
            return tensor
    
    def segment_tile(self, tile):
        """
        Run segmentation on a single tile
        
        Args:
            tile: numpy array (bands, height, width)
        
        Returns:
            Segmentation mask (height, width)
        """
        with torch.no_grad():
            # Preprocess
            inputs = self.preprocess_tile(tile)
            
            # Run model
            if self.model_name == 'segformer':
                outputs = self.model(pixel_values=inputs)
                logits = outputs.logits
                
                # Upsample to original size
                logits = F.interpolate(
                    logits,
                    size=(tile.shape[1], tile.shape[2]),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Get predictions
                predictions = logits.argmax(dim=1).squeeze().cpu().numpy()
            
            else:  # deeplabv3
                outputs = self.model(inputs)['out']
                predictions = outputs.argmax(dim=1).squeeze().cpu().numpy()
            
            # Map to land cover classes (simplified)
            mask = self._map_to_land_cover(predictions)
            
            return mask
    
    def _map_to_land_cover(self, predictions):
        """
        Map model predictions to our land cover classes
        
        Model outputs vary, so we map common classes to:
        0: background
        1: urban/built
        2: vegetation
        3: water
        4: bare soil
        5: road
        """
        # This is a simplified mapping - adjust based on your model
        # ADE20K has 150 classes, we map relevant ones
        
        mask = np.zeros_like(predictions, dtype=np.uint8)
        
        # Urban/Building classes (building, house, skyscraper, etc.)
        urban_classes = [1, 2, 3, 4, 25, 48]
        for cls in urban_classes:
            mask[predictions == cls] = 1
        
        # Vegetation classes (tree, grass, plant, etc.)
        vegetation_classes = [4, 5, 9, 17, 18]
        for cls in vegetation_classes:
            mask[predictions == cls] = 2
        
        # Water classes (water, sea, river, lake)
        water_classes = [21, 26, 27]
        for cls in water_classes:
            mask[predictions == cls] = 3
        
        # Soil classes (earth, ground, sand)
        soil_classes = [13, 28, 46]
        for cls in soil_classes:
            mask[predictions == cls] = 4
        
        # Road classes (road, path, runway)
        road_classes = [6, 11, 52]
        for cls in road_classes:
            mask[predictions == cls] = 5
        
        return mask
    
    def segment_tiles_batch(self, tiles_dir, output_dir):
        """
        Segment all tiles in a directory
        
        Args:
            tiles_dir: Directory containing tile NPY files
            output_dir: Directory to save segmentation masks
        """
        tiles_dir = Path(tiles_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all tile files
        tile_files = sorted(tiles_dir.glob('*.npy'))
        
        if not tile_files:
            logger.warning(f"No tiles found in {tiles_dir}")
            return
        
        logger.info(f"Segmenting {len(tile_files)} tiles")
        
        results = []
        
        for tile_file in tqdm(tile_files, desc="Segmenting tiles"):
            # Load tile
            tile = np.load(tile_file)
            
            # Segment
            mask = self.segment_tile(tile)
            
            # Save mask
            mask_file = output_dir / tile_file.name.replace('.npy', '_mask.npy')
            np.save(mask_file, mask)
            
            # Calculate class distribution
            unique, counts = np.unique(mask, return_counts=True)
            distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}
            
            results.append({
                'tile': tile_file.name,
                'mask': mask_file.name,
                'distribution': distribution
            })
        
        # Save results summary
        summary_file = output_dir.parent / 'segmentation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Segmentation complete! Masks saved to {output_dir}")
        return results


@click.command()
@click.option('--city', required=True, help='City name')
@click.option('--period', type=click.Choice(['before', 'after', 'both']), default='both')
@click.option('--model', type=click.Choice(['segformer', 'deeplabv3']), default='segformer')
def main(city, period, model):
    """
    Run semantic segmentation on processed tiles
    
    Example:
        python run_segmentation.py --city bangalore --period both --model segformer
    """
    
    city_dir = DATA_DIR_PROCESSED / city.lower()
    
    if not city_dir.exists():
        logger.error(f"Processed data not found: {city_dir}")
        logger.info(f"Please run preprocessing first: python scripts/preprocess.py --city {city}")
        return
    
    # Initialize segmenter
    segmenter = LandCoverSegmenter(model_name=model)
    
    # Find period directories
    period_dirs = []
    if period in ['before', 'both']:
        period_dirs.extend(city_dir.glob('before_*'))
    if period in ['after', 'both']:
        period_dirs.extend(city_dir.glob('after_*'))
    
    if not period_dirs:
        logger.error(f"No period directories found in {city_dir}")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SEGMENTATION: {city.upper()}")
    logger.info(f"{'='*60}")
    
    for period_dir in period_dirs:
        logger.info(f"\n🧠 Segmenting {period_dir.name}")
        
        tiles_dir = period_dir / 'tiles'
        masks_dir = period_dir / 'masks'
        
        if not tiles_dir.exists():
            logger.warning(f"Tiles directory not found: {tiles_dir}")
            continue
        
        # Run segmentation
        segmenter.segment_tiles_batch(tiles_dir, masks_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ SEGMENTATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"\nNext step: Run change detection")
    logger.info(f"  python scripts/detect_changes.py --city {city}")


if __name__ == '__main__':
    main()
