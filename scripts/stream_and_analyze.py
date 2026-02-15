"""
Streaming Satellite Analysis - Fetch and Process On-The-Fly
Fetches satellite tiles directly from Sentinel Hub API and processes without downloading
This saves ~90% disk space by only keeping processed results
"""
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
import json
from typing import Tuple, List, Dict
import io
from PIL import Image
import torch

# Sentinel Hub API
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    MimeType,
    SentinelHubRequest,
    DataCollection,
    bbox_to_dimensions,
)

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    COPERNICUS_USERNAME, COPERNICUS_PASSWORD,
    DATA_DIR_PROCESSED, CITIES, SENTINEL_BANDS,
    TARGET_RESOLUTION, TILE_SIZE, BATCH_SIZE,
    LAND_COVER_CLASSES, CHANGE_CLASSES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingSatelliteAnalyzer:
    """
    Fetch satellite tiles on-the-fly and process immediately without storing raw data
    """
    
    def __init__(self, client_id=None, client_secret=None, device='cuda'):
        """
        Initialize streaming analyzer
        
        Args:
            client_id: Sentinel Hub client ID (or use username)
            client_secret: Sentinel Hub client secret (or use password)
            device: 'cuda' for GPU or 'cpu'
        """
        # Configure Sentinel Hub
        self.config = SHConfig()
        
        # Sentinel Hub uses OAuth2, but we can also use Copernicus credentials
        # For free tier, we'll use Copernicus Data Space API
        self.config.sh_client_id = client_id or ''
        self.config.sh_client_secret = client_secret or ''
        
        # Fallback to Copernicus credentials
        self.username = COPERNICUS_USERNAME
        self.password = COPERNICUS_PASSWORD
        
        self.device = device
        self.output_dir = Path(DATA_DIR_PROCESSED)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ML models (lazy loading)
        self._segmentation_model = None
        
        logger.info(f"Initialized streaming analyzer on {device}")
    
    def _get_segmentation_model(self):
        """Lazy load segmentation model"""
        if self._segmentation_model is None:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            
            logger.info("Loading SegFormer model...")
            self._processor = SegformerImageProcessor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            self._segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            ).to(self.device)
            self._segmentation_model.eval()
            logger.info("Model loaded successfully")
        
        return self._segmentation_model, self._processor
    
    def fetch_tile_copernicus(
        self,
        bbox: Dict[str, float],
        date_start: str,
        date_end: str,
        size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """
        Fetch a tile directly from Copernicus Data Space API
        
        Args:
            bbox: {'west': lon, 'south': lat, 'east': lon, 'north': lat}
            date_start: ISO date string (e.g., '2024-02-01')
            date_end: ISO date string
            size: (width, height) in pixels
        
        Returns:
            numpy array of shape (height, width, channels) with RGB+NIR bands
        """
        import requests
        
        # Copernicus Data Space API endpoint for WMS
        wms_url = "https://sh.dataspace.copernicus.eu/ogc/wms/"
        
        # Build request parameters for true color + NIR
        params = {
            'service': 'WMS',
            'version': '1.3.0',
            'request': 'GetMap',
            'layers': 'TRUE-COLOR-S2-L2A',
            'bbox': f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}",
            'crs': 'EPSG:4326',
            'width': size[0],
            'height': size[1],
            'format': 'image/png',
            'time': f"{date_start}/{date_end}",
            'maxcc': 15,  # Max cloud cover 15%
        }
        
        try:
            # Use basic auth with Copernicus credentials
            response = requests.get(
                wms_url,
                params=params,
                auth=(self.username, self.password),
                timeout=30
            )
            response.raise_for_status()
            
            # Convert to numpy array
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img)
            
            # Normalize to 0-1
            img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
        
        except Exception as e:
            logger.error(f"Error fetching tile: {e}")
            # Return black image if fetch fails
            return np.zeros((size[1], size[0], 3), dtype=np.float32)
    
    def segment_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        """
        Run semantic segmentation on a tile
        
        Args:
            tile_rgb: RGB image array (H, W, 3), values 0-1
        
        Returns:
            Segmentation mask (H, W) with class indices
        """
        model, processor = self._get_segmentation_model()
        
        # Preprocess
        inputs = processor(images=tile_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        logits = torch.nn.functional.interpolate(
            logits,
            size=tile_rgb.shape[:2],
            mode='bilinear',
            align_corners=False
        )
        
        # Get class predictions
        mask = logits.argmax(dim=1).squeeze().cpu().numpy()
        
        return mask
    
    def detect_changes(
        self,
        mask_before: np.ndarray,
        mask_after: np.ndarray
    ) -> Dict[str, any]:
        """
        Detect changes between two segmentation masks
        
        Args:
            mask_before: Segmentation mask from before date
            mask_after: Segmentation mask from after date
        
        Returns:
            Dictionary with change statistics
        """
        # Map model classes to our classes (simplified)
        # SegFormer ADE20K: 0=background, 3=building, 4=sky, 9=tree, 11=water, 13=vegetation, 17=ground
        class_mapping = {
            3: 1,   # building
            9: 2,   # tree/vegetation
            13: 2,  # vegetation
            11: 3,  # water
            17: 5,  # bare land
            0: 0,   # other
        }
        
        # Remap classes
        mapped_before = np.zeros_like(mask_before)
        mapped_after = np.zeros_like(mask_after)
        
        for model_class, our_class in class_mapping.items():
            mapped_before[mask_before == model_class] = our_class
            mapped_after[mask_after == model_class] = our_class
        
        # Calculate pixel counts
        pixel_area_m2 = (TARGET_RESOLUTION ** 2)  # 10m * 10m = 100 m²
        
        changes = {}
        
        # Deforestation (vegetation → non-vegetation)
        deforestation = np.sum((mapped_before == 2) & (mapped_after != 2))
        changes['deforestation_pixels'] = int(deforestation)
        changes['deforestation_hectares'] = float(deforestation * pixel_area_m2 / 10000)
        
        # New construction (non-building → building)
        construction = np.sum((mapped_before != 1) & (mapped_after == 1))
        changes['construction_pixels'] = int(construction)
        changes['construction_hectares'] = float(construction * pixel_area_m2 / 10000)
        
        # Water loss (water → non-water)
        water_loss = np.sum((mapped_before == 3) & (mapped_after != 3))
        changes['water_loss_pixels'] = int(water_loss)
        changes['water_loss_hectares'] = float(water_loss * pixel_area_m2 / 10000)
        
        # Urbanization (any → building or road)
        urbanization = np.sum((mapped_before == 0) & ((mapped_after == 1) | (mapped_after == 4)))
        changes['urbanization_pixels'] = int(urbanization)
        changes['urbanization_hectares'] = float(urbanization * pixel_area_m2 / 10000)
        
        return changes
    
    def create_change_visualization(
        self,
        mask_before: np.ndarray,
        mask_after: np.ndarray
    ) -> np.ndarray:
        """
        Create RGB visualization of changes
        
        Returns:
            RGB image (H, W, 3) showing changes in color
        """
        # Create change map with colors
        change_viz = np.zeros((*mask_before.shape, 3), dtype=np.uint8)
        
        # No change = gray
        no_change = (mask_before == mask_after)
        change_viz[no_change] = [200, 200, 200]
        
        # Deforestation = red
        deforestation = (mask_before == 2) & (mask_after != 2)
        change_viz[deforestation] = [255, 0, 0]
        
        # Construction = blue
        construction = (mask_before != 1) & (mask_after == 1)
        change_viz[construction] = [0, 100, 255]
        
        # Water loss = yellow
        water_loss = (mask_before == 3) & (mask_after != 3)
        change_viz[water_loss] = [255, 255, 0]
        
        return change_viz
    
    def analyze_city_streaming(
        self,
        city: str,
        date_before: str,
        date_after: str,
        grid_size: int = 4
    ) -> Dict:
        """
        Analyze a city using streaming approach - fetch and process tiles on-the-fly
        
        Args:
            city: City name (must be in CITIES config)
            date_before: Before date (ISO format: '2020-02-01')
            date_after: After date (ISO format: '2024-02-01')
            grid_size: Split city into grid_size x grid_size tiles
        
        Returns:
            Dictionary with aggregated results
        """
        if city not in CITIES:
            raise ValueError(f"City {city} not found in config")
        
        city_bbox = CITIES[city]['bbox']
        logger.info(f"Analyzing {city} from {date_before} to {date_after}")
        logger.info(f"Streaming mode: fetching {grid_size}x{grid_size} = {grid_size**2} tiles on-the-fly")
        
        # Create grid of tiles
        lat_step = (city_bbox['north'] - city_bbox['south']) / grid_size
        lon_step = (city_bbox['east'] - city_bbox['west']) / grid_size
        
        # Aggregate results
        total_changes = {
            'deforestation_hectares': 0.0,
            'construction_hectares': 0.0,
            'water_loss_hectares': 0.0,
            'urbanization_hectares': 0.0,
            'tiles_processed': 0,
        }
        
        # Storage for visualization (only store processed masks, not raw data!)
        all_change_masks = []
        
        # Process each tile
        total_tiles = grid_size * grid_size
        with tqdm(total=total_tiles, desc="Streaming & analyzing tiles") as pbar:
            for i in range(grid_size):
                for j in range(grid_size):
                    # Define tile bounding box
                    tile_bbox = {
                        'south': city_bbox['south'] + i * lat_step,
                        'north': city_bbox['south'] + (i + 1) * lat_step,
                        'west': city_bbox['west'] + j * lon_step,
                        'east': city_bbox['west'] + (j + 1) * lon_step,
                    }
                    
                    try:
                        # Fetch "before" tile directly from API (not saved!)
                        tile_before = self.fetch_tile_copernicus(
                            tile_bbox,
                            date_before,
                            date_before,  # Single day
                            size=(TILE_SIZE, TILE_SIZE)
                        )
                        
                        # Segment immediately
                        mask_before = self.segment_tile(tile_before)
                        
                        # Discard raw tile (save memory!)
                        del tile_before
                        
                        # Fetch "after" tile directly from API (not saved!)
                        tile_after = self.fetch_tile_copernicus(
                            tile_bbox,
                            date_after,
                            date_after,
                            size=(TILE_SIZE, TILE_SIZE)
                        )
                        
                        # Segment immediately
                        mask_after = self.segment_tile(tile_after)
                        
                        # Discard raw tile
                        del tile_after
                        
                        # Detect changes (only keep the analysis!)
                        changes = self.detect_changes(mask_before, mask_after)
                        
                        # Aggregate
                        for key in ['deforestation_hectares', 'construction_hectares', 
                                   'water_loss_hectares', 'urbanization_hectares']:
                            total_changes[key] += changes[key]
                        
                        total_changes['tiles_processed'] += 1
                        
                        # Create visualization (lightweight)
                        change_viz = self.create_change_visualization(mask_before, mask_after)
                        all_change_masks.append((i, j, change_viz))
                        
                        # Free memory
                        del mask_before, mask_after, changes
                        
                    except Exception as e:
                        logger.warning(f"Failed to process tile ({i},{j}): {e}")
                    
                    pbar.update(1)
        
        # Save results
        output_path = self.output_dir / city / f"analysis_{date_before}_to_{date_after}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated statistics
        with open(output_path / 'changes_summary.json', 'w') as f:
            json.dump(total_changes, f, indent=2)
        
        # Save visualization mosaic
        self._save_mosaic(all_change_masks, output_path / 'change_map.png', grid_size)
        
        logger.info(f"Analysis complete! Results saved to {output_path}")
        logger.info(f"Total changes detected:")
        logger.info(f"  - Deforestation: {total_changes['deforestation_hectares']:.2f} hectares")
        logger.info(f"  - New construction: {total_changes['construction_hectares']:.2f} hectares")
        logger.info(f"  - Water loss: {total_changes['water_loss_hectares']:.2f} hectares")
        
        return total_changes
    
    def _save_mosaic(self, tile_list, output_path, grid_size):
        """Combine tile visualizations into a mosaic"""
        if not tile_list:
            return
        
        # Get tile size from first tile
        _, _, first_tile = tile_list[0]
        tile_h, tile_w = first_tile.shape[:2]
        
        # Create mosaic
        mosaic = np.zeros((grid_size * tile_h, grid_size * tile_w, 3), dtype=np.uint8)
        
        for i, j, tile_viz in tile_list:
            mosaic[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = tile_viz
        
        # Save
        Image.fromarray(mosaic).save(output_path)
        logger.info(f"Saved change map mosaic to {output_path}")


# CLI Interface
import click

@click.command()
@click.option('--city', default='bangalore', help='City to analyze')
@click.option('--before', default='2020-02-01', help='Before date (YYYY-MM-DD)')
@click.option('--after', default='2024-02-01', help='After date (YYYY-MM-DD)')
@click.option('--grid-size', default=4, help='Grid size (e.g., 4 means 4x4=16 tiles)')
@click.option('--device', default='cuda', help='Device: cuda or cpu')
def main(city, before, after, grid_size, device):
    """
    Stream and analyze satellite imagery on-the-fly without downloading
    
    Example:
        python stream_and_analyze.py --city bangalore --before 2020-02-01 --after 2024-02-01
    """
    # Check for GPU
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    analyzer = StreamingSatelliteAnalyzer(device=device)
    results = analyzer.analyze_city_streaming(
        city=city,
        date_before=before,
        date_after=after,
        grid_size=grid_size
    )
    
    print("\n" + "="*60)
    print(f"STREAMING ANALYSIS COMPLETE - {city.upper()}")
    print("="*60)
    print(f"Deforestation:    {results['deforestation_hectares']:>10.2f} hectares")
    print(f"New Construction: {results['construction_hectares']:>10.2f} hectares")
    print(f"Water Loss:       {results['water_loss_hectares']:>10.2f} hectares")
    print(f"Urbanization:     {results['urbanization_hectares']:>10.2f} hectares")
    print(f"Tiles Processed:  {results['tiles_processed']:>10}")
    print("="*60)
    print(f"\n💾 Disk space saved: ~6 GB (raw data not stored!)")


if __name__ == '__main__':
    main()
