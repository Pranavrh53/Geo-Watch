"""
Preprocessing Pipeline for Sentinel-2 Imagery
Loads, resamples, normalizes, and tiles satellite imagery
"""
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import click
import logging
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR_RAW, DATA_DIR_PROCESSED,
    SENTINEL_BANDS, BAND_RESOLUTIONS,
    TARGET_RESOLUTION, TILE_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentinelPreprocessor:
    """Preprocess Sentinel-2 imagery for ML models"""
    
    def __init__(self, input_dir, output_dir, tile_size=TILE_SIZE):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_band(self, band_name):
        """Load a single band from JP2 or TIF file"""
        # Try different file extensions
        for ext in ['.jp2', '.tif', '.tiff', '.JP2', '.TIF']:
            band_file = self.input_dir / f"{band_name}{ext}"
            if band_file.exists():
                logger.info(f"Loading {band_name} from {band_file}")
                with rasterio.open(band_file) as src:
                    data = src.read(1)
                    profile = src.profile
                    transform = src.transform
                    crs = src.crs
                return data, profile, transform, crs
        
        raise FileNotFoundError(f"Band {band_name} not found in {self.input_dir}")
    
    def resample_band(self, data, src_resolution, target_resolution, src_transform, src_crs):
        """Resample band to target resolution"""
        if src_resolution == target_resolution:
            return data, src_transform
        
        logger.info(f"Resampling from {src_resolution}m to {target_resolution}m")
        
        scale_factor = src_resolution / target_resolution
        new_height = int(data.shape[0] * scale_factor)
        new_width = int(data.shape[1] * scale_factor)
        
        # Calculate new transform
        dst_transform = src_transform * src_transform.scale(
            data.shape[1] / new_width,
            data.shape[0] / new_height
        )
        
        # Create destination array
        resampled = np.zeros((new_height, new_width), dtype=data.dtype)
        
        # Resample using bilinear interpolation
        reproject(
            source=data,
            destination=resampled,
            src_transform=src_transform,
            dst_transform=dst_transform,
            src_crs=src_crs,
            dst_crs=src_crs,
            resampling=Resampling.bilinear
        )
        
        return resampled, dst_transform
    
    def normalize_band(self, data, percentile_clip=2):
        """
        Normalize band values to 0-1 range with percentile clipping
        
        Args:
            data: Band data
            percentile_clip: Percentile for contrast stretching
        """
        # Clip outliers using percentiles
        p_low = np.percentile(data, percentile_clip)
        p_high = np.percentile(data, 100 - percentile_clip)
        
        # Clip and normalize
        clipped = np.clip(data, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
        
        return normalized.astype(np.float32)
    
    def create_composite(self, bands_dict):
        """
        Create multi-band composite
        
        Args:
            bands_dict: Dictionary mapping band names to data arrays
        
        Returns:
            Stacked array (bands, height, width)
        """
        # Get first band to determine dimensions
        first_band = list(bands_dict.values())[0]
        height, width = first_band.shape
        
        # Stack bands
        composite = np.zeros((len(bands_dict), height, width), dtype=np.float32)
        
        for i, (band_name, data) in enumerate(bands_dict.items()):
            logger.info(f"Adding {band_name} to composite")
            composite[i] = self.normalize_band(data)
        
        return composite
    
    def tile_image(self, image, overlap=0):
        """
        Split image into tiles
        
        Args:
            image: Input array (bands, height, width)
            overlap: Overlap between tiles in pixels
        
        Returns:
            List of tiles and their positions
        """
        bands, height, width = image.shape
        tiles = []
        positions = []
        
        stride = self.tile_size - overlap
        
        for i in range(0, height - self.tile_size + 1, stride):
            for j in range(0, width - self.tile_size + 1, stride):
                tile = image[:, i:i+self.tile_size, j:j+self.tile_size]
                
                # Only keep tiles that are full size
                if tile.shape[1] == self.tile_size and tile.shape[2] == self.tile_size:
                    tiles.append(tile)
                    positions.append((i, j))
        
        logger.info(f"Created {len(tiles)} tiles of size {self.tile_size}x{self.tile_size}")
        return tiles, positions
    
    def save_tiles(self, tiles, positions, prefix='tile'):
        """Save tiles to disk"""
        tiles_dir = self.output_dir / 'tiles'
        tiles_dir.mkdir(exist_ok=True)
        
        tile_info = []
        
        for idx, (tile, pos) in enumerate(tqdm(zip(tiles, positions), desc="Saving tiles")):
            tile_file = tiles_dir / f"{prefix}_{pos[0]}_{pos[1]}.npy"
            np.save(tile_file, tile)
            
            tile_info.append({
                'file': str(tile_file),
                'position': pos,
                'shape': tile.shape
            })
        
        # Save tile index
        index_file = self.output_dir / f'{prefix}_index.json'
        with open(index_file, 'w') as f:
            json.dump(tile_info, f, indent=2)
        
        logger.info(f"Saved {len(tiles)} tiles to {tiles_dir}")
        return tile_info
    
    def process(self, save_composite=True):
        """
        Full preprocessing pipeline
        
        Returns:
            Composite image and tiles
        """
        logger.info(f"Processing data from {self.input_dir}")
        
        # Load and process all bands
        bands_data = {}
        reference_transform = None
        reference_crs = None
        
        for band_name in SENTINEL_BANDS:
            try:
                # Load band
                data, profile, transform, crs = self.load_band(band_name)
                
                # Store reference CRS and transform from first band
                if reference_transform is None:
                    reference_transform = transform
                    reference_crs = crs
                
                # Resample if needed
                band_resolution = BAND_RESOLUTIONS.get(band_name, TARGET_RESOLUTION)
                if band_resolution != TARGET_RESOLUTION:
                    data, _ = self.resample_band(
                        data, band_resolution, TARGET_RESOLUTION,
                        transform, crs
                    )
                
                bands_data[band_name] = data
                
            except FileNotFoundError as e:
                logger.warning(f"Skipping {band_name}: {e}")
        
        if not bands_data:
            raise ValueError(f"No bands found in {self.input_dir}")
        
        # Create composite
        logger.info("Creating multi-band composite")
        composite = self.create_composite(bands_data)
        
        # Save full composite
        if save_composite:
            composite_file = self.output_dir / 'composite.npy'
            np.save(composite_file, composite)
            logger.info(f"Saved composite to {composite_file}")
            
            # Save metadata
            metadata = {
                'bands': list(bands_data.keys()),
                'shape': composite.shape,
                'resolution': TARGET_RESOLUTION,
                'tile_size': self.tile_size
            }
            metadata_file = self.output_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create tiles
        logger.info("Creating tiles")
        tiles, positions = self.tile_image(composite)
        
        # Save tiles
        prefix = self.output_dir.name
        tile_info = self.save_tiles(tiles, positions, prefix=prefix)
        
        logger.info(f"✅ Preprocessing complete!")
        logger.info(f"   Composite shape: {composite.shape}")
        logger.info(f"   Number of tiles: {len(tiles)}")
        logger.info(f"   Output directory: {self.output_dir}")
        
        return composite, tiles, tile_info


@click.command()
@click.option('--city', required=True, help='City name')
@click.option('--before-date', help='Before date (YYYY-MM-DD)')
@click.option('--after-date', help='After date (YYYY-MM-DD)')
@click.option('--tile-size', default=TILE_SIZE, help='Tile size in pixels')
def main(city, before_date, after_date, tile_size):
    """
    Preprocess Sentinel-2 data for a city
    
    Example:
        python preprocess.py --city bangalore --before-date 2020-02-01 --after-date 2024-02-01
    """
    
    city_dir = DATA_DIR_RAW / city.lower()
    
    if not city_dir.exists():
        logger.error(f"City directory not found: {city_dir}")
        logger.info(f"Please download data first using download_sentinel2.py")
        return
    
    # Find before and after directories
    before_dirs = list(city_dir.glob('before_*'))
    after_dirs = list(city_dir.glob('after_*'))
    
    if before_date:
        before_dirs = [d for d in before_dirs if before_date in d.name]
    if after_date:
        after_dirs = [d for d in after_dirs if after_date in d.name]
    
    if not before_dirs:
        logger.error(f"No 'before' data found in {city_dir}")
        return
    if not after_dirs:
        logger.error(f"No 'after' data found in {city_dir}")
        return
    
    before_dir = before_dirs[0]
    after_dir = after_dirs[0]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PREPROCESSING: {city.upper()}")
    logger.info(f"{'='*60}")
    
    # Process before imagery
    logger.info(f"\n📅 Processing BEFORE imagery: {before_dir.name}")
    before_output = DATA_DIR_PROCESSED / city.lower() / before_dir.name
    before_processor = SentinelPreprocessor(before_dir, before_output, tile_size)
    try:
        before_processor.process()
    except Exception as e:
        logger.error(f"Error processing before imagery: {e}")
        return
    
    # Process after imagery
    logger.info(f"\n📅 Processing AFTER imagery: {after_dir.name}")
    after_output = DATA_DIR_PROCESSED / city.lower() / after_dir.name
    after_processor = SentinelPreprocessor(after_dir, after_output, tile_size)
    try:
        after_processor.process()
    except Exception as e:
        logger.error(f"Error processing after imagery: {e}")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ PREPROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Before tiles: {before_output / 'tiles'}")
    logger.info(f"After tiles: {after_output / 'tiles'}")
    logger.info(f"\nNext step: Run segmentation")
    logger.info(f"  python scripts/run_segmentation.py --city {city}")


if __name__ == '__main__':
    main()
