"""
Change Detection Engine
Compares before/after segmentation masks to detect land cover changes
"""
import sys
from pathlib import Path
import numpy as np
import click
import logging
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR_PROCESSED, DATA_DIR_RESULTS,
    CHANGE_CLASSES, PIXEL_AREA,
    SQM_TO_HECTARES, SQM_TO_ACRES, SQM_TO_SQKM
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChangeDetector:
    """Detect land cover changes between two time periods"""
    
    def __init__(self, before_dir, after_dir, output_dir):
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.change_stats = {
            change_type: {
                'pixels': 0,
                'area_sqm': 0,
                'area_hectares': 0,
                'area_acres': 0,
                'area_sqkm': 0
            }
            for change_type in CHANGE_CLASSES.keys()
        }
    
    def detect_change_in_tile(self, before_mask, after_mask):
        """
        Detect changes between two masks
        
        Args:
            before_mask: Segmentation mask from before image
            after_mask: Segmentation mask from after image
        
        Returns:
            Dictionary of change masks for each change type
        """
        changes = {}
        
        for change_type, config in CHANGE_CLASSES.items():
            from_classes = config['from']
            to_classes = config['to']
            
            # Create boolean masks
            before_match = np.isin(before_mask, from_classes)
            after_match = np.isin(after_mask, to_classes)
            
            # Change occurred where before matched and after matched
            change_mask = before_match & after_match
            
            changes[change_type] = change_mask
        
        return changes
    
    def create_visualization_mask(self, changes):
        """
        Create RGB visualization of all changes
        
        Args:
            changes: Dictionary of change masks
        
        Returns:
            RGB image (height, width, 3)
        """
        # Get dimensions from first change mask
        first_mask = next(iter(changes.values()))
        height, width = first_mask.shape
        
        # Create RGB visualization
        vis_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply colors for each change type (later changes will overlay earlier ones)
        for change_type, change_mask in changes.items():
            if change_type in CHANGE_CLASSES:
                color = CHANGE_CLASSES[change_type]['color']
                vis_mask[change_mask] = color
        
        return vis_mask
    
    def process_tile_pair(self, before_mask_file, after_mask_file):
        """
        Process a pair of before/after masks
        
        Returns:
            Changes dictionary and statistics
        """
        # Load masks
        before_mask = np.load(before_mask_file)
        after_mask = np.load(after_mask_file)
        
        # Ensure same shape
        if before_mask.shape != after_mask.shape:
            logger.warning(f"Shape mismatch: {before_mask.shape} vs {after_mask.shape}")
            return None, None
        
        # Detect changes
        changes = self.detect_change_in_tile(before_mask, after_mask)
        
        # Calculate statistics for this tile
        tile_stats = {}
        for change_type, change_mask in changes.items():
            num_pixels = np.sum(change_mask)
            tile_stats[change_type] = {
                'pixels': int(num_pixels),
                'area_sqm': float(num_pixels * PIXEL_AREA)
            }
        
        return changes, tile_stats
    
    def process_all_tiles(self):
        """Process all tile pairs and generate change maps"""
        
        # Find mask files
        before_masks = sorted((self.before_dir / 'masks').glob('*_mask.npy'))
        after_masks = sorted((self.after_dir / 'masks').glob('*_mask.npy'))
        
        if not before_masks or not after_masks:
            logger.error("No mask files found")
            return
        
        logger.info(f"Found {len(before_masks)} before masks and {len(after_masks)} after masks")
        
        # Match before and after tiles by position
        before_dict = {f.stem.replace('_mask', ''): f for f in before_masks}
        after_dict = {f.stem.replace('_mask', ''): f for f in after_masks}
        
        common_tiles = set(before_dict.keys()) & set(after_dict.keys())
        logger.info(f"Processing {len(common_tiles)} matching tile pairs")
        
        if not common_tiles:
            logger.warning("No matching tiles found!")
            return
        
        # Create output directories
        changes_dir = self.output_dir / 'change_masks'
        vis_dir = self.output_dir / 'visualizations'
        changes_dir.mkdir(exist_ok=True)
        vis_dir.mkdir(exist_ok=True)
        
        tile_results = []
        
        for tile_name in tqdm(sorted(common_tiles), desc="Detecting changes"):
            before_file = before_dict[tile_name]
            after_file = after_dict[tile_name]
            
            # Process tile pair
            changes, tile_stats = self.process_tile_pair(before_file, after_file)
            
            if changes is None:
                continue
            
            # Update global statistics
            for change_type, stats in tile_stats.items():
                self.change_stats[change_type]['pixels'] += stats['pixels']
                self.change_stats[change_type]['area_sqm'] += stats['area_sqm']
            
            # Save individual change masks
            for change_type, change_mask in changes.items():
                output_file = changes_dir / f"{tile_name}_{change_type}.npy"
                np.save(output_file, change_mask)
            
            # Create and save visualization
            vis_mask = self.create_visualization_mask(changes)
            vis_file = vis_dir / f"{tile_name}_changes.png"
            Image.fromarray(vis_mask).save(vis_file)
            
            tile_results.append({
                'tile': tile_name,
                'stats': tile_stats
            })
        
        # Calculate area conversions
        for change_type in self.change_stats:
            area_sqm = self.change_stats[change_type]['area_sqm']
            self.change_stats[change_type]['area_hectares'] = area_sqm * SQM_TO_HECTARES
            self.change_stats[change_type]['area_acres'] = area_sqm * SQM_TO_ACRES
            self.change_stats[change_type]['area_sqkm'] = area_sqm * SQM_TO_SQKM
        
        # Save results
        results = {
            'summary': self.change_stats,
            'tiles': tile_results
        }
        
        results_file = self.output_dir / 'change_detection_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to {results_file}")
        
        return results
    
    def create_summary_visualization(self):
        """Create summary visualization of changes"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart of areas
        change_types = []
        areas_hectares = []
        colors = []
        
        for change_type, config in CHANGE_CLASSES.items():
            if self.change_stats[change_type]['area_hectares'] > 0:
                change_types.append(config['name'])
                areas_hectares.append(self.change_stats[change_type]['area_hectares'])
                colors.append([c/255 for c in config['color']])
        
        if areas_hectares:
            ax1.barh(change_types, areas_hectares, color=colors)
            ax1.set_xlabel('Area (Hectares)')
            ax1.set_title('Land Cover Changes')
            ax1.grid(axis='x', alpha=0.3)
        
        # Summary table
        table_data = []
        for change_type, config in CHANGE_CLASSES.items():
            stats = self.change_stats[change_type]
            if stats['area_hectares'] > 0:
                table_data.append([
                    config['name'],
                    f"{stats['area_hectares']:.2f}",
                    f"{stats['area_acres']:.2f}",
                    f"{stats['area_sqkm']:.4f}"
                ])
        
        if table_data:
            ax2.axis('tight')
            ax2.axis('off')
            table = ax2.table(
                cellText=table_data,
                colLabels=['Change Type', 'Hectares', 'Acres', 'Sq Km'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        plt.tight_layout()
        
        # Save figure
        fig_file = self.output_dir / 'change_summary.png'
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary visualization saved to {fig_file}")
    
    def print_summary(self):
        """Print change detection summary"""
        
        logger.info(f"\n{'='*60}")
        logger.info("CHANGE DETECTION SUMMARY")
        logger.info(f"{'='*60}\n")
        
        total_change = 0
        
        for change_type, config in CHANGE_CLASSES.items():
            stats = self.change_stats[change_type]
            
            if stats['area_hectares'] > 0:
                logger.info(f"🔹 {config['name']}:")
                logger.info(f"   Area: {stats['area_hectares']:.2f} hectares")
                logger.info(f"         {stats['area_acres']:.2f} acres")
                logger.info(f"         {stats['area_sqkm']:.4f} sq km")
                logger.info(f"   Pixels: {stats['pixels']:,}")
                logger.info("")
                
                total_change += stats['area_hectares']
        
        logger.info(f"Total Changed Area: {total_change:.2f} hectares")
        logger.info(f"{'='*60}\n")


@click.command()
@click.option('--city', required=True, help='City name')
def main(city):
    """
    Detect changes between before and after imagery
    
    Example:
        python detect_changes.py --city bangalore
    """
    
    city_dir = DATA_DIR_PROCESSED / city.lower()
    
    if not city_dir.exists():
        logger.error(f"City directory not found: {city_dir}")
        return
    
    # Find before and after directories
    before_dirs = sorted(city_dir.glob('before_*'))
    after_dirs = sorted(city_dir.glob('after_*'))
    
    if not before_dirs or not after_dirs:
        logger.error(f"Before/after directories not found in {city_dir}")
        return
    
    before_dir = before_dirs[0]
    after_dir = after_dirs[0]
    
    # Check for masks
    if not (before_dir / 'masks').exists():
        logger.error(f"Before masks not found. Run segmentation first.")
        return
    if not (after_dir / 'masks').exists():
        logger.error(f"After masks not found. Run segmentation first.")
        return
    
    # Create output directory
    output_dir = DATA_DIR_RESULTS / city.lower() / f"{before_dir.name}_vs_{after_dir.name}"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CHANGE DETECTION: {city.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Before: {before_dir.name}")
    logger.info(f"After: {after_dir.name}")
    logger.info(f"Output: {output_dir}")
    
    # Initialize detector
    detector = ChangeDetector(before_dir, after_dir, output_dir)
    
    # Process all tiles
    results = detector.process_all_tiles()
    
    if results:
        # Print summary
        detector.print_summary()
        
        # Create visualization
        detector.create_summary_visualization()
        
        logger.info(f"\n✅ Change detection complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"\nNext step: Start backend API")
        logger.info(f"  python backend/main.py")


if __name__ == '__main__':
    main()
