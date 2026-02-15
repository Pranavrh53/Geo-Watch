"""
Visualization utilities for satellite imagery
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import click
import logging

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR_RAW, DATA_DIR_PROCESSED,
    SENTINEL_BANDS, LAND_COVER_CLASSES,
    CHANGE_CLASSES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_npy_file(file_path):
    """Load numpy array from file"""
    return np.load(file_path)


def create_rgb_composite(tile, bands=[0, 1, 2]):
    """
    Create RGB composite from multi-band tile
    
    Args:
        tile: Array (bands, height, width)
        bands: Which bands to use for RGB
    
    Returns:
        RGB image (height, width, 3)
    """
    if len(tile.shape) == 2:
        # Single band - convert to RGB
        rgb = np.stack([tile] * 3, axis=-1)
    else:
        # Multi-band - select specified bands
        rgb = np.transpose(tile[bands], (1, 2, 0))
    
    # Normalize to 0-255
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    return rgb


def create_false_color_composite(tile, r=3, g=2, b=1):
    """
    Create false color composite (NIR-Red-Green)
    Good for vegetation analysis
    
    Args:
        tile: Array (bands, height, width)
        r, g, b: Band indices for Red, Green, Blue channels
    
    Returns:
        False color RGB image
    """
    if tile.shape[0] <= max(r, g, b):
        logger.warning("Not enough bands for false color composite")
        return create_rgb_composite(tile)
    
    false_color = np.stack([tile[r], tile[g], tile[b]], axis=-1)
    false_color = np.transpose(false_color, (1, 2, 0))
    
    # Normalize
    false_color = np.clip(false_color * 255, 0, 255).astype(np.uint8)
    
    return false_color


def visualize_segmentation_mask(mask, save_path=None):
    """
    Visualize segmentation mask with colors
    
    Args:
        mask: Segmentation mask array (height, width)
        save_path: Optional path to save visualization
    """
    # Define colors for each class
    colors = {
        0: [0, 0, 0],           # Background - Black
        1: [255, 100, 100],     # Urban - Light Red
        2: [0, 200, 0],         # Vegetation - Green
        3: [100, 150, 255],     # Water - Light Blue
        4: [200, 180, 100],     # Soil - Tan
        5: [150, 150, 150],     # Road - Gray
    }
    
    # Create RGB image
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        rgb[mask == class_id] = color
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.axis('off')
    ax.set_title('Land Cover Segmentation')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(color)/255, label=LAND_COVER_CLASSES.get(class_id, f'Class {class_id}'))
        for class_id, color in colors.items()
        if class_id in LAND_COVER_CLASSES
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_changes(before_mask, after_mask, save_path=None):
    """
    Visualize changes between two segmentation masks
    
    Args:
        before_mask: Before segmentation mask
        after_mask: After segmentation mask
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Before
    visualize_segmentation_mask(before_mask)
    axes[0].imshow(create_colored_mask(before_mask))
    axes[0].set_title('Before')
    axes[0].axis('off')
    
    # After
    axes[1].imshow(create_colored_mask(after_mask))
    axes[1].set_title('After')
    axes[1].axis('off')
    
    # Changes
    change_map = np.zeros((before_mask.shape[0], before_mask.shape[1], 3), dtype=np.uint8)
    
    # Apply change colors
    for change_type, config in CHANGE_CLASSES.items():
        from_classes = config['from']
        to_classes = config['to']
        
        before_match = np.isin(before_mask, from_classes)
        after_match = np.isin(after_mask, to_classes)
        change_mask = before_match & after_match
        
        if np.any(change_mask):
            change_map[change_mask] = config['color']
    
    axes[2].imshow(change_map)
    axes[2].set_title('Changes')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_colored_mask(mask):
    """Create colored RGB image from mask"""
    colors = {
        0: [0, 0, 0],
        1: [255, 100, 100],
        2: [0, 200, 0],
        3: [100, 150, 255],
        4: [200, 180, 100],
        5: [150, 150, 150],
    }
    
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in colors.items():
        rgb[mask == class_id] = color
    
    return rgb


@click.command()
@click.option('--input', required=True, help='Path to input file (tile or mask)')
@click.option('--type', type=click.Choice(['tile', 'mask', 'changes']), default='tile')
@click.option('--before', help='Path to before mask (for changes visualization)')
@click.option('--after', help='Path to after mask (for changes visualization)')
@click.option('--output', help='Path to save output image')
def main(input, type, before, after, output):
    """
    Visualize satellite data, segmentation masks, or changes
    
    Examples:
        # Visualize a tile
        python visualize.py --input data/processed/bangalore/before_2020-02-01/tiles/tile_0_0.npy --type tile
        
        # Visualize a segmentation mask
        python visualize.py --input data/processed/bangalore/before_2020-02-01/masks/tile_0_0_mask.npy --type mask
        
        # Visualize changes
        python visualize.py --type changes --before before_mask.npy --after after_mask.npy
    """
    
    if type == 'changes':
        if not before or not after:
            logger.error("For changes visualization, provide --before and --after paths")
            return
        
        before_mask = load_npy_file(before)
        after_mask = load_npy_file(after)
        visualize_changes(before_mask, after_mask, output)
    
    elif type == 'mask':
        mask = load_npy_file(input)
        visualize_segmentation_mask(mask, output)
    
    elif type == 'tile':
        tile = load_npy_file(input)
        rgb = create_rgb_composite(tile)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title('Satellite Tile')
        
        if output:
            plt.savefig(output, dpi=150, bbox_inches='tight')
            logger.info(f"Saved to {output}")
        else:
            plt.show()
        
        plt.close()


if __name__ == '__main__':
    main()
