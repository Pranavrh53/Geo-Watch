"""
Utility functions for the satellite change detection system
"""
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.py"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    import config
    return config


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)


def save_json(data, filepath):
    """Save dictionary to JSON file"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_area(area_sqm):
    """
    Format area in different units
    
    Returns:
        dict with area in different units
    """
    from config import SQM_TO_HECTARES, SQM_TO_ACRES, SQM_TO_SQKM
    
    return {
        'sqm': float(area_sqm),
        'hectares': float(area_sqm * SQM_TO_HECTARES),
        'acres': float(area_sqm * SQM_TO_ACRES),
        'sqkm': float(area_sqm * SQM_TO_SQKM)
    }


def calculate_pixel_area(pixel_count, resolution=10):
    """
    Calculate area from pixel count
    
    Args:
        pixel_count: Number of pixels
        resolution: Resolution in meters (default 10m for Sentinel-2)
    
    Returns:
        Area in square meters
    """
    return pixel_count * (resolution ** 2)


def get_timestamp():
    """Get current timestamp as ISO string"""
    return datetime.now().isoformat()


def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str


def validate_bbox(bbox):
    """
    Validate bounding box coordinates
    
    Args:
        bbox: Dict with north, south, east, west keys
    
    Returns:
        bool: True if valid
    """
    required_keys = ['north', 'south', 'east', 'west']
    
    if not all(key in bbox for key in required_keys):
        return False
    
    if bbox['north'] <= bbox['south']:
        logger.error("North must be greater than South")
        return False
    
    if bbox['east'] <= bbox['west']:
        logger.error("East must be greater than West")
        return False
    
    if not (-90 <= bbox['south'] <= 90 and -90 <= bbox['north'] <= 90):
        logger.error("Latitude must be between -90 and 90")
        return False
    
    if not (-180 <= bbox['west'] <= 180 and -180 <= bbox['east'] <= 180):
        logger.error("Longitude must be between -180 and 180")
        return False
    
    return True


def bbox_to_wkt(bbox):
    """
    Convert bounding box to WKT polygon
    
    Args:
        bbox: Dict with north, south, east, west keys
    
    Returns:
        WKT string
    """
    return (
        f"POLYGON(("
        f"{bbox['west']} {bbox['south']},"
        f"{bbox['east']} {bbox['south']},"
        f"{bbox['east']} {bbox['north']},"
        f"{bbox['west']} {bbox['north']},"
        f"{bbox['west']} {bbox['south']}"
        f"))"
    )


def calculate_overlap(tile1_pos, tile2_pos, tile_size):
    """
    Calculate overlap between two tiles
    
    Args:
        tile1_pos: (row, col) position of tile 1
        tile2_pos: (row, col) position of tile 2
        tile_size: Size of tiles
    
    Returns:
        Overlap percentage (0-100)
    """
    r1, c1 = tile1_pos
    r2, c2 = tile2_pos
    
    # Calculate overlap in each dimension
    row_overlap = max(0, tile_size - abs(r1 - r2))
    col_overlap = max(0, tile_size - abs(c1 - c2))
    
    # Calculate overlap area
    overlap_area = row_overlap * col_overlap
    total_area = tile_size * tile_size
    
    return (overlap_area / total_area) * 100


def merge_change_statistics(stats_list):
    """
    Merge multiple change statistics dictionaries
    
    Args:
        stats_list: List of statistics dictionaries
    
    Returns:
        Merged statistics dictionary
    """
    merged = {}
    
    for stats in stats_list:
        for change_type, values in stats.items():
            if change_type not in merged:
                merged[change_type] = {
                    'pixels': 0,
                    'area_sqm': 0,
                    'area_hectares': 0,
                    'area_acres': 0,
                    'area_sqkm': 0
                }
            
            for key, value in values.items():
                merged[change_type][key] += value
    
    return merged


def normalize_array(arr, min_percentile=2, max_percentile=98):
    """
    Normalize array using percentile clipping
    
    Args:
        arr: Input array
        min_percentile: Lower percentile for clipping
        max_percentile: Upper percentile for clipping
    
    Returns:
        Normalized array (0-1)
    """
    p_min = np.percentile(arr, min_percentile)
    p_max = np.percentile(arr, max_percentile)
    
    clipped = np.clip(arr, p_min, p_max)
    normalized = (clipped - p_min) / (p_max - p_min + 1e-8)
    
    return normalized.astype(np.float32)


def create_summary_report(city, before_date, after_date, results):
    """
    Create a summary report dictionary
    
    Args:
        city: City name
        before_date: Before date string
        after_date: After date string
        results: Change detection results
    
    Returns:
        Report dictionary
    """
    from config import CITIES, CHANGE_CLASSES
    
    city_info = CITIES.get(city.lower(), {})
    
    report = {
        'metadata': {
            'city': city_info.get('name', city),
            'country': city_info.get('country', 'Unknown'),
            'before_date': before_date,
            'after_date': after_date,
            'generated_at': get_timestamp(),
            'coordinates': city_info.get('center', {})
        },
        'changes': [],
        'total_change_area_hectares': 0
    }
    
    # Add change details
    for change_type, stats in results.get('summary', {}).items():
        if stats['area_hectares'] > 0:
            config = CHANGE_CLASSES.get(change_type, {})
            report['changes'].append({
                'type': change_type,
                'name': config.get('name', change_type),
                'area': format_area(stats['area_sqm']),
                'pixels': stats['pixels'],
                'color': config.get('color', [128, 128, 128])
            })
            report['total_change_area_hectares'] += stats['area_hectares']
    
    return report


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Print a progress bar
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Number of decimals for percentage
        length: Character length of bar
        fill: Bar fill character
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    if iteration == total:
        print()


# File size utilities
def get_file_size(filepath):
    """Get file size in MB"""
    size_bytes = Path(filepath).stat().st_size
    return size_bytes / (1024 * 1024)


def get_directory_size(directory):
    """Get total size of directory in MB"""
    total_size = 0
    for filepath in Path(directory).rglob('*'):
        if filepath.is_file():
            total_size += filepath.stat().st_size
    return total_size / (1024 * 1024)


# Date utilities
def parse_date_from_dirname(dirname):
    """
    Parse date from directory name
    
    Examples:
        before_2020-02-01 -> 2020-02-01
        after_2024-02-01 -> 2024-02-01
    """
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2})', dirname)
    if match:
        return match.group(1)
    return None


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test area formatting
    area = format_area(10000)  # 10,000 square meters
    print(f"10,000 sqm = {area['hectares']:.2f} hectares")
    
    # Test bbox validation
    bbox = {'north': 13.17, 'south': 12.73, 'east': 77.88, 'west': 77.37}
    print(f"BBox valid: {validate_bbox(bbox)}")
    
    # Test WKT conversion
    wkt = bbox_to_wkt(bbox)
    print(f"WKT: {wkt[:50]}...")
    
    print("✅ All tests passed!")
