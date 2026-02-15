"""
Sentinel-2 Data Download Script
Downloads satellite imagery from Copernicus Data Space for specified locations and dates
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests
from requests.auth import HTTPBasicAuth
import json
import click
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    COPERNICUS_USERNAME, COPERNICUS_PASSWORD,
    DATA_DIR_RAW, CITIES, SENTINEL_BANDS,
    MAX_CLOUD_COVER, SENTINEL_PRODUCT_TYPE
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentinelDownloader:
    """Download Sentinel-2 data from Copernicus Data Space"""
    
    def __init__(self, username=None, password=None):
        self.username = username or COPERNICUS_USERNAME
        self.password = password or COPERNICUS_PASSWORD
        self.base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
        self.auth = HTTPBasicAuth(self.username, self.password)
        
        if not self.username or not self.password:
            raise ValueError(
                "Copernicus credentials not found. "
                "Please set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env file"
            )
    
    def search_products(self, bbox, date_start, date_end, max_cloud_cover=15):
        """
        Search for Sentinel-2 products
        
        Args:
            bbox: dict with 'north', 'south', 'east', 'west' keys
            date_start: datetime object
            date_end: datetime object
            max_cloud_cover: maximum cloud cover percentage
        
        Returns:
            list of product IDs
        """
        logger.info(f"Searching for products from {date_start} to {date_end}")
        
        # Construct WKT polygon for bounding box
        wkt_geometry = (
            f"POLYGON(("
            f"{bbox['west']} {bbox['south']},"
            f"{bbox['east']} {bbox['south']},"
            f"{bbox['east']} {bbox['north']},"
            f"{bbox['west']} {bbox['north']},"
            f"{bbox['west']} {bbox['south']}"
            f"))"
        )
        
        # Build query
        query = (
            f"{self.base_url}/Products?"
            f"$filter=Collection/Name eq 'SENTINEL-2' "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_geometry}') "
            f"and ContentDate/Start gt {date_start.isoformat()}Z "
            f"and ContentDate/Start lt {date_end.isoformat()}Z "
            f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud_cover})"
            f"&$top=10"
        )
        
        try:
            response = requests.get(query, auth=self.auth)
            response.raise_for_status()
            data = response.json()
            
            products = data.get('value', [])
            logger.info(f"Found {len(products)} products")
            
            return products
        
        except Exception as e:
            logger.error(f"Error searching products: {e}")
            return []
    
    def download_product(self, product_id, output_dir, bands=None):
        """
        Download specific bands from a product
        
        Args:
            product_id: Product UUID
            output_dir: Directory to save downloaded data
            bands: List of band names (e.g., ['B02', 'B03', 'B04'])
        """
        bands = bands or SENTINEL_BANDS
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading product {product_id}")
        
        # Get product details
        product_url = f"{self.base_url}/Products({product_id})"
        response = requests.get(product_url, auth=self.auth)
        response.raise_for_status()
        product = response.json()
        
        # Save metadata
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(product, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_file}")
        
        # Download each band
        for band in bands:
            band_file = output_dir / f"{band}.jp2"
            
            if band_file.exists():
                logger.info(f"Band {band} already exists, skipping")
                continue
            
            # Note: Actual band download URL structure depends on Copernicus API
            # This is a simplified example
            logger.info(f"Downloading band {band}...")
            
            # For now, create placeholder (in real implementation, download actual data)
            # You'll need to use the correct API endpoint for band-specific downloads
            logger.warning(
                f"Band download not fully implemented. "
                f"Please download manually from Copernicus Browser and place in {output_dir}"
            )
        
        return output_dir
    
    def get_best_product(self, products):
        """
        Select the best product from search results
        (lowest cloud cover, most recent)
        """
        if not products:
            return None
        
        # Sort by cloud cover, then by date
        sorted_products = sorted(
            products,
            key=lambda p: (
                float(p.get('CloudCover', 100)),
                -datetime.fromisoformat(p['ContentDate']['Start'].replace('Z', '')).timestamp()
            )
        )
        
        best = sorted_products[0]
        logger.info(
            f"Selected product: {best['Name']} "
            f"(Cloud cover: {best.get('CloudCover', 'N/A')}%)"
        )
        
        return best


@click.command()
@click.option('--city', required=True, help='City name (bangalore, delhi, mumbai, hyderabad)')
@click.option('--before', required=True, help='Before date (YYYY-MM-DD)')
@click.option('--after', required=True, help='After date (YYYY-MM-DD)')
@click.option('--days-range', default=30, help='Date range in days to search around target date')
@click.option('--max-cloud', default=MAX_CLOUD_COVER, help='Maximum cloud cover percentage')
def main(city, before, after, days_range, max_cloud):
    """
    Download Sentinel-2 data for change detection
    
    Example:
        python download_sentinel2.py --city bangalore --before 2020-02-01 --after 2024-02-01
    """
    
    # Validate city
    if city.lower() not in CITIES:
        logger.error(f"Unknown city: {city}. Available: {list(CITIES.keys())}")
        return
    
    city_info = CITIES[city.lower()]
    logger.info(f"Downloading data for {city_info['name']}, {city_info['country']}")
    
    # Parse dates
    try:
        before_date = datetime.strptime(before, '%Y-%m-%d')
        after_date = datetime.strptime(after, '%Y-%m-%d')
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return
    
    # Initialize downloader
    try:
        downloader = SentinelDownloader()
    except ValueError as e:
        logger.error(str(e))
        logger.info(
            "\nTo get credentials:\n"
            "1. Go to https://dataspace.copernicus.eu\n"
            "2. Register for free account\n"
            "3. Copy credentials to .env file\n"
        )
        return
    
    # Create output directories
    before_dir = DATA_DIR_RAW / city.lower() / f"before_{before}"
    after_dir = DATA_DIR_RAW / city.lower() / f"after_{after}"
    
    # Search and download "before" data
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: Searching for 'BEFORE' imagery")
    logger.info(f"{'='*60}")
    
    before_start = before_date - timedelta(days=days_range)
    before_end = before_date + timedelta(days=days_range)
    
    before_products = downloader.search_products(
        bbox=city_info['bbox'],
        date_start=before_start,
        date_end=before_end,
        max_cloud_cover=max_cloud
    )
    
    if before_products:
        best_before = downloader.get_best_product(before_products)
        if best_before:
            downloader.download_product(best_before['Id'], before_dir)
    else:
        logger.warning("No suitable 'before' products found")
    
    # Search and download "after" data
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Searching for 'AFTER' imagery")
    logger.info(f"{'='*60}")
    
    after_start = after_date - timedelta(days=days_range)
    after_end = after_date + timedelta(days=days_range)
    
    after_products = downloader.search_products(
        bbox=city_info['bbox'],
        date_start=after_start,
        date_end=after_end,
        max_cloud_cover=max_cloud
    )
    
    if after_products:
        best_after = downloader.get_best_product(after_products)
        if best_after:
            downloader.download_product(best_after['Id'], after_dir)
    else:
        logger.warning("No suitable 'after' products found")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"City: {city_info['name']}")
    logger.info(f"Before directory: {before_dir}")
    logger.info(f"After directory: {after_dir}")
    logger.info(
        f"\n⚠️  MANUAL DOWNLOAD REQUIRED:\n"
        f"Due to Copernicus API complexity, please download bands manually:\n\n"
        f"1. Go to: https://dataspace.copernicus.eu/browser/\n"
        f"2. Draw rectangle around coordinates:\n"
        f"   North: {city_info['bbox']['north']}\n"
        f"   South: {city_info['bbox']['south']}\n"
        f"   East: {city_info['bbox']['east']}\n"
        f"   West: {city_info['bbox']['west']}\n"
        f"3. Filter:\n"
        f"   - Sentinel-2 L2A\n"
        f"   - Cloud cover < {max_cloud}%\n"
        f"   - Date range: {before} ± {days_range} days\n"
        f"4. Download bands: {', '.join(SENTINEL_BANDS)}\n"
        f"5. Place in: {before_dir}\n"
        f"\nRepeat for 'after' date: {after}\n"
        f"Place in: {after_dir}\n"
    )


if __name__ == '__main__':
    main()
