"""
Satellite tile fetcher for custom regions
Fetches tiles from Copernicus Data Space API on-demand
"""
import hashlib
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import requests
from PIL import Image
import numpy as np
from sqlalchemy.orm import Session
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import COPERNICUS_USERNAME, COPERNICUS_PASSWORD, SENTINEL_HUB_INSTANCE_ID
from backend.database import CachedTile
from backend.image_processor import ImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "tile_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache expiration (30 days)
CACHE_EXPIRE_DAYS = 30


class TileFetcher:
    """Fetch satellite tiles for custom regions"""
    
    def __init__(self, username: str = None, password: str = None):
        self.username = username or COPERNICUS_USERNAME
        self.password = password or COPERNICUS_PASSWORD
        self.access_token = None
        self.token_expires_at = None
        
        # Proxy configuration (set in .env if needed)
        import os
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
        self.proxies = {}
        if http_proxy:
            self.proxies['http'] = http_proxy
        if https_proxy:
            self.proxies['https'] = https_proxy
        
        if not self.username or not self.password:
            logger.warning("Copernicus credentials not set. Using demo mode.")
            self.demo_mode = True
        else:
            logger.info(f"Initializing with Copernicus credentials for user: {self.username}")
            self.demo_mode = False
    
    def _get_bbox_hash(self, bbox: Dict[str, float], date: str) -> str:
        """Create unique hash for bbox + date combination"""
        bbox_str = f"{bbox['west']:.6f}_{bbox['south']:.6f}_{bbox['east']:.6f}_{bbox['north']:.6f}_{date}"
        return hashlib.md5(bbox_str.encode()).hexdigest()
    
    def _get_cache_path(self, bbox_hash: str) -> Path:
        """Get cache file path for bbox hash"""
        return CACHE_DIR / f"{bbox_hash}.png"
    
    def _get_access_token(self) -> Optional[str]:
        """
        Get OAuth access token from Copernicus Data Space
        Token is cached and reused until expiration
        """
        # Check if we have a valid token
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token
        
        # Get new token
        logger.info("Requesting new access token from Copernicus...")
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public"
        }
        
        try:
            # Increase timeout and add retries
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry
            
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            response = session.post(token_url, data=data, timeout=60, proxies=self.proxies)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 600)  # Default 10 minutes
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # 1 minute buffer
            
            logger.info("✓ Access token obtained successfully")
            return self.access_token
        
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            return None
    
    def check_cache(
        self,
        db: Session,
        bbox: Dict[str, float],
        date: str
    ) -> Optional[Path]:
        """Check if tile is cached and not expired"""
        bbox_hash = self._get_bbox_hash(bbox, date)
        
        # Check database
        cached = db.query(CachedTile).filter(
            CachedTile.bbox_hash == bbox_hash,
            CachedTile.date == date,
            CachedTile.expires_at > datetime.utcnow()
        ).first()
        
        if cached:
            cache_path = Path(cached.image_path)
            if cache_path.exists():
                return cache_path
        
        return None
    
    def save_to_cache(
        self,
        db: Session,
        bbox: Dict[str, float],
        date: str,
        image: Image.Image
    ) -> Path:
        """Save tile to cache"""
        bbox_hash = self._get_bbox_hash(bbox, date)
        cache_path = self._get_cache_path(bbox_hash)
        
        # Save image
        image.save(cache_path, "PNG")
        
        # Save to database
        expires_at = datetime.utcnow() + timedelta(days=CACHE_EXPIRE_DAYS)
        
        cached_tile = CachedTile(
            bbox_hash=bbox_hash,
            date=date,
            image_path=str(cache_path),
            expires_at=expires_at,
            bbox_west=bbox['west'],
            bbox_south=bbox['south'],
            bbox_east=bbox['east'],
            bbox_north=bbox['north']
        )
        
        db.add(cached_tile)
        db.commit()
        
        logger.info(f"✓ Cached tile for {date} at {bbox}")
        return cache_path
    
    def fetch_tile_demo(
        self,
        bbox: Dict[str, float],
        date: str,
        size: Tuple[int, int] = (512, 512)
    ) -> Image.Image:
        """
        Demo mode: Generate a synthetic satellite image
        Used when Copernicus credentials are not available
        """
        logger.info(f"Demo mode: Generating synthetic image for {date}")
        
        # Create a gradient-based fake satellite image
        width, height = size
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate pattern based on coordinates and date
        year = int(date[:4])
        coord_hash = (bbox['west'] + bbox['south']) * 1000
        
        for y in range(height):
            for x in range(width):
                # Create varied terrain patterns
                val = (x + y + coord_hash + year) % 255
                
                # Green for vegetation
                if val < 100:
                    img_array[y, x] = [34, 139, 34]
                # Gray for urban
                elif val < 150:
                    img_array[y, x] = [128, 128, 128]
                # Blue for water
                elif val < 180:
                    img_array[y, x] = [65, 105, 225]
                # Brown for bare land
                else:
                    img_array[y, x] = [139, 90, 43]
        
        # Add some random noise for realism
        noise = np.random.randint(-30, 30, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def fetch_tile_api(
        self,
        bbox: Dict[str, float],
        date: str,
        size: Tuple[int, int] = (1024, 1024)
    ) -> Image.Image:
        """
        Fetch tile from Copernicus Data Space using Sentinel Hub
        
        Args:
            bbox: {'west': lon, 'south': lat, 'east': lon, 'north': lat}
            date: ISO date string (YYYY-MM-DD)
            size: (width, height) in pixels
        """
        # Get OAuth access token
        token = self._get_access_token()
        if not token:
            logger.error("Failed to get access token, falling back to demo mode")
            return self.fetch_tile_demo(bbox, date, size)
        
        # Your Sentinel Hub WMS endpoint
        instance_id = SENTINEL_HUB_INSTANCE_ID or "b874cadc-06ff-41f8-b1c3-4e567e6354c1"
        wms_url = f"https://sh.dataspace.copernicus.eu/ogc/wms/{instance_id}"
        
        # Build WMS request parameters
        # Use date range to find closest available image (Sentinel-2 revisits every 5-10 days)
        from datetime import datetime, timedelta
        req_date = datetime.strptime(date, "%Y-%m-%d")
        # Look for images 60 days before to 1 day after requested date (wider range for better quality)
        start_date = (req_date - timedelta(days=60)).strftime("%Y-%m-%d")
        end_date = (req_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        params = {
            'service': 'WMS',
            'version': '1.3.0',
            'request': 'GetMap',
            'layers': 'TRUE_COLOR',  # Your configured true color layer
            'styles': '',
            'format': 'image/png',
            'transparent': 'false',
            'width': size[0],
            'height': size[1],
            'crs': 'EPSG:4326',
            'bbox': f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}",
            'time': f"{start_date}/{end_date}",  # 60-day range increases chance of finding clear data
            'maxcc': 50,  # Allow up to 50% cloud coverage for better availability globally
            'priority': 'leastCC',  # Request least cloudy image in range
        }
        
        headers = {
            'Authorization': f'Bearer {token}'
        }
        
        try:
            logger.info(f"Fetching from Sentinel Hub (test11 config): {date} at bbox {bbox}")
            
            # Use session with retry logic
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry
            
            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            response = session.get(
                wms_url,
                params=params,
                headers=headers,
                timeout=60,
                proxies=self.proxies
            )
            response.raise_for_status()
            
            # Check if response is an image
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
                logger.warning(f"Response: {response.text[:500]}")
                raise Exception(f"WMS returned non-image response: {content_type}")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(response.content))
            
            # Check if image is completely blank/white (failed to get data from Sentinel Hub)
            img_array = np.array(img)
            img_std = float(np.std(img_array))
            img_mean = float(np.mean(img_array))
            
            if img_std < 1.0:
                logger.warning(f"⚠️ Sentinel Hub returned blank image (std={img_std:.2f}, mean={img_mean:.2f})")
                logger.warning("This usually means no satellite data available for this location/date")
                logger.info("Falling back to demo mode with synthetic data")
                return self.fetch_tile_demo(bbox, date, size)
            
            logger.info(f"✓ Successfully fetched tile from Copernicus API (std={img_std:.1f})")
            return img
        
        except Exception as e:
            logger.error(f"Error fetching from Copernicus API: {e}")
            logger.info("Falling back to demo mode")
            return self.fetch_tile_demo(bbox, date, size)
    
    def get_tile(
        self,
        db: Session,
        bbox: Dict[str, float],
        date: str,
        size: Tuple[int, int] = (512, 512),
        force_refresh: bool = False
    ) -> Path:
        """
        Get satellite tile from Sentinel Hub (from cache or fetch new)
        
        Args:
            db: Database session
            bbox: Bounding box dict
            date: Date string (YYYY-MM-DD)
            size: Image size
            force_refresh: Skip cache and fetch fresh
        
        Returns:
            Path to image file
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_path = self.check_cache(db, bbox, date)
            if cached_path:
                logger.info(f"✓ Cache hit for {date}")
                return cached_path
        
        # Fetch new tile
        logger.info(f"Fetching tile for {date} at {bbox}")
        
        if self.demo_mode:
            image = self.fetch_tile_demo(bbox, date, size)
        else:
            # Fetch from Sentinel Hub
            image = self.fetch_tile_api(bbox, date, size)
        
        # Save to cache
        cache_path = self.save_to_cache(db, bbox, date, image)
        
        return cache_path
    
    def cleanup_expired_cache(self, db: Session):
        """Remove expired cache entries"""
        expired = db.query(CachedTile).filter(
            CachedTile.expires_at < datetime.utcnow()
        ).all()
        
        for cached in expired:
            # Delete file
            cache_path = Path(cached.image_path)
            if cache_path.exists():
                cache_path.unlink()
            
            # Delete database entry
            db.delete(cached)
        
        db.commit()
        logger.info(f"✓ Cleaned up {len(expired)} expired cache entries")


# Global tile fetcher instance
_tile_fetcher = None

def get_tile_fetcher() -> TileFetcher:
    """Get global tile fetcher instance"""
    global _tile_fetcher
    if _tile_fetcher is None:
        _tile_fetcher = TileFetcher()
    return _tile_fetcher
