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
    
    # ── Evalscript for multi-band fetch (B01, B02, B03, B04, B05, B07, B08, B11) ──
    MULTIBAND_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B01", "B02", "B03", "B04", "B05", "B07", "B08", "B11"], units: "DN" }],
    output: { bands: 8, sampleType: "UINT16" }
  };
}
function evaluatePixel(sample) {
  return [sample.B01, sample.B02, sample.B03, sample.B04,
          sample.B05, sample.B07, sample.B08, sample.B11];
}
"""

    PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

    # Band index mapping in the 8-channel output
    BAND_INDEX = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3,
        "B05": 4, "B07": 5, "B08": 6, "B11": 7,
    }

    def fetch_multiband_tile(
        self,
        bbox: Dict[str, float],
        date: str,
        size: Tuple[int, int] = (512, 512),
    ) -> Optional[np.ndarray]:
        """
        Fetch raw multi-spectral Sentinel-2 bands via the Process API.

        Returns an ndarray of shape (H, W, 8) as float32 reflectances [0-1],
        with bands [B01, B02, B03, B04, B05, B07, B08, B11].
        Returns None on failure (caller should fall back to RGB-based approximation).
        """
        # Check multiband cache first
        mb_hash = hashlib.md5(
            f"{bbox['west']:.6f}_{bbox['south']:.6f}_{bbox['east']:.6f}_{bbox['north']:.6f}_{date}_mb".encode()
        ).hexdigest()
        mb_cache_path = CACHE_DIR / f"{mb_hash}_multiband.npy"
        if mb_cache_path.exists():
            try:
                arr = np.load(mb_cache_path)
                if arr.ndim == 3 and arr.shape[2] == 8:
                    logger.info(f"✓ Multiband cache hit for {date}")
                    return arr
            except Exception:
                pass

        if self.demo_mode:
            return self._generate_demo_multiband(bbox, date, size)

        token = self._get_access_token()
        if not token:
            logger.warning("No access token for multiband fetch, using demo bands")
            return self._generate_demo_multiband(bbox, date, size)

        req_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = (req_date - timedelta(days=60)).strftime("%Y-%m-%d")
        end_date = (req_date + timedelta(days=1)).strftime("%Y-%m-%d")

        body = {
            "input": {
                "bounds": {
                    "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z",
                        },
                        "maxCloudCoverage": 50,
                        "mosaickingOrder": "leastCC",
                    },
                }],
            },
            "output": {
                "width": size[0],
                "height": size[1],
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": self.MULTIBAND_EVALSCRIPT,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        }

        try:
            logger.info(f"Fetching multiband tile for {date} via Process API")
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry

            session = requests.Session()
            retry_strategy = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)

            response = session.post(
                self.PROCESS_API_URL,
                json=body,
                headers=headers,
                timeout=90,
                proxies=self.proxies,
            )

            if response.status_code == 401:
                # Force token refresh and retry once
                self.access_token = None
                self.token_expires_at = None
                token = self._get_access_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                    response = session.post(
                        self.PROCESS_API_URL,
                        json=body,
                        headers=headers,
                        timeout=90,
                        proxies=self.proxies,
                    )

            if not response.ok:
                logger.warning(f"Multiband fetch failed HTTP {response.status_code}: {response.text[:300]}")
                return self._generate_demo_multiband(bbox, date, size)

            content_type = response.headers.get("Content-Type", "")
            raw = response.content

            # Try to decode as TIFF first, then fall back to PIL
            arr = None
            try:
                import rasterio
                from rasterio.io import MemoryFile
                with MemoryFile(raw) as mem:
                    with mem.open() as ds:
                        bands = ds.read()  # [C, H, W]
                        if bands.shape[0] >= 8:
                            arr = np.moveaxis(bands[:8], 0, -1)  # [H, W, 8]
            except Exception as rio_exc:
                logger.info(f"Rasterio decode failed ({rio_exc}), trying PIL")
                try:
                    img = Image.open(io.BytesIO(raw))
                    arr = np.array(img)
                except Exception as pil_exc:
                    logger.warning(f"PIL decode also failed: {pil_exc}")

            if arr is None or arr.ndim < 3 or arr.shape[2] < 8:
                logger.warning("Multiband fetch returned insufficient bands, using demo")
                return self._generate_demo_multiband(bbox, date, size)

            # Convert DN to reflectance [0-1]
            arr = arr.astype(np.float32) / 10000.0
            arr = np.clip(arr, 0.0, 1.0)

            # Sanity check: if too uniform, it's likely blank
            if float(np.std(arr)) < 0.001:
                logger.warning("Multiband tile appears blank, using demo")
                return self._generate_demo_multiband(bbox, date, size)

            # Cache the result
            np.save(mb_cache_path, arr)
            logger.info(f"✓ Multiband tile fetched and cached for {date} ({arr.shape})")
            return arr

        except Exception as e:
            logger.warning(f"Multiband fetch error: {e}")
            return self._generate_demo_multiband(bbox, date, size)

    def _generate_demo_multiband(
        self,
        bbox: Dict[str, float],
        date: str,
        size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Generate synthetic multi-spectral band data for demo mode.
        Returns (H, W, 8) float32 array simulating [B01, B02, B03, B04, B05, B07, B08, B11].
        """
        height, width = size
        rng = np.random.RandomState(
            abs(hash(f"{bbox['west']:.4f}_{bbox['south']:.4f}_{date}")) % (2**31)
        )

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

        # Base terrain patterns
        veg_pattern = np.clip(0.5 + 0.3 * np.sin(xx / 60.0) * np.cos(yy / 45.0), 0, 1)
        water_pattern = np.clip(0.3 * np.sin((xx + yy) / 100.0), 0, 1)
        urban_pattern = np.clip(0.4 + 0.2 * np.sin(xx / 30.0) * np.sin(yy / 30.0), 0, 1)

        # Simulate realistic band reflectances
        b01_coastal = np.clip(0.04 + 0.02 * water_pattern + rng.normal(0, 0.005, (height, width)), 0, 0.3)
        b02_blue = np.clip(0.06 + 0.03 * water_pattern + 0.02 * urban_pattern + rng.normal(0, 0.005, (height, width)), 0, 0.3)
        b03_green = np.clip(0.07 + 0.05 * veg_pattern + 0.02 * water_pattern + rng.normal(0, 0.005, (height, width)), 0, 0.3)
        b04_red = np.clip(0.05 + 0.03 * urban_pattern - 0.02 * veg_pattern + rng.normal(0, 0.005, (height, width)), 0, 0.3)
        b05_rededge = np.clip(0.10 + 0.08 * veg_pattern + rng.normal(0, 0.005, (height, width)), 0, 0.5)
        b07_rededge2 = np.clip(0.20 + 0.15 * veg_pattern + rng.normal(0, 0.008, (height, width)), 0, 0.6)
        b08_nir = np.clip(0.25 + 0.25 * veg_pattern - 0.10 * water_pattern + rng.normal(0, 0.01, (height, width)), 0, 0.7)
        b11_swir = np.clip(0.15 + 0.10 * urban_pattern - 0.05 * veg_pattern - 0.08 * water_pattern + rng.normal(0, 0.008, (height, width)), 0, 0.5)

        bands = np.stack([b01_coastal, b02_blue, b03_green, b04_red,
                          b05_rededge, b07_rededge2, b08_nir, b11_swir], axis=-1)
        return bands.astype(np.float32)

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
