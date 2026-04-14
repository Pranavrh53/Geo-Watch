"""
FastAPI Backend for Satellite Change Detection System
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import sys
import json
import logging
import hashlib
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    API_HOST, API_PORT,
    CITIES, DATA_DIR_RESULTS,
    DATA_DIR_PROCESSED
)

# Import auth and database modules
from backend.database import get_db, init_db, User, AnalysisHistory
from backend.auth import (
    UserCreate, UserLogin, Token, UserResponse,
    get_current_active_user, authenticate_user, create_access_token,
    create_user, ACCESS_TOKEN_EXPIRE_MINUTES
)
from backend.tile_fetcher import get_tile_fetcher
from backend.unified_change_detector import get_unified_detector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_image_quality_simple(img_array: np.ndarray) -> Dict[str, object]:
    std_dev = float(np.std(img_array))
    mean_intensity = float(np.mean(img_array))
    is_valid = std_dev >= 1.0
    return {
        "is_valid": is_valid,
        "std_dev": std_dev,
        "mean_intensity": mean_intensity,
        "quality": "Good" if std_dev >= 45 else "Acceptable" if std_dev >= 20 else "Poor",
        "reason": None if is_valid else "Image is nearly uniform/blank",
    }


FEATURE_RENDER_MODES = {
    "natural_color": "Natural Color",
    "geology": "Geology",
    "ndvi": "NDVI",
    "bathymetric": "Bathymetric",
    "infrared": "Infrared",
    "moisture_index": "Moisture Index",
    "ndwi": "NDWI",
}


def _normalize01(arr: np.ndarray) -> np.ndarray:
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if amax - amin < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - amin) / (amax - amin)).astype(np.float32)


def _stretch_8bit(bands: np.ndarray, lower_percent: float = 2, higher_percent: float = 98) -> np.ndarray:
    """
    Percentile contrast stretch — same method as Orbitia / standard remote sensing.
    Input:  (H, W, C) any dtype
    Output: (H, W, C) uint8 [0-255]
    """
    out = np.zeros_like(bands, dtype=np.float64)
    num_channels = bands.shape[2] if bands.ndim == 3 else 1
    if bands.ndim == 2:
        bands = bands[:, :, np.newaxis]
        out = out[:, :, np.newaxis]

    for i in range(num_channels):
        c = float(np.percentile(bands[:, :, i], lower_percent))
        d = float(np.percentile(bands[:, :, i], higher_percent))
        if d - c < 1e-10:
            out[:, :, i] = 0
        else:
            t = (bands[:, :, i].astype(np.float64) - c) * 255.0 / (d - c)
            t = np.clip(t, 0, 255)
            out[:, :, i] = t

    return out.astype(np.uint8)


def _index_to_grayscale(index_arr: np.ndarray) -> np.ndarray:
    """
    Convert a spectral index (e.g. NDVI, NDWI, NDMI) to a grayscale uint8 image
    using percentile stretch. Bright = high index, Dark = low index.
    Same approach as Orbitia's calculate_ndvi / calculate_ndwi / calculate_moisture.
    """
    idx = np.expand_dims(index_arr, -1) if index_arr.ndim == 2 else index_arr
    stretched = _stretch_8bit(idx.astype(np.float64))
    # Return as 3-channel grayscale (RGB all same) so it works with PNG saving
    gray = stretched[:, :, 0] if stretched.ndim == 3 else stretched
    return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# PRIMARY: Multiband rendering (uses real Sentinel-2 spectral bands)
# Approach: Exactly like Orbitia — direct band composites with percentile
#           stretch for composites; grayscale percentile-stretched output
#           for spectral indices (NDVI, NDWI, Moisture).
# ---------------------------------------------------------------------------

def _apply_feature_render_multiband(bands: np.ndarray, render_mode: str) -> np.ndarray:
    """
    Render feature analysis from real multi-band satellite data.

    bands: (H, W, 8) float32 reflectances [0-1]
           channels = [B01, B02, B03, B04, B05, B07, B08, B11]
    """
    mode = (render_mode or "natural_color").lower()

    B01 = bands[:, :, 0]   # Coastal Aerosol (Band 1)
    B02 = bands[:, :, 1]   # Blue (Band 2)
    B03 = bands[:, :, 2]   # Green (Band 3)
    B04 = bands[:, :, 3]   # Red (Band 4)
    B05 = bands[:, :, 4]   # Red Edge 1 (Band 5)
    B07 = bands[:, :, 5]   # Red Edge 3 (Band 7)
    B08 = bands[:, :, 6]   # NIR (Band 8)
    B11 = bands[:, :, 7]   # SWIR 1 (Band 11)

    eps = 1e-10

    # ── Natural Color: R=B04, G=B03, B=B02 ────────────────────────────────
    if mode == "natural_color":
        rgb = np.stack([B04, B03, B02], axis=-1)
        return _stretch_8bit(rgb)

    # ── Geology (False Color SWIR): R=B11, G=B08, B=B02 ──────────────────
    #    Orbitia Sentinel: bands 12,11,2  →  Our mapping: B11→SWIR, B08→NIR, B02→Blue
    if mode == "geology":
        rgb = np.stack([B11, B08, B02], axis=-1)
        return _stretch_8bit(rgb)

    # ── NDVI: (NIR - Red) / (NIR + Red) → Grayscale ──────────────────────
    #    Bright/White = healthy vegetation, Dark = bare/water
    if mode == "ndvi":
        ndvi = (B08 - B04) / (B08 + B04 + eps)
        return _index_to_grayscale(ndvi)

    # ── Bathymetric: R=B04, G=B03, B=B01 (Masked strictly to water) ───────
    #    Coastal aerosol penetrates water → reveals depth. Land kept natural.
    if mode == "bathymetric":
        # Calculate natural color for land
        nat_rgb = np.stack([B04, B03, B02], axis=-1)
        nat_stretch = _stretch_8bit(nat_rgb)
        
        # Calculate bathymetric for water
        bathy_rgb = np.stack([B04, B03, B01], axis=-1)
        bathy_stretch = _stretch_8bit(bathy_rgb)
        
        # Mask water using NDWI (> 0 indicates water)
        ndwi = (B03 - B08) / (B03 + B08 + eps)
        water_mask = np.expand_dims(ndwi > 0.0, -1)
        
        # Combine: natural on land, bathymetric on water
        out = np.where(water_mask, bathy_stretch, nat_stretch)
        return out.astype(np.uint8)

    # ── Color Infrared (CIR): R=B08(NIR), G=B04(Red), B=B03(Green) ───────
    #    Red/Pink = healthy vegetation, Dark = water/barren
    if mode == "infrared":
        rgb = np.stack([B08, B04, B03], axis=-1)
        return _stretch_8bit(rgb)

    # ── Moisture Index (NDMI): (NIR - SWIR) / (NIR + SWIR) → Grayscale ───
    #    Bright = high moisture, Dark = dry
    if mode == "moisture_index":
        ndmi = (B08 - B11) / (B08 + B11 + eps)
        return _index_to_grayscale(ndmi)

    # ── NDWI: (Green - NIR) / (Green + NIR) → Grayscale ──────────────────
    #    Bright = water, Dark = land
    if mode == "ndwi":
        ndwi = (B03 - B08) / (B03 + B08 + eps)
        return _index_to_grayscale(ndwi)

    # Fallback: natural color
    rgb = np.stack([B04, B03, B02], axis=-1)
    return _stretch_8bit(rgb)


# ---------------------------------------------------------------------------
# FALLBACK: RGB-only pseudo-rendering (when multiband data unavailable)
# ---------------------------------------------------------------------------

def _apply_feature_render_mode(rgb_u8: np.ndarray, render_mode: str) -> np.ndarray:
    """
    Approximate feature rendering from 3-channel RGB when real bands unavailable.
    Uses pseudo-NIR/SWIR derived from visible channels.
    """
    mode = (render_mode or "natural_color").lower()
    if mode == "natural_color":
        return rgb_u8

    rgb = rgb_u8.astype(np.float64) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Pseudo spectral approximations
    pseudo_nir  = np.clip(0.50 * r + 0.40 * g - 0.10 * b + 0.12, 0.0, 1.0)
    pseudo_swir = np.clip(0.45 * r + 0.20 * g - 0.15 * b + 0.05, 0.0, 1.0)
    eps = 1e-10

    # ── Geology: SWIR-false-color approximation ───────────────────────────
    if mode == "geology":
        composite = np.stack([pseudo_swir, pseudo_nir, b], axis=-1)
        return _stretch_8bit(composite)

    # ── NDVI → Grayscale ──────────────────────────────────────────────────
    if mode == "ndvi":
        ndvi = (pseudo_nir - r) / (pseudo_nir + r + eps)
        return _index_to_grayscale(ndvi)

    # ── Bathymetric (Masked strictly to water) ────────────────────────────
    if mode == "bathymetric":
        coastal_approx = np.clip(b * 0.7 + g * 0.3, 0.0, 1.0)
        composite = np.stack([r, g, coastal_approx], axis=-1)
        bathy_stretch = _stretch_8bit(composite)
        
        # Mask using NDWI approx
        ndwi = (g - pseudo_nir) / (g + pseudo_nir + eps)
        water_mask = np.expand_dims(ndwi > 0.0, -1)
        
        out = np.where(water_mask, bathy_stretch, rgb_u8)
        return out.astype(np.uint8)

    # ── Infrared (CIR) ───────────────────────────────────────────────────
    if mode == "infrared":
        composite = np.stack([pseudo_nir, r, g], axis=-1)
        return _stretch_8bit(composite)

    # ── Moisture Index → Grayscale ────────────────────────────────────────
    if mode == "moisture_index":
        moisture = (pseudo_nir - pseudo_swir) / (pseudo_nir + pseudo_swir + eps)
        return _index_to_grayscale(moisture)

    # ── NDWI → Grayscale ──────────────────────────────────────────────────
    if mode == "ndwi":
        ndwi = (g - pseudo_nir) / (g + pseudo_nir + eps)
        return _index_to_grayscale(ndwi)

    return rgb_u8

# In-memory task status storage (use Redis in production)
task_status = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    # --- startup ---
    init_db()
    logger.info("\u2713 Database initialized")
    yield
    # --- shutdown (add cleanup here if needed) ---


# Create FastAPI app
app = FastAPI(
    title="Satellite Change Detection API",
    description="AI-based geospatial change detection system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

# Request/Response Models
class AnalysisRequest(BaseModel):
    city: str
    before_date: str  # YYYY-MM-DD
    after_date: str   # YYYY-MM-DD
    # Streaming mode is now default - no separate steps needed

class CityInfo(BaseModel):
    name: str
    country: str
    bbox: Dict[str, float]
    center: Dict[str, float]

class ChangeStats(BaseModel):
    change_type: str
    name: str
    pixels: int
    area_sqm: float
    area_hectares: float
    area_acres: float
    area_sqkm: float
    color: List[int]

class AnalysisResult(BaseModel):
    city: str
    before_date: str
    after_date: str
    status: str
    timestamp: str
    changes: List[ChangeStats]
    total_change_hectares: float




# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Satellite Change Detection API",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "register": "/api/auth/register",
            "login": "/api/auth/login",
            "cities": "/api/cities",
            "fetch_tile": "/api/tile/fetch",
            "analyze": "/api/analyze",
            "results": "/api/results/{city}",
            "visualization": "/api/visualization/{city}/{filename}"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ========== Authentication Endpoints ==========

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        db_user = create_user(db, user)
        return db_user
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return current_user


# ========== Tile Fetching Endpoints ==========

class TileFetchRequest(BaseModel):
    bbox: Dict[str, float]  # {west, south, east, north}
    date: str  # YYYY-MM-DD
    size: Optional[int] = 512
    render_mode: Optional[str] = "natural_color"


@app.post("/api/tile/fetch")
async def fetch_tile(
    request: TileFetchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Fetch satellite tile for custom region.
    For non-natural-color modes, uses real multi-spectral band data from
    the Sentinel Hub Process API to compute accurate composites/indices.
    """
    try:
        fetcher = get_tile_fetcher()
        
        # Validate bbox
        bbox = request.bbox
        if not all(k in bbox for k in ['west', 'south', 'east', 'north']):
            raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        render_mode = (request.render_mode or "natural_color").lower()
        if render_mode not in FEATURE_RENDER_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid render_mode '{render_mode}'. Allowed: {', '.join(FEATURE_RENDER_MODES.keys())}"
            )

        # Get tile (from cache or fetch new) — always needed for quality check
        size = (request.size, request.size)
        tile_path = fetcher.get_tile(db, bbox, request.date, size)
        
        # Check image quality
        from PIL import Image
        img = Image.open(tile_path).convert("RGB")
        quality_check = check_image_quality_simple(np.array(img))
        
        image_name = tile_path.name
        data_source = "rgb_fallback"

        if render_mode != "natural_color":
            # Build a cache key that includes render_mode and uses "mb" suffix
            # to distinguish multiband-rendered images from old RGB-approximated ones
            mb_seed = f"{tile_path.stem}:{render_mode}:mb"
            mb_name = f"{hashlib.md5(mb_seed.encode()).hexdigest()}_{render_mode}_mb.png"
            mb_path = tile_path.parent / mb_name

            if mb_path.exists():
                # Already processed with multiband data
                image_name = mb_name
                data_source = "multiband_cached"
            else:
                # Try to fetch real multi-spectral bands
                multiband = fetcher.fetch_multiband_tile(bbox, request.date, size)

                if multiband is not None and multiband.ndim == 3 and multiband.shape[2] == 8:
                    # Use real spectral bands for accurate rendering
                    logger.info(f"Using multiband data for {render_mode} rendering")
                    processed_array = _apply_feature_render_multiband(multiband, render_mode)
                    Image.fromarray(processed_array).save(mb_path, "PNG")
                    image_name = mb_name
                    data_source = "multiband"
                else:
                    # Fallback: use RGB approximation
                    logger.warning(f"Multiband unavailable, using RGB fallback for {render_mode}")
                    fallback_seed = f"{tile_path.stem}:{render_mode}:fallback"
                    fallback_name = f"{hashlib.md5(fallback_seed.encode()).hexdigest()}_{render_mode}.png"
                    fallback_path = tile_path.parent / fallback_name

                    if not fallback_path.exists():
                        processed_array = _apply_feature_render_mode(np.array(img), render_mode)
                        Image.fromarray(processed_array).save(fallback_path, "PNG")

                    image_name = fallback_name
                    data_source = "rgb_fallback"

        # Return file path as URL with quality info
        return {
            "status": "success",
            "image_url": f"/api/tile/image/{image_name}",
            "cached": True,
            "date": request.date,
            "bbox": bbox,
            "source": "sentinel",
            "render_mode": render_mode,
            "render_label": FEATURE_RENDER_MODES[render_mode],
            "data_source": data_source,
            "quality": quality_check
        }
    
    except Exception as e:
        logger.error(f"Tile fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tile/image/{image_name}")
async def get_tile_image(image_name: str):
    """Get cached tile image (public endpoint for browser img tags)"""
    cache_dir = Path(__file__).parent.parent / "data" / "tile_cache"
    image_path = cache_dir / image_name
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        image_path, 
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=86400"
        }
    )


# ========== Change Detection Endpoints ==========

class ChangeDetectionRequest(BaseModel):
    bbox: Dict[str, float]
    before_date: str
    after_date: str
    sensitivity: float = 30.0
    enhance_images: bool = True


@app.post("/api/analyze/detect-changes")
async def detect_changes(
    request: ChangeDetectionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Unified multi-temporal change detection.
    Legacy RGB/ExG rule-based flow has been removed.
    """
    try:
        fetcher = get_tile_fetcher()

        logger.info(f"Starting change detection: {request.before_date} → {request.after_date}")

        # Step 1: Fetch both images
        before_path = fetcher.get_tile(db, request.bbox, request.before_date, (1024, 1024))
        after_path = fetcher.get_tile(db, request.bbox, request.after_date, (1024, 1024))

        # Step 2: Load and check quality
        from PIL import Image
        before_img = Image.open(before_path).convert("RGB")
        after_img = Image.open(after_path).convert("RGB")

        before_quality = check_image_quality_simple(np.array(before_img))
        after_quality = check_image_quality_simple(np.array(after_img))

        logger.info("Before image quality: %s", before_quality["quality"])
        logger.info("After image quality: %s", after_quality["quality"])

        if not before_quality['is_valid'] or not after_quality['is_valid']:
            return {
                "status": "error",
                "message": "One or both images are blank/invalid",
                "before_quality": before_quality,
                "after_quality": after_quality,
                "suggestion": "Try different dates or a different region"
            }

        # Step 3: Run unified temporal detector
        logger.info("Running unified multi-temporal detector...")
        detector = get_unified_detector()
        results = detector.analyze_changes(
            bbox=request.bbox,
            before_date=request.before_date,
            after_date=request.after_date,
            before_rgb=np.array(before_img),
            after_rgb=np.array(after_img),
        )

        summary = results["change_summary"]

        return {
            "status": "success",
            "before_date": request.before_date,
            "after_date": request.after_date,
            "method": results["method"],
            "change_detected": summary["change_percent"] > 1.0,
            "change_percentage": summary["change_percent"],
            "change_area_hectares": summary["change_area_hectares"],
            "change_area_km2": round(summary["change_area_hectares"] / 100.0, 4),
            "severity": summary.get("severity", "Low"),
            "confidence": summary.get("confidence", 0),
            "overlay": results["overlays"]["classified"],
            "heatmap": results["overlays"]["change_probability"],
            "image_quality": {
                "before": {
                    "quality": before_quality['quality'],
                    "std_dev": round(float(before_quality.get('std_dev', 0.0)), 2),
                },
                "after": {
                    "quality": after_quality['quality'],
                    "std_dev": round(float(after_quality.get('std_dev', 0.0)), 2),
                }
            },
            "analysis_details": {
                "total_pixels": summary["total_pixels"],
                "changed_pixels": summary["changed_pixels"],
                "years_used": results["years_used"],
                "temporal_windows": results.get("temporal_windows", {}),
            },
            "categories": results["classified_changes"],
        }
    
    except Exception as e:
        logger.error(f"Change detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/overlay/image/{image_name}")
async def get_overlay_image(image_name: str):
    """Get change detection overlay image"""
    overlay_dir = Path(__file__).parent.parent / "data" / "overlays"
    image_path = overlay_dir / image_name
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found")
    
    return FileResponse(
        image_path,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600"
        }
    )


# ========== Analysis History Endpoints ==========

class SaveAnalysisRequest(BaseModel):
    region_name: Optional[str] = None
    bbox: Dict[str, float]
    before_date: str
    after_date: str
    result_json: Optional[Dict] = None


@app.post("/api/history/save")
async def save_analysis(
    request: SaveAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Save analysis to user history"""
    history = AnalysisHistory(
        user_id=current_user.id,
        region_name=request.region_name,
        bbox_west=request.bbox['west'],
        bbox_south=request.bbox['south'],
        bbox_east=request.bbox['east'],
        bbox_north=request.bbox['north'],
        before_date=request.before_date,
        after_date=request.after_date,
        result_json=json.dumps(request.result_json) if request.result_json else None
    )
    
    db.add(history)
    db.commit()
    db.refresh(history)
    
    return {"status": "success", "id": history.id}


@app.get("/api/history/list")
async def list_analysis_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's analysis history"""
    history = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id
    ).order_by(AnalysisHistory.created_at.desc()).limit(20).all()
    
    return [
        {
            "id": h.id,
            "region_name": h.region_name,
            "bbox": {
                "west": h.bbox_west,
                "south": h.bbox_south,
                "east": h.bbox_east,
                "north": h.bbox_north
            },
            "before_date": h.before_date,
            "after_date": h.after_date,
            "created_at": h.created_at.isoformat(),
            "status": h.status
        }
        for h in history
    ]


@app.get("/api/cities")
async def get_cities():
    """Get list of available cities with their metadata"""
    cities_list = []
    
    for city_id, city_data in CITIES.items():
        city_info = CityInfo(**city_data)
        
        # Check if city has results
        city_results_dir = DATA_DIR_RESULTS / city_id
        has_results = city_results_dir.exists() and list(city_results_dir.glob('*/change_detection_results.json'))
        
        cities_list.append({
            "id": city_id,
            "info": city_info.dict(),
            "has_results": has_results
        })
    
    return {
        "cities": cities_list,
        "count": len(cities_list)
    }


@app.get("/api/cities/{city_id}")
async def get_city(city_id: str):
    """Get specific city metadata"""
    if city_id not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city_id}' not found")
    
    city_data = CITIES[city_id]
    
    # Find available results
    city_results_dir = DATA_DIR_RESULTS / city_id
    available_results = []
    
    if city_results_dir.exists():
        for result_dir in city_results_dir.iterdir():
            if result_dir.is_dir():
                results_file = result_dir / 'change_detection_results.json'
                if results_file.exists():
                    # Parse directory name to get dates
                    dir_name = result_dir.name
                    available_results.append({
                        "directory": dir_name,
                        "results_file": str(results_file)
                    })
    
    return {
        "id": city_id,
        "info": city_data,
        "available_results": available_results
    }


@app.post("/api/analyze")
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Trigger change detection analysis
    
    This can run the full pipeline or just specific steps
    """
    city_id = request.city.lower()
    
    if city_id not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city_id}' not found")
    
    # Create task ID
    task_id = f"{city_id}_{request.before_date}_{request.after_date}_{datetime.now().timestamp()}"
    
    # Update task status
    task_status[task_id] = {
        "status": "queued",
        "city": city_id,
        "before_date": request.before_date,
        "after_date": request.after_date,
        "started_at": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Run analysis in background
    background_tasks.add_task(
        run_analysis_pipeline,
        task_id,
        request
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Analysis started. Use /api/status/{task_id} to check progress."
    }


@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of an analysis task"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    
    return task_status[task_id]


@app.get("/api/results/{city_id}")
async def get_results(city_id: str, before_date: Optional[str] = None, after_date: Optional[str] = None):
    """
    Get change detection results for a city
    
    If dates are not specified, returns the most recent results
    """
    if city_id not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city_id}' not found")
    
    city_results_dir = DATA_DIR_RESULTS / city_id
    
    if not city_results_dir.exists():
        raise HTTPException(status_code=404, detail=f"No results found for city '{city_id}'")
    
    # Find matching results
    result_dirs = list(city_results_dir.iterdir())
    
    if not result_dirs:
        raise HTTPException(status_code=404, detail=f"No results found for city '{city_id}'")
    
    # Filter by dates if specified
    if before_date or after_date:
        result_dirs = [
            d for d in result_dirs
            if (not before_date or before_date in d.name) and
               (not after_date or after_date in d.name)
        ]
    
    if not result_dirs:
        raise HTTPException(status_code=404, detail=f"No matching results found")
    
    # Get most recent (or matching) result
    result_dir = sorted(result_dirs)[-1]
    
    # Check for streaming results format (new)
    results_file = result_dir / 'changes_summary.json'
    if not results_file.exists():
        # Fallback to old format
        results_file = result_dir / 'change_detection_results.json'
    
    if not results_file.exists():
        raise HTTPException(status_code=404, detail=f"Results file not found")
    
    # Load results
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Format response
    changes = []
    total_change = 0
    
    from config import CHANGE_CLASSES
    for change_type, stats in results_data['summary'].items():
        if stats['area_hectares'] > 0:
            config = CHANGE_CLASSES.get(change_type, {})
            changes.append(ChangeStats(
                change_type=change_type,
                name=config.get('name', change_type),
                pixels=stats['pixels'],
                area_sqm=stats['area_sqm'],
                area_hectares=stats['area_hectares'],
                area_acres=stats['area_acres'],
                area_sqkm=stats['area_sqkm'],
                color=config.get('color', [128, 128, 128])
            ))
            total_change += stats['area_hectares']
    
    # Parse dates from directory name
    dir_name = result_dir.name
    parts = dir_name.split('_vs_')
    before_date_str = parts[0].replace('before_', '') if len(parts) > 0 else "unknown"
    after_date_str = parts[1].replace('after_', '') if len(parts) > 1 else "unknown"
    
    result = AnalysisResult(
        city=city_id,
        before_date=before_date_str,
        after_date=after_date_str,
        status="completed",
        timestamp=datetime.now().isoformat(),
        changes=changes,
        total_change_hectares=total_change
    )
    
    return result


@app.get("/api/visualization/{city_id}/{filename}")
async def get_visualization(city_id: str, filename: str):
    """Get visualization image file"""
    if city_id not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city_id}' not found")
    
    # Find visualization file
    city_results_dir = DATA_DIR_RESULTS / city_id
    
    if not city_results_dir.exists():
        raise HTTPException(status_code=404, detail=f"No results found for city '{city_id}'")
    
    # Search in all result directories
    for result_dir in city_results_dir.iterdir():
        vis_file = result_dir / filename
        if vis_file.exists():
            return FileResponse(vis_file)
        
        # Also check in visualizations subdirectory
        vis_file = result_dir / 'visualizations' / filename
        if vis_file.exists():
            return FileResponse(vis_file)
    
    raise HTTPException(status_code=404, detail=f"Visualization file '{filename}' not found")


@app.get("/api/tiles/{city_id}")
async def get_tiles_geojson(city_id: str):
    """
    Get GeoJSON representation of analyzed tiles
    Useful for displaying on map
    """
    if city_id not in CITIES:
        raise HTTPException(status_code=404, detail=f"City '{city_id}' not found")
    
    # This would need proper implementation with georeferencing
    # For now, return placeholder
    return {
        "type": "FeatureCollection",
        "features": []
    }


# ---- Feature Detection Endpoints ----

class FeatureDetectionRequest(BaseModel):
    before_image_path: str
    after_image_path: str
    pixel_resolution: Optional[float] = 10.0


@app.post("/api/ai/change-detection")
async def detect_changes_api(
    request: FeatureDetectionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Colour-coded change detection between two satellite images.
    Classifies changes as: new construction, vegetation loss, new vegetation,
    water change, demolition, or other.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"🔍  CHANGE DETECTION ENDPOINT", flush=True)
    print(f"User: {current_user.username}", flush=True)
    print(f"Before: {request.before_image_path}", flush=True)
    print(f"After: {request.after_image_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:
        project_root = Path(__file__).parent.parent
        before_path = project_root / request.before_image_path
        after_path = project_root / request.after_image_path

        if not before_path.exists():
            raise HTTPException(status_code=404, detail=f"Before image not found: {before_path}")
        if not after_path.exists():
            raise HTTPException(status_code=404, detail=f"After image not found: {after_path}")

        from PIL import Image
        before_img = Image.open(str(before_path)).convert("RGB")
        after_img = Image.open(str(after_path)).convert("RGB")

        # Backward-compatible wrapper over unified pipeline.
        h, w = before_img.height, before_img.width
        bbox = {"west": 0.0, "south": 0.0, "east": float(w), "north": float(h)}
        detector = get_unified_detector()
        unified = detector.analyze_changes(
            bbox=bbox,
            before_date="2024-01-01",
            after_date="2025-01-01",
            before_rgb=np.array(before_img),
            after_rgb=np.array(after_img),
            pixel_resolution=request.pixel_resolution or 10.0,
        )

        summary = unified["change_summary"]
        return {
            "status": "success",
            "total_pixels": summary["total_pixels"],
            "changed_pixels": summary["changed_pixels"],
            "change_percentage": summary["change_percent"],
            "change_area_hectares": summary["change_area_hectares"],
            "severity": summary.get("severity", "Low"),
            "change_type": "Multi-temporal",
            "confidence": summary.get("confidence", 0),
            "categories": unified["classified_changes"],
            "before_overlay": unified["overlays"]["classified"],
            "after_overlay": unified["overlays"]["classified"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Change detection failed: {str(e)}")


# ---- Spectral-based Deforestation Detection ----

class DeforestationRequest(BaseModel):
    bbox: Dict[str, float]   # {west, south, east, north}
    before_date: str         # YYYY-MM-DD
    after_date: str          # YYYY-MM-DD
    before_image_path: Optional[str] = None  # optional true-color for overlay
    after_image_path: Optional[str] = None
    pixel_resolution: Optional[float] = 10.0


@app.post("/api/ai/deforestation")
def detect_deforestation(
    request: DeforestationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Spectral deforestation detection using NDVI + NDBI indices.

    Uses Sentinel-2 Process API to fetch real NIR, SWIR, SCL bands.
    Much more accurate than RGB-only methods.

    Categories detected:
      - Deforestation due to construction (NDVI drop + NDBI rise)
      - Vegetation loss (general)
      - New construction on bare land
      - Vegetation recovery
    """
    print(f"\n{'='*60}", flush=True)
    print(f"🌳  SPECTRAL DEFORESTATION DETECTION ENDPOINT", flush=True)
    print(f"User: {current_user.username}", flush=True)
    print(f"Dates: {request.before_date} → {request.after_date}", flush=True)
    print(f"Bbox: {request.bbox}", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:
        before_arr = None
        after_arr = None
        project_root = Path(__file__).parent.parent
        if request.before_image_path:
            p = project_root / request.before_image_path
            if p.exists():
                from PIL import Image
                before_arr = np.array(Image.open(str(p)).convert("RGB"))
        if request.after_image_path:
            p = project_root / request.after_image_path
            if p.exists():
                from PIL import Image
                after_arr = np.array(Image.open(str(p)).convert("RGB"))

        detector = get_unified_detector()
        results = detector.analyze_changes(
            bbox=request.bbox,
            before_date=request.before_date,
            after_date=request.after_date,
            before_rgb=before_arr,
            after_rgb=after_arr,
            pixel_resolution=request.pixel_resolution or 10.0,
        )

        return {
            "status": "success",
            "method": results["method"],
            "feature": "deforestation",
            "data_source": "Sentinel-2 multi-temporal indices",
            "before": {
                "overlay_image": results["overlays"]["classified"],
            },
            "after": {
                "overlay_image": results["overlays"]["classified"],
            },
            "change": results["change_summary"],
            "categories": results["classified_changes"],
            "overlays": {
                "deforestation": results["overlays"]["classified"],
                "heatmap": results["overlays"]["change_probability"],
                "spotlight": results["overlays"]["trend"],
            },
            "trend_summary": results["trend_summary"],
            "years_used": results["years_used"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deforestation detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Deforestation detection failed: {str(e)}"
        )



# ---- Animation Frames Endpoint ----

class AnimationFramesRequest(BaseModel):
    bbox: Dict[str, float]
    before_date: str
    after_date: str


@app.post("/api/ai/animation-frames")
def get_animation_frames(
    request: AnimationFramesRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Generate per-year animation frames with spectral metrics.
    Returns ordered satellite tile images + NDVI/NDBI/vegetation/urban stats.
    """
    try:
        detector = get_unified_detector()
        fetcher = get_tile_fetcher()

        results = detector.generate_animation_frames(
            bbox=request.bbox,
            before_date=request.before_date,
            after_date=request.after_date,
            tile_fetcher=fetcher,
            db=db,
        )
        return results

    except Exception as e:
        logger.error(f"Animation frames failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---- Unified Multi-Temporal Change Detection ----

class AnalyzeChangesRequest(BaseModel):
    bbox: Dict[str, float]                     # {west, south, east, north}
    before_date: str                            # YYYY-MM-DD
    after_date: str                             # YYYY-MM-DD
    before_image_path: Optional[str] = None     # cached true-color tile
    after_image_path: Optional[str] = None
    detect_types: Optional[List[str]] = None    # subset of categories
    pixel_resolution: Optional[float] = 10.0


@app.post("/api/ai/analyze-changes")
def analyze_changes_ml(
    request: AnalyzeChangesRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
        Full unified multi-temporal change detection pipeline.

    Categories:
      - Deforestation (forest → built-up)
      - New Construction
      - Water Body Changes
      - Agricultural Changes
      - Vegetation Recovery
    """
    print(f"\n{'='*60}", flush=True)
    print(f"\U0001f9e0  ML CHANGE DETECTION — FULL PIPELINE", flush=True)
    print(f"User: {current_user.username}", flush=True)
    print(f"Dates: {request.before_date} \u2192 {request.after_date}", flush=True)
    print(f"Detect: {request.detect_types or 'ALL'}", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:
        project_root = Path(__file__).parent.parent
        fetcher = get_tile_fetcher()

        # Load or fetch before/after true-color images
        if request.before_image_path:
            before_path = project_root / request.before_image_path
        else:
            before_path = fetcher.get_tile(
                db, request.bbox, request.before_date, (1024, 1024)
            )
        if request.after_image_path:
            after_path = project_root / request.after_image_path
        else:
            after_path = fetcher.get_tile(
                db, request.bbox, request.after_date, (1024, 1024)
            )

        from PIL import Image
        before_img = Image.open(str(before_path)).convert("RGB")
        after_img = Image.open(str(after_path)).convert("RGB")

        # Run unified temporal pipeline
        detector = get_unified_detector()
        results = detector.analyze_changes(
            bbox=request.bbox,
            before_date=request.before_date,
            after_date=request.after_date,
            before_rgb=np.array(before_img),
            after_rgb=np.array(after_img),
            pixel_resolution=request.pixel_resolution or 10.0,
        )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML change detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"ML change detection failed: {str(e)}"
        )


# Background task functions

async def run_analysis_pipeline(task_id: str, request: AnalysisRequest):
    """
    Run streaming analysis pipeline - fetches and processes on-the-fly
    No disk storage of raw satellite images (saves ~6 GB!)
    """
    import subprocess
    import torch
    
    try:
        task_status[task_id]["status"] = "running"
        task_status[task_id]["progress"] = 10
        task_status[task_id]["current_step"] = "initializing"
        
        city = request.city.lower()
        
        logger.info(f"Starting streaming analysis for {city}")
        logger.info(f"Mode: Stream tiles directly from Copernicus API (no download)")
        task_status[task_id]["progress"] = 20
        
        # Determine device (GPU or CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Run streaming analysis script (single step!)
        task_status[task_id]["current_step"] = "streaming_and_analyzing"
        task_status[task_id]["progress"] = 30
        
        cmd = [
            "python", "scripts/stream_and_analyze.py",
            "--city", city,
            "--before", request.before_date,
            "--after", request.after_date,
            "--grid-size", "4",  # 4x4 = 16 tiles
            "--device", device
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Streaming analysis failed: {result.stderr}")
        
        # Update progress as tiles are processed
        # (In production, could parse stdout for real-time progress)
        task_status[task_id]["progress"] = 90
        
        # Complete
        task_status[task_id]["status"] = "completed"
        task_status[task_id]["progress"] = 100
        task_status[task_id]["current_step"] = "done"
        task_status[task_id]["completed_at"] = datetime.now().isoformat()
        task_status[task_id]["mode"] = "streaming"
        task_status[task_id]["disk_space_saved"] = "~6 GB (raw data not stored)"
        
        logger.info(f"✓ Streaming analysis complete for {city}")
        logger.info(f"✓ Disk space saved: ~6 GB")
        logger.info(result.stdout)
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        task_status[task_id]["status"] = "failed"
        task_status[task_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    logger.info(f"Documentation available at http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )

