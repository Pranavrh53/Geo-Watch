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
import sys
import json
import logging
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
from backend.image_processor import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Satellite Change Detection API",
    description="AI-based geospatial change detection system",
    version="1.0.0"
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


# In-memory task status storage (use Redis in production)
task_status = {}

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    logger.info("✓ Database initialized")


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


@app.post("/api/tile/fetch")
async def fetch_tile(
    request: TileFetchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Fetch satellite tile for custom region
    Returns URL to cached image with quality metrics
    """
    try:
        fetcher = get_tile_fetcher()
        processor = ImageProcessor()
        
        # Validate bbox
        bbox = request.bbox
        if not all(k in bbox for k in ['west', 'south', 'east', 'north']):
            raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        # Get tile (from cache or fetch new)
        size = (request.size, request.size)
        
        tile_path = fetcher.get_tile(db, bbox, request.date, size)
        
        # Check image quality
        from PIL import Image
        img = Image.open(tile_path)
        quality_check = processor.check_image_quality(img)
        
        # Return file path as URL with quality info
        return {
            "status": "success",
            "image_url": f"/api/tile/image/{tile_path.name}",
            "cached": True,
            "date": request.date,
            "bbox": bbox,
            "source": "sentinel",
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
    AI-powered change detection between two satellite images
    Returns detailed analysis with change metrics and overlay
    """
    try:
        fetcher = get_tile_fetcher()
        processor = ImageProcessor()
        
        logger.info(f"Starting change detection: {request.before_date} → {request.after_date}")
        
        # Step 1: Fetch both images
        before_path = fetcher.get_tile(db, request.bbox, request.before_date, (1024, 1024))
        after_path = fetcher.get_tile(db, request.bbox, request.after_date, (1024, 1024))
        
        # Step 2: Load and check quality
        from PIL import Image
        before_img = Image.open(before_path)
        after_img = Image.open(after_path)
        
        before_quality = processor.check_image_quality(before_img)
        after_quality = processor.check_image_quality(after_img)
        
        logger.info(f"Before image quality: {before_quality['quality']}, Blur score: {before_quality.get('blur_score', 0):.2f}")
        logger.info(f"After image quality: {after_quality['quality']}, Blur score: {after_quality.get('blur_score', 0):.2f}")
        
        if not before_quality['is_valid'] or not after_quality['is_valid']:
            return {
                "status": "error",
                "message": "One or both images are blank/invalid",
                "before_quality": before_quality,
                "after_quality": after_quality,
                "suggestion": "Try different dates or a different region"
            }
        
        # Step 3: Enhance images if requested
        if request.enhance_images:
            logger.info("Enhancing images for better analysis...")
            before_img = processor.enhance_image(before_img, sharpen=True, denoise=True)
            after_img = processor.enhance_image(after_img, sharpen=True, denoise=True)
        
        # Step 4: Detect changes
        logger.info("Running change detection algorithm...")
        changes = processor.detect_changes(before_img, after_img, request.sensitivity)
        
        # Step 5: Save overlay image
        overlay_dir = Path(__file__).parent.parent / "data" / "overlays"
        overlay_dir.mkdir(exist_ok=True)
        
        overlay_filename = f"overlay_{request.before_date}_{request.after_date}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        overlay_path = overlay_dir / overlay_filename
        changes['overlay_image'].save(overlay_path)
        
        mask_filename = f"mask_{request.before_date}_{request.after_date}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        mask_path = overlay_dir / mask_filename
        changes['change_mask'].save(mask_path)
        
        # Step 6: Calculate real-world area (rough estimate based on bbox size)
        bbox = request.bbox
        lat_diff = abs(bbox['north'] - bbox['south'])
        lon_diff = abs(bbox['east'] - bbox['west'])
        
        # Approximate area in km² (1 degree ≈ 111 km at equator)
        area_km2 = lat_diff * lon_diff * 111 * 111
        area_hectares = area_km2 * 100
        actual_change_hectares = (changes['change_percentage'] / 100) * area_hectares
        
        # Return results
        return {
            "status": "success",
            "before_date": request.before_date,
            "after_date": request.after_date,
            "change_detected": changes['change_percentage'] > 1.0,
            "change_percentage": changes['change_percentage'],
            "change_area_hectares": round(actual_change_hectares, 4),
            "change_area_km2": round(actual_change_hectares / 100, 4),
            "severity": changes['severity'],
            "change_type": changes.get('change_type', 'Unknown'),
            "confidence": changes.get('confidence', 'Low'),
            "overlay_url": f"/api/overlay/image/{overlay_filename}",
            "mask_url": f"/api/overlay/image/{mask_filename}",
            "image_quality": {
                "before": {
                    "quality": before_quality['quality'],
                    "blur_score": round(before_quality.get('blur_score', 0), 2),
                    "is_blurry": before_quality.get('is_blurry', True)
                },
                "after": {
                    "quality": after_quality['quality'],
                    "blur_score": round(after_quality.get('blur_score', 0), 2),
                    "is_blurry": after_quality.get('is_blurry', True)
                }
            },
            "analysis_details": {
                "total_pixels": changes['total_pixels'],
                "changed_pixels": changes['changed_pixels'],
                "sensitivity_threshold": request.sensitivity,
                "images_enhanced": request.enhance_images
            }
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
