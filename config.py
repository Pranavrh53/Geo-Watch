"""Configuration settings for the Satellite Change Detection System"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', './models'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', './logs'))

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'raw').mkdir(exist_ok=True)
    (directory / 'processed').mkdir(exist_ok=True)

DATA_DIR_RAW = DATA_DIR / 'raw'
DATA_DIR_PROCESSED = DATA_DIR / 'processed'
DATA_DIR_RESULTS = DATA_DIR / 'results'
DATA_DIR_RESULTS.mkdir(exist_ok=True)

# Copernicus Credentials
COPERNICUS_USERNAME = os.getenv('COPERNICUS_USERNAME')
COPERNICUS_PASSWORD = os.getenv('COPERNICUS_PASSWORD')
SENTINEL_HUB_INSTANCE_ID = os.getenv('SENTINEL_HUB_INSTANCE_ID')

# Sentinel-2 Configuration
SENTINEL_BANDS = ['B02', 'B03', 'B04', 'B08', 'B11']  # Blue, Green, Red, NIR, SWIR
SENTINEL_PLATFORM = 'Sentinel-2'
SENTINEL_PRODUCT_TYPE = 'S2MSI2A'  # Level-2A (atmospherically corrected)
MAX_CLOUD_COVER = int(os.getenv('MAX_CLOUD_COVER', 15))

# Processing Configuration
TILE_SIZE = int(os.getenv('TILE_SIZE', 512))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))
USE_GPU = os.getenv('USE_GPU', 'true').lower() == 'true'

# Sentinel-2 Band Resolutions (meters)
BAND_RESOLUTIONS = {
    'B02': 10,  # Blue
    'B03': 10,  # Green
    'B04': 10,  # Red
    'B08': 10,  # NIR
    'B11': 20,  # SWIR
}

# Target resolution for processing (meters)
TARGET_RESOLUTION = 10

# Pixel area in square meters
PIXEL_AREA = TARGET_RESOLUTION * TARGET_RESOLUTION  # 100 sq meters

# Conversion factors
SQM_TO_HECTARES = 0.0001
SQM_TO_ACRES = 0.000247105
SQM_TO_SQKM = 0.000001

# Land Cover Classes
LAND_COVER_CLASSES = {
    0: 'background',
    1: 'urban_built',
    2: 'vegetation',
    3: 'water',
    4: 'bare_soil',
    5: 'road',
}

# Change Detection Classes
CHANGE_CLASSES = {
    'deforestation': {
        'from': [2],  # vegetation
        'to': [1, 4, 5],  # urban, soil, road
        'color': [255, 0, 0],  # Red
        'name': 'Deforestation'
    },
    'construction': {
        'from': [2, 4],  # vegetation, soil
        'to': [1],  # urban
        'color': [0, 0, 255],  # Blue
        'name': 'New Construction'
    },
    'new_roads': {
        'from': [2, 4],  # vegetation, soil
        'to': [5],  # road
        'color': [255, 255, 0],  # Yellow
        'name': 'New Roads'
    },
    'water_loss': {
        'from': [3],  # water
        'to': [1, 2, 4],  # urban, vegetation, soil
        'color': [128, 0, 128],  # Purple
        'name': 'Water Bodies Drying'
    },
    'vegetation_gain': {
        'from': [4],  # soil
        'to': [2],  # vegetation
        'color': [0, 255, 0],  # Green
        'name': 'Vegetation Increase'
    }
}

# Model Configuration
MODELS = {
    'segformer': {
        'name': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'input_size': 512,
        'use_for': ['vegetation', 'water', 'general']
    },
    'deeplabv3': {
        'name': 'deeplabv3_resnet101',
        'input_size': 512,
        'use_for': ['urban', 'construction']
    }
}

# City Coordinates (Bounding Boxes)
CITIES = {
    'bangalore': {
        'name': 'Bangalore',
        'country': 'India',
        'bbox': {
            'north': 13.1730,
            'south': 12.7340,
            'east': 77.8800,
            'west': 77.3700
        },
        'center': {
            'lat': 12.9716,
            'lon': 77.5946
        }
    },
    'delhi': {
        'name': 'Delhi',
        'country': 'India',
        'bbox': {
            'north': 28.8833,
            'south': 28.4041,
            'east': 77.3465,
            'west': 76.8389
        },
        'center': {
            'lat': 28.7041,
            'lon': 77.1025
        }
    },
    'mumbai': {
        'name': 'Mumbai',
        'country': 'India',
        'bbox': {
            'north': 19.2695,
            'south': 18.8942,
            'east': 72.9781,
            'west': 72.7757
        },
        'center': {
            'lat': 19.0760,
            'lon': 72.8777
        }
    },
    'hyderabad': {
        'name': 'Hyderabad',
        'country': 'India',
        'bbox': {
            'north': 17.5640,
            'south': 17.2403,
            'east': 78.6530,
            'west': 78.2543
        },
        'center': {
            'lat': 17.3850,
            'lon': 78.4867
        }
    }
}

# API Configuration
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
API_WORKERS = int(os.getenv('API_WORKERS', 4))

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
