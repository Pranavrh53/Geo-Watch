"""
Create project directory structure
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, MODELS_DIR, LOGS_DIR

def create_directories():
    """Create all necessary project directories"""
    
    directories = [
        # Data directories
        DATA_DIR / 'raw',
        DATA_DIR / 'processed',
        DATA_DIR / 'results',
        
        # City directories
        DATA_DIR / 'raw' / 'bangalore',
        DATA_DIR / 'raw' / 'delhi',
        DATA_DIR / 'raw' / 'mumbai',
        DATA_DIR / 'raw' / 'hyderabad',
        
        # Models directory
        MODELS_DIR / 'segformer',
        MODELS_DIR / 'deeplabv3',
        MODELS_DIR / 'weights',
        
        # Logs directory
        LOGS_DIR,
        
        # Backend directory (if needed)
        Path('backend'),
        
        # Frontend directory
        Path('frontend'),
        
        # Scripts directory
        Path('scripts'),
    ]
    
    print("Creating project directory structure...")
    print()
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print()
    print("✅ Directory structure created successfully!")
    print()
    print("Next steps:")
    print("1. Copy .env.example to .env and add your Copernicus credentials")
    print("2. Download Sentinel-2 data for your target city")
    print("3. Run preprocessing: python scripts/preprocess.py --city bangalore")

if __name__ == '__main__':
    create_directories()
