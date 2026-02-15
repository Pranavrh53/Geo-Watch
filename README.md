# 🛰️ AI-Based Satellite Change Detection System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

An AI-powered geospatial application that detects land cover changes using satellite imagery. Detects deforestation, new construction, roads, urban expansion, and water bodies drying up.

## 🎯 Features

- 🌳 **Deforestation Detection** - Identify areas where vegetation has been lost
- 🏗️ **Construction Monitoring** - Detect new buildings and urban development
- 🛣️ **Road Detection** - Track new road construction
- 💧 **Water Body Changes** - Monitor lakes and rivers drying up
- 📊 **Area Calculations** - Precise measurements in hectares, acres, and square kilometers
- 🗺️ **Interactive Map** - Visual representation using Leaflet.js
- 📄 **Reports** - Generate detailed PDF reports
- 🌊 **Streaming Mode** - Fetch and process images on-the-fly (saves 90% disk space!)

## 🌊 Streaming Mode (New!)

**Fetch satellite images directly from API without downloading!**

- ✅ **Saves 90% disk space** (~600 MB instead of 6 GB)
- ✅ **No cleanup needed** - raw files never stored
- ✅ **Always fresh data** - fetches latest imagery
- ⚡ **Easy setup** - just need free Copernicus account

See [STREAMING_MODE_GUIDE.md](STREAMING_MODE_GUIDE.md) for full details.

**Quick example:**
```bash
# Get free account at https://dataspace.copernicus.eu
# Add credentials to .env file
# Run streaming analysis
python scripts/stream_and_analyze.py --city bangalore --before 2020-02-01 --after 2024-02-01
```

## 🏗️ Architecture

```
User → Frontend (Leaflet.js)
     → Backend (FastAPI)
     → Data Download (Sentinel-2)
     → Preprocessing (Rasterio)
     → ML Segmentation (SegFormer/DeepLabV3)
     → Change Detection
     → Statistics & Visualization
```

## 📁 Project Structure

```
geo-watch/
├── backend/
│   └── main.py                 # FastAPI application
├── frontend/
│   ├── index.html              # Main UI
│   └── app.js                  # Frontend logic
├── scripts/
│   ├── download_sentinel2.py   # Download satellite data
│   ├── preprocess.py           # Image preprocessing
│   ├── run_segmentation.py     # ML segmentation
│   └── detect_changes.py       # Change detection engine
├── data/
│   ├── raw/                    # Raw satellite imagery
│   ├── processed/              # Processed tiles
│   └── results/                # Analysis results
├── models/                     # ML model weights
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
└── SETUP_GUIDE.md             # Detailed setup instructions
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- 50GB+ free disk space
- Optional: NVIDIA GPU with CUDA support

### 2. Installation

```bash
# Clone or navigate to project directory
cd geo-watch

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your Copernicus credentials
# Get credentials from: https://dataspace.copernicus.eu
```

### 4. Download Data (Manual)

**For Bangalore (Example):**

1. Go to [Copernicus Browser](https://dataspace.copernicus.eu/browser/)
2. Draw rectangle around Bangalore:
   - North: 13.1730°N
   - South: 12.7340°N
   - East: 77.8800°E
   - West: 77.3700°W
3. Filter:
   - Satellite: Sentinel-2
   - Product Type: S2MSI2A (Level-2A)
   - Cloud Cover: < 15%
   - Date: Select two time periods (e.g., Feb 2020 and Feb 2024)
4. Download bands: B02, B03, B04, B08, B11
5. Place in: `data/raw/bangalore/before_2020-02-01/` and `data/raw/bangalore/after_2024-02-01/`

### 5. Run Analysis Pipeline

```bash
# Step 1: Preprocess data
python scripts/preprocess.py --city bangalore

# Step 2: Run segmentation
python scripts/run_segmentation.py --city bangalore

# Step 3: Detect changes
python scripts/detect_changes.py --city bangalore
```

### 6. Start Application

```bash
# Terminal 1: Start Backend
python backend/main.py

# Terminal 2: Start Frontend (simple HTTP server)
cd frontend
python -m http.server 3000

# Open browser: http://localhost:3000
```

## 📊 Data Sources

### Primary Satellite Data
- **Sentinel-2 L2A**: European Space Agency
  - URL: https://dataspace.copernicus.eu
  - Resolution: 10m (RGB, NIR), 20m (SWIR)
  - Revisit: Every 5 days
  - Cost: **FREE**

### Validation Data (Optional)
- **ISRO Bhuvan**: https://bhuvan.nrsc.gov.in
- **Forest Survey India**: https://fsi.nic.in
- **OpenStreetMap**: https://www.openstreetmap.org

## 🧠 ML Models

### 1. SegFormer (Primary)
- **Model**: nvidia/segformer-b0-finetuned-ade-512-512
- **Source**: Hugging Face
- **Use**: General land cover segmentation
- **Size**: ~14MB

### 2. DeepLabV3 (Urban Detection)
- **Model**: deeplabv3_resnet101
- **Source**: PyTorch TorchVision
- **Use**: Enhanced urban area detection
- **Size**: ~233MB

### 3. UNet (Roads - Optional)
- **Dataset**: SpaceNet
- **Use**: Specialized road detection

## 📈 Usage Examples

### Analyze a City

```bash
# Full pipeline for Bangalore
python scripts/preprocess.py --city bangalore
python scripts/run_segmentation.py --city bangalore --period both
python scripts/detect_changes.py --city bangalore
```

### Using the API

```bash
# Get available cities
curl http://localhost:8000/api/cities

# Start analysis
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "city": "bangalore",
    "before_date": "2020-02-01",
    "after_date": "2024-02-01"
  }'

# Get results
curl http://localhost:8000/api/results/bangalore
```

## 🎨 Understanding Results

### Color Coding
- 🔴 **Red**: Deforestation (vegetation lost)
- 🔵 **Blue**: New construction (urban expansion)
- 🟡 **Yellow**: New roads
- 🟣 **Purple**: Water bodies shrinking
- 🟢 **Green**: Vegetation increase

### Land Cover Classes
- **Class 0**: Background/No data
- **Class 1**: Urban/Built-up
- **Class 2**: Vegetation/Forest
- **Class 3**: Water
- **Class 4**: Bare soil
- **Class 5**: Roads

## 🔧 Configuration

Key settings in `config.py`:

```python
# Processing
TILE_SIZE = 512          # Tile size for ML models
BATCH_SIZE = 4           # Batch size for processing
USE_GPU = True           # Enable GPU acceleration

# Sentinel-2
MAX_CLOUD_COVER = 15     # Maximum cloud cover percentage
TARGET_RESOLUTION = 10   # Target resolution in meters

# Area calculations
PIXEL_AREA = 100         # Square meters (10m x 10m)
```

## 🐛 Troubleshooting

### Cannot download Sentinel-2 data
- **Fix**: Check Copernicus credentials in `.env`
- Try manual download via browser first

### Out of memory during segmentation
- **Fix**: Reduce `TILE_SIZE` to 256 in `config.py`
- Process fewer tiles at once
- Use CPU instead of GPU

### No changes detected
- **Fix**: Ensure sufficient time gap between dates
- Verify cloud cover < 15%
- Check that locations match for both dates

## 📚 Documentation

- [Setup Guide](SETUP_GUIDE.md) - Detailed setup instructions
- [API Documentation](http://localhost:8000/docs) - FastAPI interactive docs
- [Sentinel-2 Guide](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi)

## 🛠️ Development

### Adding New Cities

Edit `config.py`:

```python
CITIES = {
    'your_city': {
        'name': 'Your City',
        'country': 'Country',
        'bbox': {
            'north': 0.0,
            'south': 0.0,
            'east': 0.0,
            'west': 0.0
        },
        'center': {
            'lat': 0.0,
            'lon': 0.0
        }
    }
}
```

### Running Tests

```bash
# Test single tile segmentation
python scripts/test_segmentation.py --tile data/processed/bangalore/before_2020-02-01/tiles/tile_0_0.npy
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Model Fine-tuning** - Train on Indian/region-specific datasets
2. **Time Series** - Analyze multiple dates
3. **Automated Alerts** - Email notifications for changes
4. **GIS Export** - Support for GeoJSON, Shapefiles
5. **Mobile App** - React Native frontend

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **ESA** - Sentinel-2 satellite data
- **ISRO** - Bhuvan validation data
- **Hugging Face** - SegFormer model
- **PyTorch** - Deep learning framework
- **Leaflet** - Interactive mapping

## 📧 Support

For questions or issues:
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. Review logs in `logs/` directory
3. Open an issue on GitHub

## 🚀 Roadmap

- [x] Basic change detection
- [x] Web interface
- [ ] PDF report generation
- [ ] Multi-temporal analysis
- [ ] Automated data download
- [ ] Docker deployment
- [ ] Cloud deployment (AWS/Azure)

---

**Built with ❤️ for environmental monitoring and urban planning**
