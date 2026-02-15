# 🛰️ AI-Based Satellite Change Detection System - Setup Guide

## 📋 Project Overview
Build a machine learning system to detect:
- 🌳 Deforestation
- 🏗️ New Construction  
- 🛣️ New Roads
- 🏙️ Urban Expansion
- 💧 Water Bodies Drying Up

**Starting Location:** Bangalore, India (12.9716°N, 77.5946°E)

---

## 🎯 STEP-BY-STEP IMPLEMENTATION GUIDE

### PHASE 1: Environment Setup (Day 1)

#### Step 1.1: Install Python and Dependencies
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install core dependencies
pip install -r requirements.txt
```

#### Step 1.2: Create Copernicus Account
1. Go to: https://dataspace.copernicus.eu
2. Click "Register" (top right)
3. Fill in details:
   - Email
   - Username
   - Password
4. Verify email
5. Login and go to "User Settings" → "OData Credentials"
6. Save your credentials (needed for API access)

---

### PHASE 2: Data Acquisition (Day 1-2)

#### Step 2.1: Get Bangalore Coordinates
Bangalore bounding box:
- **North:** 13.1730°N
- **South:** 12.7340°N
- **East:** 77.8800°E
- **West:** 77.3700°E

#### Step 2.2: Download Sentinel-2 Data

**Option A: Manual Download (Recommended for First Time)**
1. Go to: https://dataspace.copernicus.eu/browser/
2. Draw rectangle around Bangalore
3. Set filters:
   - **Satellite:** Sentinel-2
   - **Product Type:** S2MSI2A (Level-2A - atmospherically corrected)
   - **Date Range Before:** Jan 2020 - Mar 2020
   - **Date Range After:** Jan 2024 - Mar 2024
   - **Cloud Cover:** < 15%
4. Select matching scenes
5. Download required bands:
   - B02 (Blue) - 10m resolution
   - B03 (Green) - 10m resolution
   - B04 (Red) - 10m resolution
   - B08 (NIR) - 10m resolution
   - B11 (SWIR) - 20m resolution

**Option B: Automated Download (Use our script)**
```bash
# Configure credentials in .env file
python scripts/download_sentinel2.py --city bangalore --before 2020-02-01 --after 2024-02-01
```

#### Step 2.3: Organize Data Structure
```
data/
├── raw/
│   ├── bangalore/
│   │   ├── before_2020-02-01/
│   │   │   ├── B02.jp2
│   │   │   ├── B03.jp2
│   │   │   ├── B04.jp2
│   │   │   ├── B08.jp2
│   │   │   └── B11.jp2
│   │   └── after_2024-02-01/
│   │       ├── B02.jp2
│   │       ├── B03.jp2
│   │       ├── B04.jp2
│   │       ├── B08.jp2
│   │       └── B11.jp2
├── processed/
└── results/
```

---

### PHASE 3: Data Preprocessing (Day 2-3)

#### Step 3.1: Load and Visualize Raw Data
```bash
python scripts/visualize_bands.py --input data/raw/bangalore/before_2020-02-01
```

#### Step 3.2: Preprocess Images
```bash
# This will:
# - Resample 20m bands to 10m
# - Stack bands into RGB and NIR composites
# - Normalize values
# - Tile into 512x512 patches
# - Save processed data
python scripts/preprocess.py --city bangalore
```

Expected output:
```
data/processed/bangalore/
├── before_tiles/
│   ├── tile_0_0.npy
│   ├── tile_0_1.npy
│   └── ...
└── after_tiles/
    ├── tile_0_0.npy
    └── ...
```

---

### PHASE 4: ML Model Setup (Day 3-4)

#### Step 4.1: Download Pre-trained Models
```bash
# Downloads SegFormer and DeepLabV3
python scripts/download_models.py
```

Models saved to:
```
models/
├── segformer-b0/
├── deeplabv3/
└── weights/
```

#### Step 4.2: Run Segmentation on Test Tile
```bash
# Test on single tile
python scripts/test_segmentation.py --tile data/processed/bangalore/before_tiles/tile_0_0.npy
```

#### Step 4.3: Run Full Segmentation
```bash
# Segment all tiles (both before and after)
python scripts/run_segmentation.py --city bangalore
```

Output:
```
data/processed/bangalore/
├── before_masks/
│   ├── tile_0_0_mask.npy
│   └── ...
└── after_masks/
    ├── tile_0_0_mask.npy
    └── ...
```

---

### PHASE 5: Change Detection (Day 4-5)

#### Step 5.1: Run Change Detection
```bash
python scripts/detect_changes.py --city bangalore
```

This compares before/after masks and generates:
- **Deforestation mask:** Vegetation → Urban/Soil
- **Construction mask:** Soil/Vegetation → Urban
- **Road mask:** Non-road → Road
- **Water loss mask:** Water → Non-water

#### Step 5.2: Calculate Statistics
```bash
python scripts/calculate_stats.py --city bangalore
```

Output example:
```json
{
  "deforestation": {
    "pixels": 125000,
    "area_sqm": 12500000,
    "area_hectares": 1250,
    "area_acres": 3089
  },
  "construction": {
    "pixels": 80000,
    "area_sqm": 8000000,
    "area_hectares": 800,
    "area_acres": 1977
  }
}
```

---

### PHASE 6: Backend API (Day 5-6)

#### Step 6.1: Start FastAPI Server
```bash
python backend/main.py
```

Server runs on: http://localhost:8000

#### Step 6.2: Test API Endpoints
```bash
# Get available cities
curl http://localhost:8000/api/cities

# Trigger analysis for Bangalore
curl -X POST http://localhost:8000/api/analyze -d '{"city": "bangalore", "before": "2020-02-01", "after": "2024-02-01"}'

# Get results
curl http://localhost:8000/api/results/bangalore
```

---

### PHASE 7: Frontend (Day 6-7)

#### Step 7.1: Install Frontend Dependencies
```bash
cd frontend
npm install
```

#### Step 7.2: Start Development Server
```bash
npm run dev
```

Frontend runs on: http://localhost:3000

#### Step 7.3: Use the Application
1. Open browser: http://localhost:3000
2. Map centered on Bangalore
3. Draw polygon or select district
4. Choose before/after dates
5. Click "Analyze"
6. View results:
   - Side-by-side comparison
   - Color-coded change overlay
   - Statistics panel
7. Generate PDF report

---

## 🎨 Understanding the Output

### Color Coding
- 🔴 **Red:** Deforestation (vegetation lost)
- 🔵 **Blue:** New construction (urban expansion)
- 🟡 **Yellow:** New roads
- 🟣 **Purple:** Water bodies shrinking
- 🟢 **Green:** Vegetation increase

### Land Cover Classes
- **Class 0:** Background/No data
- **Class 1:** Urban/Built-up
- **Class 2:** Vegetation/Forest
- **Class 3:** Water
- **Class 4:** Bare soil
- **Class 5:** Roads

---

## 📊 Data Sources Reference

### Primary Data
- **Sentinel-2 L2A:** https://dataspace.copernicus.eu
- Resolution: 10m (RGB, NIR), 20m (SWIR)
- Revisit time: 5 days
- Free and open

### Validation Data (Optional)
- **ISRO Bhuvan:** https://bhuvan.nrsc.gov.in
- **Forest Survey India:** https://fsi.nic.in
- **OpenStreetMap:** https://www.openstreetmap.org

---

## 🧠 ML Models Used

### 1. SegFormer (Primary)
- **Source:** Hugging Face
- **Model:** nvidia/segformer-b0-finetuned-ade-512-512
- **Use:** General land cover segmentation
- **Size:** ~14MB

### 2. DeepLabV3 (Urban Detection)
- **Source:** PyTorch torchvision
- **Model:** deeplabv3_resnet101
- **Use:** Enhanced urban area detection
- **Size:** ~233MB

### 3. UNet (Roads - Optional)
- **Source:** Custom training on SpaceNet
- **Dataset:** https://spacenet.ai/datasets/
- **Use:** Specialized road detection

---

## 🔧 Troubleshooting

### Issue: Cannot download Sentinel-2 data
**Solution:**
- Check Copernicus credentials
- Verify API limits (max 2 concurrent downloads)
- Try manual download first

### Issue: Out of memory during segmentation
**Solution:**
- Reduce tile size in config: `TILE_SIZE = 256`
- Process fewer tiles at once
- Use CPU instead of GPU for small batches

### Issue: No changes detected
**Solution:**
- Check date range (need significant time gap)
- Verify cloud cover < 15%
- Ensure same location/bounds for both dates

---

## 📈 Next Steps After Bangalore

1. **Add more Indian cities:**
   - Delhi
   - Mumbai
   - Hyderabad
   - Chennai

2. **Improve models:**
   - Fine-tune on Indian landscapes
   - Train custom road detector
   - Add building footprint detection

3. **Add features:**
   - Time-series analysis (multiple dates)
   - Automated alerts
   - Export to GIS formats (GeoJSON, Shapefile)

---

## 💾 System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB
- GPU: Optional (makes it 10x faster)

**Recommended:**
- CPU: 8 cores
- RAM: 16GB
- Storage: 200GB
- GPU: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)

---

## 📚 Additional Resources

- **Sentinel-2 User Guide:** https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi
- **SegFormer Paper:** https://arxiv.org/abs/2105.15203
- **DeepLabV3 Paper:** https://arxiv.org/abs/1706.05587
- **Leaflet.js Docs:** https://leafletjs.com/reference.html

---

## 🤝 Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in `logs/` directory
3. Verify data integrity with validation scripts

---

**Ready to start? Begin with Phase 1! 🚀**
