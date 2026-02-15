# 📥 Dataset Acquisition Guide

## Complete Guide to Downloading Sentinel-2 Data for Bangalore

This guide provides detailed, step-by-step instructions for downloading satellite imagery from Copernicus Data Space.

---

## 🎯 What You Need

### Target Area: Bangalore, India

**Coordinates:**
- **North:** 13.1730°N
- **South:** 12.7340°N  
- **East:** 77.8800°E
- **West:** 77.3700°W

**Dates for Analysis:**
- **Before:** February 2020
- **After:** February 2024

**Required Bands:**
- B02 (Blue) - 10m resolution
- B03 (Green) - 10m resolution
- B04 (Red) - 10m resolution
- B08 (Near Infrared) - 10m resolution
- B11 (Short Wave Infrared) - 20m resolution

---

## 📋 Method 1: Manual Download via Browser (Recommended)

### Step 1: Create Account

1. Go to: https://dataspace.copernicus.eu
2. Click **"Register"** (top right)
3. Fill in the registration form:
   - Email address
   - Username
   - Password (min 12 characters)
   - Accept terms
4. Verify your email
5. Log in to your account

### Step 2: Access the Browser

1. Once logged in, click **"Browser"** in the top menu
2. You'll see a map interface

### Step 3: Define Your Area of Interest

**Option A: Draw Rectangle**
1. Click the **polygon drawing tool** in the left toolbar
2. Draw a rectangle covering Bangalore
3. Use these coordinates as guides:
   - Northwest corner: 13.1730°N, 77.3700°E
   - Northeast corner: 13.1730°N, 77.8800°E
   - Southwest corner: 12.7340°N, 77.3700°E
   - Southeast corner: 12.7340°N, 77.8800°E

**Option B: Enter Coordinates**
1. Click the search box
2. Enter: `12.9716, 77.5946` (Bangalore center)
3. Zoom to appropriate level

### Step 4: Set Search Filters

In the left panel:

1. **Satellite Mission:**
   - Select: `Sentinel-2`

2. **Product Type:**
   - Select: `S2MSI2A` (Level-2A - Atmospherically corrected)

3. **Sensing Period (for BEFORE image):**
   - Start: `2020-01-15`
   - End: `2020-03-15`
   - (60-day window to find clear image)

4. **Cloud Cover:**
   - Min: `0%`
   - Max: `15%`

5. **Additional Filters:**
   - Timeliness: All
   - Processing Baseline: All (latest)

### Step 5: Search for Products

1. Click **"Search"** button
2. Wait for results to load
3. You should see several products listed

### Step 6: Select Best Product

Look for products with:
- ✅ Lowest cloud cover percentage
- ✅ Date closest to February 1, 2020
- ✅ Status: ONLINE (can download immediately)

**Product name will look like:**
```
S2B_MSIL2A_20200201T051709_N0214_R019_T43PGN_20200201T091028
```

### Step 7: Download Bands

1. **Click on the product** to open details
2. **Click the download icon** (⬇️)
3. You'll see download options:

**Important:** Sentinel-2 products are LARGE (5-10 GB). We only need specific bands.

**Download Options:**

**Option A: Direct Band Download**
- Look for "Product Components" section
- Download only these files:
  - `IMG_DATA/R10m/B02_10m.jp2` (~80 MB)
  - `IMG_DATA/R10m/B03_10m.jp2` (~80 MB)
  - `IMG_DATA/R10m/B04_10m.jp2` (~80 MB)
  - `IMG_DATA/R10m/B08_10m.jp2` (~80 MB)
  - `IMG_DATA/R20m/B11_20m.jp2` (~40 MB)

**Option B: Download Full Product (if band selection not available)**
- Download the full product
- Extract only the bands we need
- Delete the rest to save space

### Step 8: Repeat for AFTER Image

1. **Change the date filter:**
   - Start: `2024-01-15`
   - End: `2024-03-15`

2. **Search again**

3. **Select best product** (lowest cloud cover, close to Feb 1, 2024)

4. **Download the same bands:**
   - B02, B03, B04, B08, B11

### Step 9: Organize Downloaded Files

Create this directory structure:

```
e:/geo-watch/data/raw/bangalore/
│
├── before_2020-02-01/
│   ├── B02.jp2
│   ├── B03.jp2
│   ├── B04.jp2
│   ├── B08.jp2
│   └── B11.jp2
│
└── after_2024-02-01/
    ├── B02.jp2
    ├── B03.jp2
    ├── B04.jp2
    ├── B08.jp2
    └── B11.jp2
```

**Steps:**
1. Create folders: `before_2020-02-01` and `after_2024-02-01`
2. Rename downloaded files to simple names: `B02.jp2`, `B03.jp2`, etc.
3. Place in appropriate folder

---

## 📋 Method 2: Using Sentinel Hub (Alternative)

### Sentinel Hub Browser

1. Go to: https://apps.sentinel-hub.com/eo-browser/
2. Free to use with registration
3. More user-friendly interface
4. Can visualize before downloading

### Steps:

1. **Register/Login**
2. **Select location:** Search for "Bangalore"
3. **Select date:** Use calendar to pick Feb 2020
4. **View imagery:** Check cloud cover visually
5. **Download:**
   - Click "Download" button
   - Select "Analytical" download
   - Choose bands: B02, B03, B04, B08, B11
   - Download

---

## 📋 Method 3: API Download (Automated)

### Using SentinelSat Python Library

**Setup:**
```bash
pip install sentinelsat
```

**Script:**
```python
from sentinelsat import SentinelAPI

# Connect to API
api = SentinelAPI('your_username', 'your_password', 
                  'https://catalogue.dataspace.copernicus.eu')

# Bangalore bounding box
bbox = (77.37, 12.734, 77.88, 13.173)  # West, South, East, North

# Search for products
products = api.query(
    area=bbox,
    date=('20200115', '20200315'),
    platformname='Sentinel-2',
    producttype='S2MSI2A',
    cloudcoverpercentage=(0, 15)
)

# Download
for product_id, product_info in products.items():
    api.download(product_id)
```

**Note:** This downloads full products. You'll still need to extract individual bands.

---

## ✅ Verification Checklist

After downloading, verify you have:

- [ ] Two time periods (before and after)
- [ ] Five bands each (B02, B03, B04, B08, B11)
- [ ] Files are in `.jp2` or `.tif` format
- [ ] Files are in correct directories
- [ ] Each file is 40-100 MB (full resolution)

**Check file sizes:**
```powershell
# Windows
dir /s e:\geo-watch\data\raw\bangalore\
```

**Expected:**
```
before_2020-02-01/
├── B02.jp2  (~80 MB)
├── B03.jp2  (~80 MB)
├── B04.jp2  (~80 MB)
├── B08.jp2  (~80 MB)
└── B11.jp2  (~40 MB)
```

---

## 🎨 Visualizing Your Data

Before processing, you can view your downloaded data:

### Using QGIS (Free GIS Software)

1. **Download QGIS:** https://qgis.org/
2. **Open QGIS**
3. **Add Raster Layer:** Layer → Add Layer → Add Raster Layer
4. **Select your B04.jp2** (Red band)
5. **Repeat for B03 and B02**
6. **Create RGB composite**

### Using Python (Quick View)

```python
import rasterio
import matplotlib.pyplot as plt

# Load bands
with rasterio.open('data/raw/bangalore/before_2020-02-01/B04.jp2') as src:
    red = src.read(1)

with rasterio.open('data/raw/bangalore/before_2020-02-01/B03.jp2') as src:
    green = src.read(1)

with rasterio.open('data/raw/bangalore/before_2020-02-01/B02.jp2') as src:
    blue = src.read(1)

# Create RGB
rgb = np.stack([red, green, blue], axis=-1)
rgb = np.clip(rgb / 3000 * 255, 0, 255).astype(np.uint8)

# Display
plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.title('Bangalore - Before (RGB)')
plt.axis('off')
plt.show()
```

---

## 🐛 Troubleshooting

### Problem: Cannot find download option for individual bands

**Solution:** 
- Download full product
- Extract using 7-Zip or similar
- Navigate to `GRANULE/.../IMG_DATA/`
- Copy only needed bands

### Problem: Download is very slow

**Solution:**
- Copernicus servers can be slow during peak hours
- Try downloading during off-peak (evening/night in Europe)
- Use a download manager for resume capability

### Problem: File format is wrong (.SAFE folder instead of .jp2)

**Solution:**
- `.SAFE` is the container format
- Inside the folder, navigate to:
  - `GRANULE/[product]/IMG_DATA/R10m/` for 10m bands
  - `GRANULE/[product]/IMG_DATA/R20m/` for 20m bands
- Copy the `.jp2` files from there

### Problem: Cloud cover too high in all available images

**Solution:**
- Expand date range (Jan-Mar instead of just Feb)
- Consider using composite of multiple images
- Try different months (avoid monsoon season in India: June-September)

---

## 📊 Data Specifications

### Sentinel-2 Product Specifications

| Parameter | Value |
|-----------|-------|
| Satellite | Sentinel-2A or Sentinel-2B |
| Product Level | Level-2A (Surface Reflectance) |
| Tile Size | 100 km × 100 km |
| Revisit Time | 5 days (both satellites) |
| Bands | 13 multispectral bands |
| Resolution | 10m, 20m, 60m depending on band |
| Format | JPEG2000 (.jp2) |

### Bands We Use

| Band | Name | Wavelength (nm) | Resolution | Use |
|------|------|----------------|------------|-----|
| B02 | Blue | 490 | 10m | True color, water |
| B03 | Green | 560 | 10m | True color, vegetation |
| B04 | Red | 665 | 10m | True color, vegetation |
| B08 | NIR | 842 | 10m | Vegetation, water |
| B11 | SWIR | 1610 | 20m | Built-up, soil moisture |

---

## 🌍 Data for Other Cities

### Delhi
- Bounding Box: 28.4041°N to 28.8833°N, 76.8389°E to 77.3465°E
- Tiles: T43RGN, T43RGP, T43RHN

### Mumbai
- Bounding Box: 18.8942°N to 19.2695°N, 72.7757°E to 72.9781°E
- Tiles: T43QFE, T43QFG

### Hyderabad
- Bounding Box: 17.2403°N to 17.5640°N, 78.2543°E to 78.6530°E
- Tiles: T44PLR, T44PLS

---

## 📚 Additional Resources

- **Copernicus User Guide:** https://documentation.dataspace.copernicus.eu/
- **Sentinel-2 Product Spec:** https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/product-types
- **Sentinel-2 Band Info:** https://gisgeography.com/sentinel-2-bands-combinations/
- **Download Tutorial (Video):** https://www.youtube.com/results?search_query=copernicus+sentinel-2+download

---

## ⏱️ Time Estimate

- **Account creation:** 5 minutes
- **Finding suitable image:** 10-15 minutes
- **Downloading (per date):** 20-30 minutes (depends on connection)
- **Total for Bangalore:** ~1 hour

---

**Once you have the data downloaded and organized, proceed to preprocessing!**

```bash
python scripts/preprocess.py --city bangalore
```

Good luck! 🚀
