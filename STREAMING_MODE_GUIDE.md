# 🌊 Streaming Mode - Setup Guide

## ✅ What Changed?

The system now **fetches satellite images directly from the API** and processes on-the-fly **without downloading raw files**.

### Before (Download Mode):
```
Download → Store 6 GB → Process → Analyze → Results
❌ Uses 6 GB disk space
```

### After (Streaming Mode): 
```
Fetch tile → Process → Discard → Fetch next → Process → Results
✅ Uses only 600 MB disk space (90% less!)
```

---

## 📊 Benefits

| Feature | Download Mode | ✨ Streaming Mode |
|---------|--------------|------------------|
| **Disk Space** | ~6 GB | ~600 MB (10x less!) |
| **Processing Time** | 20-25 min | 25-30 min (slightly slower) |
| **Internet** | Only for download | During entire process |
| **Reprocess** | Fast (local files) | Need to refetch |
| **Best For** | Multiple analyses | One-time analysis |

---

## 🔑 API Credentials (Free!)

### Option 1: Copernicus Data Space (Recommended - Free Forever)

**What you need:**
- Free account at https://dataspace.copernicus.eu
- Username and password (that's it!)

**Steps:**
1. Go to https://dataspace.copernicus.eu
2. Click **"Register"**
3. Fill out form (takes 2 minutes)
4. Confirm email
5. Login and note your **username** and **password**

**Add to `.env` file:**
```env
COPERNICUS_USERNAME=your_username_here
COPERNICUS_PASSWORD=your_password_here
```

**Free tier:**
- ✅ Unlimited requests
- ✅ No credit card required
- ✅ Access to all Sentinel-2 data
- ✅ No quotas or restrictions

---

### Option 2: Sentinel Hub (More features, has limits)

**What you need:**
- Free tier account at https://www.sentinel-hub.com/
- OAuth2 client ID and secret

**Steps:**
1. Go to https://www.sentinel-hub.com/
2. Click **"Try for free"**
3. Create account
4. Go to Dashboard → User Settings
5. Create new **OAuth Client**
6. Copy **Client ID** and **Client Secret**

**Add to `.env` file:**
```env
SENTINEL_HUB_CLIENT_ID=your_client_id_here
SENTINEL_HUB_CLIENT_SECRET=your_client_secret_here
```

**Free tier limits:**
- ✅ 1,000 requests/month
- ✅ Good for occasional use
- 💳 Pay-as-you-go after free tier

---

## 🚀 How to Use

### 1. Get Credentials (5 minutes)

Follow Option 1 or Option 2 above.

### 2. Update .env File

Edit `e:\geo-watch\.env`:

```env
# Copernicus Data Space (free, unlimited)
COPERNICUS_USERNAME=your_copernicus_username
COPERNICUS_PASSWORD=your_copernicus_password

# OR Sentinel Hub (free tier: 1000 requests/month)
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret
```

### 3. Install Package (if not done)

```powershell
pip install sentinelhub>=3.10.0
```

### 4. Run Streaming Analysis

**Command line:**
```powershell
python scripts/stream_and_analyze.py --city bangalore --before 2020-02-01 --after 2024-02-01
```

**Via Web Interface:**
1. Start backend: `python backend/main.py`
2. Open `frontend/index.html`
3. Select city and dates
4. Click "Analyze Changes"
5. Results stream in real-time!

---

## 📊 What Happens During Streaming

```
Step 1: Initialize API connection (5 seconds)
   ↓
Step 2: Fetch tile 1/16 from 2020 (2 seconds)
   ↓
Step 3: Segment tile 1 (30 seconds with GPU)
   ↓
Step 4: Fetch same tile from 2024 (2 seconds)
   ↓
Step 5: Segment tile 1 again (30 seconds)
   ↓
Step 6: Compare & detect changes (1 second)
   ↓
Step 7: Save only results (1 KB, not the raw 200 MB!)
   ↓
Step 8: Discard raw tiles, free memory
   ↓
Repeat for tiles 2-16...
   ↓
Done! Total: ~25 minutes, used <1 GB disk
```

---

## 💾 Disk Space Comparison

### Download Mode (Old):
```
data/
├── raw/
│   ├── bangalore/
│   │   ├── before_2020-02-01/
│   │   │   ├── B02.jp2        (500 MB)
│   │   │   ├── B03.jp2        (500 MB)
│   │   │   ├── B04.jp2        (500 MB)
│   │   │   ├── B08.jp2        (500 MB)
│   │   │   └── B11.jp2        (500 MB)
│   │   └── after_2024-02-01/
│   │       └── ...            (2.5 GB)
│   └── TOTAL RAW: 6 GB ❌
├── processed/
│   └── ...                     (500 MB)
└── results/
    └── ...                     (100 MB)

TOTAL: ~6.6 GB
```

### Streaming Mode (New):
```
data/
├── processed/          (only final processed tiles)
│   └── bangalore/
│       └── analysis_2020-02-01_to_2024-02-01/
│           ├── changes_summary.json     (2 KB)
│           ├── change_map.png           (500 KB)
│           └── temporary_cache/         (auto-cleared)
└── results/
    └── ...                               (100 MB)

TOTAL: ~600 MB ✅ (10x less!)
```

---

## 🛠️ Advanced Options

### Adjust Grid Size (Speed vs Accuracy)

**Small grid (faster, less detail):**
```powershell
python scripts/stream_and_analyze.py --grid-size 2  # 2x2 = 4 tiles only
# Time: ~8 minutes
# Resolution: Lower
```

**Large grid (slower, more detail):**
```powershell
python scripts/stream_and_analyze.py --grid-size 8  # 8x8 = 64 tiles
# Time: ~2 hours
# Resolution: Higher
```

**Default (balanced):**
```powershell
python scripts/stream_and_analyze.py --grid-size 4  # 4x4 = 16 tiles
# Time: ~25 minutes
# Resolution: Good
```

### Use CPU if No GPU

```powershell
python scripts/stream_and_analyze.py --device cpu
```

Will take longer (~1.5 hours) but works without CUDA.

---

## 🐛 Troubleshooting

### "No credentials found"

**Fix:** Make sure `.env` file has credentials:
```env
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password
```

### "Authentication failed"

**Fix:** 
1. Check credentials are correct
2. Try logging into https://dataspace.copernicus.eu manually
3. Make sure account is activated (check email)

### "Tile fetch timeout"

**Fix:**
1. Check internet connection
2. Copernicus API might be slow, retry later
3. Reduce grid size: `--grid-size 2`

### "Out of memory"

**Fix:**
1. Reduce grid size: `--grid-size 2`
2. Use CPU instead: `--device cpu`
3. Close other programs

---

## 📝 Example Output

```
Streaming & analyzing tiles: 100%|████████| 16/16 [25:30<00:00, 95.6s/tile]

✓ Analysis complete! Results saved to data/processed/bangalore/analysis_2020-02-01_to_2024-02-01

Total changes detected:
  - Deforestation: 450.25 hectares
  - New construction: 230.45 hectares
  - Water loss: 15.80 hectares

============================================================
STREAMING ANALYSIS COMPLETE - BANGALORE
============================================================
Deforestation:         450.25 hectares
New Construction:      230.45 hectares
Water Loss:             15.80 hectares
Urbanization:          340.60 hectares
Tiles Processed:            16
============================================================

💾 Disk space saved: ~6 GB (raw data not stored!)
```

---

## 🎯 Recommended Workflow

### For One-Time Analysis:
✅ Use Streaming Mode (this guide)
- Less disk space
- No cleanup needed
- Fetch fresh data every time

### For Research / Multiple Runs:
❌ Use Download Mode (old scripts)
- Download once, analyze multiple times
- Faster for repeated analysis
- More disk space needed

---

## 🔄 Switching Between Modes

**Use Streaming (default now):**
```powershell
python scripts/stream_and_analyze.py --city bangalore
```

**Use Download Mode (old way):**
```powershell
python scripts/download_sentinel2.py --city bangalore
python scripts/preprocess.py --city bangalore
python scripts/run_segmentation.py --city bangalore
python scripts/detect_changes.py --city bangalore
```

Both produce the same results, just different disk usage!

---

## ✅ Quick Start Checklist

- [ ] Created Copernicus account (5 min)
- [ ] Added credentials to `.env`
- [ ] Installed sentinelhub: `pip install sentinelhub`
- [ ] Tested streaming: `python scripts/stream_and_analyze.py --city bangalore`
- [ ] Checked results in `data/processed/bangalore/`
- [ ] Verified disk usage is ~600 MB instead of 6 GB
- [ ] Celebrated saving 90% disk space! 🎉

---

## 🌐 API Status

Check API availability:
- **Copernicus:** https://dataspace.copernicus.eu/status
- **Sentinel Hub:** https://status.sentinel-hub.com/

---

**Questions?** Check the main [README.md](README.md) or [DATASET_GUIDE.md](DATASET_GUIDE.md)
