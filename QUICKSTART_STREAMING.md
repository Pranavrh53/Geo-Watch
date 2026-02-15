# 🚀 Quick Start - Streaming Mode

## What You Get

✅ Satellite change detection for any city  
✅ Fetches images directly from API  
✅ **Saves 6 GB disk space** (stores only 600 MB)  
✅ Detects: deforestation, construction, water loss  

---

## 3-Step Setup (10 minutes)

### Step 1: Get Free API Access (5 min)

1. Go to https://dataspace.copernicus.eu
2. Click "Register" → Create account (free!)
3. Confirm your email
4. Note your **username** and **password**

### Step 2: Add Credentials (1 min)

Edit `e:\geo-watch\.env`:

```env
COPERNICUS_USERNAME=your_username_here
COPERNICUS_PASSWORD=your_password_here
```

### Step 3: Install & Run (4 min)

```powershell
# Make sure Python 3.11 venv is active
cd e:\geo-watch
.\venv\Scripts\Activate.ps1

# Install streaming package
pip install sentinelhub>=3.10.0

# Install PyTorch with GPU (if not done)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt

# Run streaming analysis!
python scripts/stream_and_analyze.py --city bangalore --before 2020-02-01 --after 2024-02-01
```

---

## ⏱️ What to Expect

```
[1/16] Fetching & analyzing tile... ████████ 100%
[2/16] Fetching & analyzing tile... ████████ 100%
...
[16/16] Fetching & analyzing tile... ████████ 100%

✓ Analysis complete! (25 minutes with GPU)

Results:
  - Deforestation: 450 hectares
  - New construction: 230 hectares
  - Water loss: 15 hectares

💾 Disk space used: 600 MB (saved 6 GB!)
```

---

## 🌐 Use Web Interface

### Start Backend:
```powershell
python backend/main.py
```

### Open Frontend:
1. Open `frontend/index.html` in browser
2. Select city: **Bangalore**
3. Before date: **2020-02-01**
4. After date: **2024-02-01**
5. Click **"Analyze Changes"**
6. Watch progress bar!

Results appear as an interactive map showing:
- 🟥 Red = Deforestation
- 🟦 Blue = New buildings
- 🟨 Yellow = Water loss

---

## 📊 Disk Space Comparison

**Old way (download everything):**
```
Raw images: 6 GB ❌
Processed: 500 MB
Results: 100 MB
─────────────────
TOTAL: 6.6 GB
```

**New way (streaming):**
```
Processed: 500 MB ✅
Results: 100 MB
─────────────────
TOTAL: 600 MB
```

**You save: 6 GB per city analyzed!**

---

## 📁 Where Are Results?

```
e:\geo-watch\data\processed\bangalore\
└── analysis_2020-02-01_to_2024-02-01\
    ├── changes_summary.json     ← Statistics
    └── change_map.png           ← Visualization
```

---

## 🎯 Analyze Other Cities

### Mumbai:
```powershell
python scripts/stream_and_analyze.py --city mumbai --before 2020-02-01 --after 2024-02-01
```

### Delhi:
```powershell
python scripts/stream_and_analyze.py --city delhi --before 2020-02-01 --after 2024-02-01
```

### Hyderabad:
```powershell
python scripts/stream_and_analyze.py --city hyderabad --before 2020-02-01 --after 2024-02-01
```

---

## 🔧 Options

**Faster (lower detail):**
```powershell
python scripts/stream_and_analyze.py --city bangalore --grid-size 2
# 2x2 = 4 tiles, ~8 minutes
```

**Default (balanced):**
```powershell
python scripts/stream_and_analyze.py --city bangalore --grid-size 4
# 4x4 = 16 tiles, ~25 minutes
```

**Higher detail (slower):**
```powershell
python scripts/stream_and_analyze.py --city bangalore --grid-size 8
# 8x8 = 64 tiles, ~2 hours
```

---

## ⚠️ Troubleshooting

**"No credentials found"**
→ Check `.env` file has COPERNICUS_USERNAME and COPERNICUS_PASSWORD

**"Authentication failed"**
→ Verify credentials at https://dataspace.copernicus.eu

**"CUDA not available"**
→ Will use CPU automatically (slower, ~1.5 hours)

**"Out of memory"**
→ Use smaller grid: `--grid-size 2`

---

## 📚 More Info

- Full guide: [STREAMING_MODE_GUIDE.md](STREAMING_MODE_GUIDE.md)
- Setup details: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- Dataset info: [DATASET_GUIDE.md](DATASET_GUIDE.md)

---

## ✅ Done!

You now have:
- ✅ Free satellite data access
- ✅ AI change detection running
- ✅ Results displayed on map
- ✅ 90% less disk usage

**Next:** Analyze your city or any location worldwide! 🌍
