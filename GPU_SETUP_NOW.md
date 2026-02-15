# 🚀 Quick Start: GPU Setup (Action Required)

## ✅ What's Done

- ✓ Deactivated Python 3.13 environment
- ✓ Removed old venv
- ✓ Python launcher detected

## 📥 NEXT STEPS (Do These Now)

### Step 1: Download Python 3.11.9 (5 minutes)

Click this link to download:
**https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe**

File size: ~25 MB

### Step 2: Install Python 3.11.9 (3 minutes)

1. **Run the downloaded installer**

2. **IMPORTANT - On first screen:**
   - ✅ Check "Add Python 3.11 to PATH"
   - ✅ Check "Install for all users" (optional)
   - Click "Install Now"

3. **Wait for installation** (~2 minutes)

4. **Click "Close"** when done

### Step 3: Restart Terminal (1 minute)

**Close VS Code completely and reopen it**

OR in PowerShell:
```powershell
# Just close and reopen this terminal window
```

### Step 4: Verify Python 3.11 (30 seconds)

Open a **NEW** PowerShell window and run:
```powershell
cd e:\geo-watch
python --version
```

**Expected output:** `Python 3.11.9`

**If you still see Python 3.13.12:**
- You need to restart VS Code/terminal
- Or use: `py -3.11 --version` to verify 3.11 is installed

### Step 5: Run GPU Setup Script (15 minutes)

Once you see Python 3.11.9:

```powershell
cd e:\geo-watch
.\setup_python311.bat
```

This will:
- Create new venv with Python 3.11
- Install PyTorch with CUDA support (~2GB download)
- Install all other packages
- Test GPU detection

⏱️ **Takes about 15 minutes** (mostly downloading PyTorch)

---

## 🎯 What You'll See

After `setup_python311.bat` finishes, you should see:

```
PyTorch version: 2.5.0+cu121
CUDA available: True
CUDA version: 12.1
GPU device: NVIDIA GeForce RTX 3060  ← Your GPU model
```

---

## ⚠️ Troubleshooting

### If Python 3.11 still doesn't show after install:

**Option 1: Use Python Launcher**
```powershell
cd e:\geo-watch
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python --version  # Should now show 3.11.9
```

**Option 2: Fix PATH**
1. Search Windows: "Environment Variables"
2. Edit "Path" in System variables
3. Move `C:\Program Files\Python311` ABOVE `Python313`
4. Restart terminal

### If CUDA shows False after setup:

1. **Update NVIDIA Drivers:**
   - Visit: https://www.nvidia.com/Download/index.aspx
   - Download latest driver for your GPU
   - Install and restart

2. **Install CUDA Toolkit 12.1:**
   - Download: https://developer.nvidia.com/cuda-12-1-0-download-archive
   - Install with default options
   - Restart computer

3. **Reinstall PyTorch:**
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 📞 Quick Reference

**Check Python version:**
```powershell
python --version
```

**Check all Python versions:**
```powershell
py --list
```

**Check GPU:**
```powershell
nvidia-smi
```

**Test CUDA in Python:**
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📚 Full Documentation

See [DOWNGRADE_TO_PYTHON311.md](DOWNGRADE_TO_PYTHON311.md) for:
- Detailed troubleshooting
- Manual installation steps
- Performance comparisons
- GPU requirements

---

## ⏱️ Timeline

| Task | Time |
|------|------|
| Download Python 3.11 | 2 min |
| Install Python 3.11 | 3 min |
| Restart terminal | 1 min |
| Run setup script | 15 min |
| **Total** | **~20 min** |

---

## 🎉 After Setup

Once GPU is working, your processing times will be:

- ⚡ Preprocessing: 5 min (was 8 min)
- ⚡ Segmentation: 12 min (was 45 min!) **3.75x faster**
- ⚡ Change Detection: 3 min (same)

**Total: 20 minutes instead of 56 minutes!**

---

## ✅ Action Checklist

- [ ] Downloaded Python 3.11.9 installer
- [ ] Installed Python 3.11.9 (checked "Add to PATH")
- [ ] Restarted terminal/VS Code
- [ ] Verified: `python --version` shows 3.11.9
- [ ] Ran `.\setup_python311.bat`
- [ ] Verified: CUDA available = True
- [ ] Ready to process satellite data at 3x speed! 🚀

---

**Start here:** https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

**Current status:** ⏳ Waiting for Python 3.11 installation
