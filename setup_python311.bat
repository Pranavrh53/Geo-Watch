@echo off
REM Setup Python 3.11 with CUDA Support
REM Run this AFTER installing Python 3.11

echo ========================================
echo Python 3.11 + CUDA Setup
echo ========================================
echo.

echo Checking Python version...
python --version | findstr "3.11" >nul
if errorlevel 1 (
    echo ERROR: Python 3.11 not found!
    echo Please install Python 3.11.9 first.
    echo Download from: https://www.python.org/ftp/python/3.11.9/
    pause
    exit /b 1
)

python --version
echo ✓ Python 3.11 detected
echo.

echo Creating new virtual environment with Python 3.11...
python -m venv venv
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel
echo.

echo Installing core utilities...
pip install python-dotenv click pyyaml requests tqdm
echo.

echo Installing numpy and pandas...
pip install "numpy>=1.24.0,<2.0" "pandas>=2.0.0"
echo.

echo ========================================
echo Installing PyTorch with CUDA 12.1 Support
echo ========================================
echo.
echo This may take 5-10 minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

echo Installing Transformers and ML libraries...
pip install transformers scikit-learn opencv-python
echo.

echo Installing FastAPI and web frameworks...
pip install fastapi uvicorn[standard] pydantic
echo.

echo Installing geospatial libraries...
pip install rasterio geopandas shapely pyproj
echo.

echo Installing remaining packages...
pip install sentinelsat folium seaborn plotly reportlab fpdf2 scikit-image python-multipart aiofiles httpx matplotlib
echo.

echo ========================================
echo Verifying CUDA Support
echo ========================================
echo.
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo.

if errorlevel 1 (
    echo.
    echo ⚠️ Warning: CUDA verification failed
    echo Your GPU may not be properly configured.
    echo.
) else (
    echo ✅ Setup complete!
    echo.
)

echo ========================================
echo Next Steps
echo ========================================
echo.
echo 1. Check that CUDA is available above
echo 2. If CUDA shows False, you may need to:
echo    - Update NVIDIA GPU drivers
echo    - Install CUDA Toolkit 12.1
echo.
echo 3. If CUDA shows True, you're ready!
echo    Run: python scripts\setup_directories.py
echo.
pause
