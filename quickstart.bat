@echo off
REM Quick Start Script for Windows

echo ========================================
echo GEO-WATCH Quick Start Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check Python version
echo Checking Python version...
python -c "import sys; v=sys.version_info; exit(0 if v.major==3 and v.minor in [8,9,10,11,12,13] else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8-3.13 is required
    python --version
    pause
    exit /b 1
)
python --version
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import torch" 2>nul
if errorlevel 1 (
    echo Installing dependencies... This may take 10-15 minutes.
    echo.
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    echo.
) else (
    echo Dependencies already installed.
    echo.
)

REM Setup directories
echo Setting up project directories...
python scripts\setup_directories.py
echo.

REM Check for .env file
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo ⚠️  IMPORTANT: Edit .env file and add your Copernicus credentials!
    echo Get credentials from: https://dataspace.copernicus.eu
    echo.
    pause
)

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure you've added Copernicus credentials to .env file
echo 2. Download Sentinel-2 data for Bangalore
echo 3. Run: python scripts\preprocess.py --city bangalore
echo 4. Run: python scripts\run_segmentation.py --city bangalore
echo 5. Run: python scripts\detect_changes.py --city bangalore
echo 6. Start backend: python backend\main.py
echo 7. Start frontend: cd frontend ^&^& python -m http.server 3000
echo.
echo For detailed instructions, see QUICKSTART_BANGALORE.md
echo.
pause
