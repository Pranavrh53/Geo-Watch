@echo off
REM Python 3.11 GPU Setup Script
REM This script helps you set up Python 3.11 with CUDA support

echo ========================================
echo Python 3.11 + GPU Setup Script
echo ========================================
echo.

echo Step 1: Checking current Python version...
python --version
echo.

echo Step 2: Deactivating current virtual environment...
call deactivate 2>nul
echo.

echo Step 3: Removing old virtual environment...
echo This will delete the existing venv folder...
pause
rmdir /s /q venv
echo Old venv removed.
echo.

echo ========================================
echo MANUAL STEPS REQUIRED
echo ========================================
echo.
echo Please complete these steps:
echo.
echo 1. Download Python 3.11.9 from:
echo    https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
echo.
echo 2. Run the installer:
echo    - Check "Add Python 3.11 to PATH"
echo    - Click "Install Now"
echo.
echo 3. Close this window and RESTART VS Code/Terminal
echo.
echo 4. Run: python --version
echo    (should show Python 3.11.9)
echo.
echo 5. Run: setup_python311.bat
echo    (this will complete the setup with GPU support)
echo.
pause
