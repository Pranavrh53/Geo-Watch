@echo off
echo ================================================
echo   Geo-Watch Interactive Dashboard
echo ================================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_python311.bat first
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/3] Initializing database...
python backend\database.py
if errorlevel 1 (
    echo ERROR: Database initialization failed
    pause
    exit /b 1
)

echo [3/3] Starting backend server...
echo.
echo ================================================
echo   Server starting on http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo ================================================
echo.
echo Open in browser: frontend\login.html
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

python backend\main.py

pause
