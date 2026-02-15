@echo off
echo ========================================
echo Restarting GeoWatch Backend Server
echo ========================================
echo.
echo Killing existing server...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo Starting backend with real satellite image support...
start /B python backend\main.py

echo.
echo ✓ Server starting on http://localhost:8000
echo ✓ Open frontend/dashboard.html in your browser
echo ✓ Draw a region and fetch REAL satellite images!
echo.
pause
