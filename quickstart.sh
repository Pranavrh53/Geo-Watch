#!/bin/bash
# Quick Start Script for Linux/Mac

echo "========================================"
echo "GEO-WATCH Quick Start Script"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Check if requirements are installed
echo "Checking dependencies..."
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies... This may take 10-15 minutes."
    pip install -r requirements.txt
    echo ""
else
    echo "Dependencies already installed."
    echo ""
fi

# Setup directories
echo "Setting up project directories..."
python scripts/setup_directories.py
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your Copernicus credentials!"
    echo "Get credentials from: https://dataspace.copernicus.eu"
    echo ""
    read -p "Press enter to continue..."
fi

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Make sure you've added Copernicus credentials to .env file"
echo "2. Download Sentinel-2 data for Bangalore"
echo "3. Run: python scripts/preprocess.py --city bangalore"
echo "4. Run: python scripts/run_segmentation.py --city bangalore"
echo "5. Run: python scripts/detect_changes.py --city bangalore"
echo "6. Start backend: python backend/main.py"
echo "7. Start frontend: cd frontend && python -m http.server 3000"
echo ""
echo "For detailed instructions, see QUICKSTART_BANGALORE.md"
echo ""
