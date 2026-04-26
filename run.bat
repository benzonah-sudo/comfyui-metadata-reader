@echo off
echo ComfyUI Metadata Reader
echo -----------------------

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python from https://python.org
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop.
echo.

start "" http://localhost:5000
python app.py
pause
