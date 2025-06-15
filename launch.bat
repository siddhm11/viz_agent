@echo off
echo 🤖 AI Data Analysis Dashboard
echo ==============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 🔧 Setting up virtual environment...
    python setup.py
    echo.
    echo 💡 Please activate the virtual environment and run this script again:
    echo    venv\Scripts\activate
    echo    launch.bat
    pause
    exit /b 0
)

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  Virtual environment not activated
    echo 🔧 Activating virtual environment...
    call venv\Scripts\activate
)

REM Launch the application
echo 🚀 Launching AI Data Analysis Dashboard...
python launch.py %*

pause
