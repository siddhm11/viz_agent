#!/bin/bash
echo "🤖 AI Data Analysis Dashboard"
echo "=============================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔧 Setting up virtual environment..."
    python3 setup.py
    echo
    echo "💡 Please activate the virtual environment and run this script again:"
    echo "   source venv/bin/activate"
    echo "   ./launch.sh"
    exit 0
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Launch the application
echo "🚀 Launching AI Data Analysis Dashboard..."
python3 launch.py "$@"
