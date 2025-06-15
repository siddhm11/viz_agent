# Create a simple launcher script
launcher_script = '''#!/usr/bin/env python3
"""
Quick launcher for the AI Data Analysis Dashboard
This script provides an easy way to launch the application with different options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up"""
    issues = []
    
    # Check if virtual environment exists
    if not os.path.exists('venv'):
        issues.append("Virtual environment not found. Run: python setup.py")
    
    # Check if requirements are installed
    try:
        import streamlit
    except ImportError:
        issues.append("Streamlit not installed. Run: pip install -r requirements.txt")
    
    # Check if main app files exist
    if not os.path.exists('streamlit_app.py'):
        issues.append("Main app file (streamlit_app.py) not found")
    
    # Check if paste.py exists
    if not os.path.exists('paste.py'):
        issues.append("DataAnalystAgent code (paste.py) not found")
    
    return issues

def get_groq_key():
    """Get Groq API key from environment or .env file"""
    # Check environment variable first
    key = os.getenv('GROQ_API_KEY')
    if key:
        return key
    
    # Check .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GROQ_API_KEY='):
                    return line.split('=', 1)[1].strip()
    
    return None

def launch_app(app_type='enhanced', port=8501, debug=False):
    """Launch the Streamlit application"""
    
    # Determine which app file to run
    if app_type == 'basic':
        app_file = 'streamlit_app.py'
    elif app_type == 'enhanced':
        app_file = 'enhanced_app.py'
    else:
        print(f"❌ Unknown app type: {app_type}")
        return False
    
    if not os.path.exists(app_file):
        print(f"❌ App file not found: {app_file}")
        return False
    
    # Build command
    cmd = [
        'streamlit', 'run', app_file,
        '--server.port', str(port),
        '--server.address', 'localhost'
    ]
    
    if debug:
        cmd.extend(['--logger.level', 'debug'])
    
    # Set environment variables
    env = os.environ.copy()
    if debug:
        env['DEBUG'] = 'true'
    
    print(f"🚀 Launching {app_type} app on port {port}...")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print(f"🛑 Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run(cmd, env=env)
        return True
    except KeyboardInterrupt:
        print("\\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description='AI Data Analysis Dashboard Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    # Launch enhanced app (default)
  python launch.py --basic            # Launch basic app
  python launch.py --port 8502        # Launch on different port
  python launch.py --debug            # Launch with debug output
  python launch.py --check            # Check environment setup
        """
    )
    
    parser.add_argument(
        '--basic', 
        action='store_true',
        help='Launch basic app instead of enhanced version'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port to run the application on (default: 8501)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--check', 
        action='store_true',
        help='Check environment setup without launching'
    )
    
    args = parser.parse_args()
    
    print("🤖 AI Data Analysis Dashboard Launcher")
    print("=" * 50)
    
    # Check environment
    issues = check_environment()
    if issues:
        print("❌ Environment issues found:")
        for issue in issues:
            print(f"   • {issue}")
        print()
        print("🔧 Run the setup script to fix these issues:")
        print("   python setup.py")
        return 1
    
    # Check API key
    api_key = get_groq_key()
    if not api_key or api_key.startswith('your_'):
        print("⚠️  Groq API key not configured")
        print("   Edit .env file and add your API key")
        print("   You can still launch the app and enter the key in the interface")
        print()
    else:
        print("✅ Groq API key configured")
    
    if args.check:
        print("✅ Environment check complete")
        return 0
    
    # Determine app type
    app_type = 'basic' if args.basic else 'enhanced'
    
    # Launch the application
    success = launch_app(app_type, args.port, args.debug)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
'''

with open('launch.py', 'w', encoding='utf-8') as f:
    f.write(launcher_script)

# Create a simple batch file for Windows users
windows_launcher = '''@echo off
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
    echo    venv\\Scripts\\activate
    echo    launch.bat
    pause
    exit /b 0
)

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  Virtual environment not activated
    echo 🔧 Activating virtual environment...
    call venv\\Scripts\\activate
)

REM Launch the application
echo 🚀 Launching AI Data Analysis Dashboard...
python launch.py %*

pause
'''

with open('launch.bat', 'w', encoding='utf-8') as f:
    f.write(windows_launcher)

# Create a shell script for Unix-like systems
unix_launcher = '''#!/bin/bash
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
'''

with open('launch.sh', 'w', encoding='utf-8') as f:
    f.write(unix_launcher)

# Make the shell script executable
os.chmod('launch.sh', 0o755)

print("✅ Launcher scripts created:")
print("   - launch.py (Python launcher with options)")
print("   - launch.bat (Windows batch file)")
print("   - launch.sh (Unix/Linux shell script)")