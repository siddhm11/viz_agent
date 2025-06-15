# Create a deployment and setup script
setup_script = '''#!/usr/bin/env python3
"""
Setup script for the AI Data Analysis Dashboard
This script helps set up the environment and dependencies for the Streamlit app.
"""

import os
import subprocess
import sys
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"{'='*50}")
    print(f"🔄 {description if description else command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}")
    return True

def create_virtual_environment():
    """Create a virtual environment"""
    if os.path.exists("venv"):
        print("📁 Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def activate_virtual_environment():
    """Instructions for activating virtual environment"""
    system = platform.system().lower()
    
    print("\\n🔧 To activate the virtual environment:")
    if system == "windows":
        print("   venv\\\\Scripts\\\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\\n💡 Then run: python setup.py install")

def install_dependencies():
    """Install required dependencies"""
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  Warning: Not in a virtual environment")
        print("It's recommended to use a virtual environment")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Install packages
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def create_env_file():
    """Create a .env file template"""
    env_content = """# Environment Configuration for AI Data Analysis Dashboard

# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# App Configuration
ENVIRONMENT=development
DEBUG=true

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Cache Configuration
CACHE_TTL=3600

# Security
SECRET_KEY=your_secret_key_here

# Optional: External Services
# DATABASE_URL=sqlite:///data/analytics.db
# REDIS_URL=redis://localhost:6379
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file template")
        print("📝 Please edit .env and add your API keys")
    else:
        print("📁 .env file already exists")

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'data', 'exports', 'cache']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def run_tests():
    """Run basic tests to verify installation"""
    print("\\n🧪 Running basic tests...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError:
        print("❌ Failed to import Streamlit")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError:
        print("❌ Failed to import Pandas")
        return False
    
    try:
        import langchain_groq
        print("✅ LangChain Groq imported successfully")
    except ImportError:
        print("❌ Failed to import LangChain Groq")
        return False
    
    # Check if the main app file exists
    if os.path.exists('streamlit_app.py'):
        print("✅ Main app file found")
    else:
        print("❌ Main app file not found")
        return False
    
    print("🎉 All tests passed!")
    return True

def setup_complete():
    """Display setup completion message"""
    print("\\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print()
    print("📋 Next steps:")
    print("1. Edit the .env file and add your Groq API key")
    print("2. Activate your virtual environment:")
    
    system = platform.system().lower()
    if system == "windows":
        print("   venv\\\\Scripts\\\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("3. Run the application:")
    print("   streamlit run streamlit_app.py")
    print("   or")
    print("   streamlit run enhanced_app.py  (for the enhanced version)")
    print()
    print("🌐 The app will be available at: http://localhost:8501")
    print()
    print("📚 For help, see README.md or visit:")
    print("   - Streamlit docs: https://docs.streamlit.io")
    print("   - LangGraph docs: https://python.langchain.com/docs/langgraph")
    print()

def main():
    """Main setup function"""
    print("🚀 AI Data Analysis Dashboard Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if len(sys.argv) > 1 and sys.argv[1] == "install":
        # Install mode - assumes venv is already activated
        print("📦 Installing dependencies...")
        if not install_dependencies():
            sys.exit(1)
        
        create_directories()
        create_env_file()
        
        if run_tests():
            setup_complete()
        else:
            print("❌ Setup completed with errors")
            sys.exit(1)
    
    else:
        # Initial setup mode
        print("🔧 Initial setup...")
        if create_virtual_environment():
            create_directories()
            create_env_file()
            activate_virtual_environment()
        else:
            print("❌ Failed to create virtual environment")
            sys.exit(1)

if __name__ == "__main__":
    main()
'''

with open('setup.py', 'w', encoding='utf-8') as f:
    f.write(setup_script)

print("✅ Setup script created: setup.py")