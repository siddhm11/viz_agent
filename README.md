# 🤖 AI Data Analysis Dashboard

A powerful, AI-driven data analysis dashboard built with Streamlit and LangGraph that automatically analyzes your data and generates insightful visualizations using cutting-edge AI technology.

## ✨ Features

### 🎯 Core Capabilities
- **🔍 Automatic Data Analysis**: AI-powered analysis of data types, relationships, and patterns
- **📊 Smart Visualizations**: Automatically generates appropriate charts based on your data
- **🤖 AI Insights**: LLM-generated insights and recommendations for your data
- **🔄 Self-Healing Code**: Advanced error recovery and code correction mechanisms
- **📈 Interactive Dashboard**: Beautiful, modern interface with real-time progress tracking

### 🎨 User Experience
- **🚀 One-Click Analysis**: Upload data and get insights in seconds
- **📱 Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **🎭 Modern UI**: Custom CSS with animations, gradients, and professional styling
- **📚 Analysis History**: Track all your previous analyses
- **💾 Export Options**: Download visualizations and analysis reports

### 🔧 Technical Features
- **⚡ Async Processing**: Non-blocking analysis with real-time progress updates
- **🔄 LangGraph Integration**: Sophisticated workflow orchestration
- **🛡️ Error Handling**: Comprehensive error recovery and user feedback
- **🐳 Docker Support**: Easy deployment with containerization
- **⚙️ Configurable**: Extensive configuration options and environment management

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/keys))

### 🔧 Installation

#### Option 1: Automated Setup (Recommended)
```bash
# Clone or download the project files
# Run the setup script
python setup.py

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
python setup.py install
```

#### Option 2: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir logs data exports cache
```

### 🔑 Configuration

1. **Edit the `.env` file** (created during setup):
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
ENVIRONMENT=development
DEBUG=true
```

2. **Place your DataAnalystAgent code** as `paste.py` in the project directory

### 🎮 Running the Application

#### Basic Version
```bash
streamlit run streamlit_app.py
```

#### Enhanced Version (Recommended)
```bash
streamlit run enhanced_app.py
```

The app will be available at: **http://localhost:8501**

## 🐳 Docker Deployment

### Quick Docker Run
```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t ai-data-dashboard .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key ai-data-dashboard
```

### Production Deployment
```bash
# Set environment variables
export GROQ_API_KEY=your_actual_key

# Run in production mode
docker-compose -f docker-compose.yml up -d
```

## 📖 How to Use

### 1. 📊 Data Upload
- **Supported formats**: CSV, Excel (.xlsx, .xls)
- **File size limit**: Up to 200MB
- **Data validation**: Automatic quality checks and issue detection

### 2. 🔑 API Configuration
- Enter your Groq API key in the sidebar
- The key is stored securely in session state
- Required for AI-powered analysis

### 3. 🚀 Run Analysis
- Click "Run Analysis" after uploading data
- Watch real-time progress updates
- Get comprehensive results in seconds

### 4. 📈 View Results
- **Visualizations**: Interactive charts with insights
- **AI Summary**: Detailed analysis and recommendations
- **Technical Details**: Code and debugging information
- **Export Options**: Download images and reports

### 5. 📚 Track History
- View all previous analyses
- Compare results across different datasets
- Track success rates and performance

## 🛠️ Architecture

### Frontend (Streamlit)
- **Modern UI**: Custom CSS with animations and responsive design
- **Real-time Updates**: Progress tracking and live feedback
- **Interactive Components**: File upload, settings, export options
- **Session Management**: Persistent state across user interactions

### Backend (LangGraph + Groq)
- **AI Agent**: DataAnalystAgent with workflow orchestration
- **Smart Analysis**: Automatic data type detection and relationship identification
- **Visualization Generation**: AI-generated plotting code with self-healing
- **Error Recovery**: Advanced debugging and code correction

### Integration Layer
- **State Management**: Seamless data flow between frontend and backend
- **Error Handling**: Comprehensive error catching and user feedback
- **Performance Optimization**: Caching and efficient processing

## 📁 Project Structure

```
ai-data-dashboard/
├── streamlit_app.py          # Basic Streamlit application
├── enhanced_app.py           # Enhanced version with advanced features
├── config.py                 # Configuration management
├── paste.py                  # Your DataAnalystAgent code
├── requirements.txt          # Python dependencies
├── setup.py                 # Automated setup script
├── .env                     # Environment variables (created during setup)
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── .dockerignore           # Docker ignore rules
├── README.md               # This file
├── logs/                   # Application logs
├── data/                   # Uploaded data files
├── exports/                # Exported reports and visualizations
└── cache/                  # Temporary cache files
```

## ⚙️ Configuration Options

### Environment Variables
```bash
# API Configuration
GROQ_API_KEY=your_groq_api_key

# App Settings
ENVIRONMENT=development|production
DEBUG=true|false
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR

# Streamlit Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Cache Settings
CACHE_TTL=3600
```

### App Configuration (config.py)
- **File upload limits**: Customize maximum file size
- **Visualization settings**: Control chart generation parameters
- **UI themes**: Customize colors and styling
- **Performance tuning**: Adjust cache and processing settings

## 🎨 Customization

### Styling
- **Custom CSS**: Modify the `load_enhanced_css()` function
- **Theme colors**: Update CSS variables in the configuration
- **Layout options**: Adjust sidebar, main content, and component spacing

### Functionality
- **Data processing**: Extend the DataUploader class
- **Visualization types**: Add new chart types to the analysis agent
- **Export formats**: Add new export options in ResultsDisplay
- **Analysis parameters**: Customize the AnalysisRunner configuration

## 🔍 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### "API key not configured" error
- Check that your `.env` file contains `GROQ_API_KEY=your_actual_key`
- Verify the key is valid at [Groq Console](https://console.groq.com/keys)
- Restart the Streamlit app after updating the key

#### "DataAnalystAgent not found" error
- Ensure your original code is saved as `paste.py` in the project directory
- Check that the import statement matches your class name
- Verify the file contains the complete DataAnalystAgent implementation

#### Visualization generation failures
- Check the logs in the `logs/` directory for detailed error information
- Verify your data doesn't have formatting issues
- Try with a smaller dataset first

### Performance Issues
- **Large datasets**: Consider sampling your data for initial analysis
- **Memory usage**: Monitor system resources during analysis
- **Processing time**: Complex datasets may take several minutes

### Docker Issues
```bash
# Check container logs
docker-compose logs streamlit-app

# Rebuild container
docker-compose down
docker-compose up --build

# Check port conflicts
netstat -an | grep 8501
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/ai-data-dashboard.git

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Start development server
streamlit run enhanced_app.py --server.runOnSave true
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing framework that makes web apps simple
- **LangGraph**: For powerful workflow orchestration capabilities
- **Groq**: For lightning-fast AI inference
- **LangChain**: For comprehensive LLM integration tools

## 📞 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Community**: Join the Streamlit community for general questions

## 🔄 Version History

### v2.0 (Enhanced Edition)
- ✨ Modern UI with animations and gradients
- 🔄 Real-time progress tracking
- 📊 Enhanced visualization display
- 💾 Export and download options
- 📚 Analysis history tracking
- 🐳 Docker support
- ⚙️ Advanced configuration options

### v1.0 (Basic Edition)
- 🎯 Core data analysis functionality
- 📊 Basic visualization generation
- 🤖 AI-powered insights
- 📁 File upload support
- 🔑 API key management

---

**Happy Analyzing! 📊✨**

*Built with ❤️ using Streamlit, LangGraph, and AI*