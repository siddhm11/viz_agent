# Create a configuration file for the app
config_code = '''
import os
from typing import Dict, Any

class AppConfig:
    """Configuration settings for the Data Analysis Dashboard"""
    
    # App settings
    APP_TITLE = "AI Data Analysis Dashboard"
    APP_ICON = "📊"
    LAYOUT = "wide"
    
    # File upload settings
    MAX_UPLOAD_SIZE_MB = 200
    ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']
    
    # Analysis settings
    DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"
    MAX_VISUALIZATIONS = 10
    
    # UI settings
    THEME_COLORS = {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545',
        'info': '#17a2b8'
    }
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    
    @classmethod
    def get_groq_api_key(cls) -> str:
        """Get Groq API key from environment or session state"""
        return os.getenv('GROQ_API_KEY', '')
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary"""
        return {
            'app_title': cls.APP_TITLE,
            'app_icon': cls.APP_ICON,
            'layout': cls.LAYOUT,
            'max_upload_size_mb': cls.MAX_UPLOAD_SIZE_MB,
            'allowed_file_types': cls.ALLOWED_FILE_TYPES,
            'default_model': cls.DEFAULT_MODEL,
            'max_visualizations': cls.MAX_VISUALIZATIONS,
            'theme_colors': cls.THEME_COLORS,
            'cache_ttl': cls.CACHE_TTL
        }

# Environment-specific settings
class DevelopmentConfig(AppConfig):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(AppConfig):
    DEBUG = False
    LOG_LEVEL = "INFO"

# Get configuration based on environment
def get_config():
    env = os.getenv('ENVIRONMENT', 'development').lower()
    if env == 'production':
        return ProductionConfig()
    return DevelopmentConfig()
'''

with open('config.py', 'w') as f:
    f.write(config_code)

print("✅ Configuration file created: config.py")