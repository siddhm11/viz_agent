# Create Docker configuration files
dockerfile_content = '''# Dockerfile for AI Data Analysis Dashboard
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data exports cache

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

with open('Dockerfile', 'w') as f:
    f.write(dockerfile_content)

# Create docker-compose.yml
docker_compose_content = '''version: '3.8'

services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - ENVIRONMENT=production
      - DEBUG=false
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./exports:/app/exports
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Redis for caching (uncomment if needed)
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
  #   restart: unless-stopped
  #   volumes:
  #     - redis_data:/data

# volumes:
#   redis_data:
'''

with open('docker-compose.yml', 'w') as f:
    f.write(docker_compose_content)

# Create .dockerignore
dockerignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/*.log

# Cache
cache/
.streamlit/

# Environment files
.env
.env.local
.env.production

# Data files (if large)
data/*.csv
data/*.xlsx
data/*.json

# Exports
exports/

# Git
.git/
.gitignore

# Documentation
README.md
docs/

# Docker
Dockerfile
docker-compose.yml
.dockerignore
'''

with open('.dockerignore', 'w') as f:
    f.write(dockerignore_content)

print("✅ Docker configuration files created:")
print("   - Dockerfile")
print("   - docker-compose.yml") 
print("   - .dockerignore")