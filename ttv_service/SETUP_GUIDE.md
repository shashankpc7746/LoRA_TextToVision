# TTV Service Quick Setup Guide

## 🚀 Quick Start (Windows)

### 1. Install Dependencies

Open PowerShell in the project root and run:

```powershell
# Run the automated setup script
.\ttv_service\setup_windows.ps1
```

Or manually install dependencies:

```powershell
# Create virtual environment
python -m venv ttv_env

# Activate virtual environment
.\ttv_env\Scripts\Activate.ps1

# Install dependencies
pip install -r ttv_service\requirements.txt

# Install additional dependencies for complete functionality
pip install boto3 supabase PyJWT prometheus-client GPUtil
```

### 2. Setup Environment

```powershell
# Copy environment template
copy ttv_service\.env.example .env

# Edit .env file with your settings (optional for development)
notepad .env
```

### 3. Start Redis (Required)

```powershell
# Using Docker (recommended)
docker run -d -p 6379:6379 --name ttv-redis redis:7-alpine

# Or install Redis for Windows from https://redis.io/download
```

### 4. Start Development Server

```powershell
# Option 1: Use the startup script
.\start_dev.ps1

# Option 2: Manual startup
.\ttv_env\Scripts\Activate.ps1
uvicorn ttv_service.main:app --reload --host 0.0.0.0 --port 8002

# In another terminal for Celery worker:
.\ttv_env\Scripts\Activate.ps1
celery -A ttv_service.job_manager.celery_app worker --loglevel=debug --pool=solo
```

### 5. Test the Service

```powershell
# Run tests
.\run_tests.ps1

# Or manually
pytest ttv_service\tests\ -v
```

### 6. Access the Service

- **API Documentation**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/health
- **Metrics**: http://localhost:8002/metrics

## 🔧 Resolving Import Errors

The import errors you're seeing are expected before installing dependencies. After running the setup:

1. **Install all dependencies** using the setup script or pip
2. **Restart VS Code** to refresh the Python environment
3. **Select the correct Python interpreter**: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `ttv_env\Scripts\python.exe`

## 📦 Dependencies Breakdown

### Core Dependencies
- `fastapi[all]` - Web framework
- `uvicorn[standard]` - ASGI server
- `sqlmodel` - Database ORM
- `celery[redis]` - Task queue
- `redis` - Cache and message broker

### Storage Dependencies
- `boto3` - AWS S3 integration
- `supabase` - Supabase integration

### Security Dependencies
- `PyJWT` - JWT token handling
- `python-jose[cryptography]` - Encryption

### Monitoring Dependencies
- `sentry-sdk[fastapi]` - Error tracking
- `prometheus-client` - Metrics
- `psutil` - System monitoring
- `GPUtil` - GPU monitoring (optional)

## 🐛 Troubleshooting

### Redis Connection Issues
```powershell
# Check if Redis is running
Test-NetConnection -ComputerName localhost -Port 6379

# Start Redis with Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### Python Import Issues
```powershell
# Ensure virtual environment is activated
.\ttv_env\Scripts\Activate.ps1

# Reinstall dependencies
pip install --upgrade -r ttv_service\requirements.txt
```

### GPU Dependencies (Optional)
```powershell
# For GPU monitoring (requires NVIDIA drivers)
pip install GPUtil nvidia-ml-py3

# Skip if no GPU available - service will work without it
```

## 🎯 Production Deployment

For production deployment, see:
- `ttv_service\docker-compose.yml` - Container orchestration
- `ttv_service\nginx\` - Reverse proxy configuration
- `Task-8-Implementation-Summary.md` - Complete deployment guide

## 📞 Support

If you encounter issues:
1. Check the setup logs for error messages
2. Ensure all dependencies are installed
3. Verify Redis is running
4. Check Python version (3.10+ required)

The service is designed to be robust and will gracefully handle missing optional dependencies.