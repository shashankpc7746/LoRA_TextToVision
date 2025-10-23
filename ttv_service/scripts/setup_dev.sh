#!/bin/bash
# TTV Service Development Setup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Setting up TTV Service Development Environment${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}❌ Python 3.10+ is required. Current version: $python_version${NC}"
    exit 1
fi

# Create virtual environment
if [ ! -d "ttv_env" ]; then
    echo -e "${GREEN}📦 Creating Python virtual environment${NC}"
    python3 -m venv ttv_env
fi

# Activate virtual environment
source ttv_env/bin/activate

# Upgrade pip
echo -e "${GREEN}⬆️  Upgrading pip${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${GREEN}📦 Installing Python dependencies${NC}"
pip install -r ttv_service/requirements.txt
pip install -r requirements-dev.txt

# Create development .env file
if [ ! -f .env ]; then
    echo -e "${GREEN}📋 Creating development .env file${NC}"
    cp ttv_service/.env.example .env
    
    # Update for development
    sed -i 's/ENVIRONMENT=production/ENVIRONMENT=development/' .env
    sed -i 's/TTV_DEBUG=false/TTV_DEBUG=true/' .env
    sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/' .env
fi

# Create development directories
echo -e "${GREEN}📁 Creating development directories${NC}"
mkdir -p storage cache temp logs

# Set up pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo -e "${GREEN}🪝 Setting up pre-commit hooks${NC}"
    pre-commit install
fi

# Install development tools
echo -e "${GREEN}🛠️  Installing development tools${NC}"
pip install black flake8 pytest pytest-asyncio pytest-cov mypy

# Set up database for development (SQLite)
echo -e "${GREEN}🗄️  Setting up development database${NC}"
python -c "
from ttv_service.job_manager import job_manager
from ttv_service.security import audit_logger
from sqlmodel import SQLModel
SQLModel.metadata.create_all(job_manager.db_engine)
print('✅ Database tables created')
"

# Run tests
echo -e "${GREEN}🧪 Running initial tests${NC}"
pytest ttv_service/tests/ -v || echo -e "${YELLOW}⚠️  Some tests failed (expected for initial setup)${NC}"

# Display development commands
echo -e "\n${GREEN}🚀 Development Commands:${NC}"
echo -e "  Start API server:     uvicorn ttv_service.main:app --reload --host 0.0.0.0 --port 8002"
echo -e "  Start Celery worker:  celery -A ttv_service.job_manager.celery_app worker --loglevel=debug"
echo -e "  Start Celery beat:    celery -A ttv_service.job_manager.celery_app beat --loglevel=debug"
echo -e "  Run tests:            pytest ttv_service/tests/ -v"
echo -e "  Format code:          black ttv_service/"
echo -e "  Lint code:            flake8 ttv_service/"
echo -e "  Type check:           mypy ttv_service/"

# Create development start script
cat > start_dev.sh << 'EOF'
#!/bin/bash
# Development startup script

# Activate virtual environment
source ttv_env/bin/activate

# Start Redis (if not running)
if ! pgrep redis-server > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Export environment variables
export $(cat .env | grep -v '^#' | xargs)

# Start services in background
echo "Starting TTV API server..."
uvicorn ttv_service.main:app --reload --host 0.0.0.0 --port 8002 &
API_PID=$!

echo "Starting Celery worker..."
celery -A ttv_service.job_manager.celery_app worker --loglevel=debug &
WORKER_PID=$!

echo "Starting Celery beat..."
celery -A ttv_service.job_manager.celery_app beat --loglevel=debug &
BEAT_PID=$!

# Wait for interrupt
echo "Services started. Press Ctrl+C to stop..."
trap "kill $API_PID $WORKER_PID $BEAT_PID" EXIT
wait
EOF

chmod +x start_dev.sh

echo -e "\n${GREEN}🎉 Development environment setup completed!${NC}"
echo -e "\n${GREEN}📝 To start development:${NC}"
echo -e "  1. source ttv_env/bin/activate"
echo -e "  2. ./start_dev.sh"
echo -e "  3. Open http://localhost:8002/docs in your browser"