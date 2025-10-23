#!/bin/bash
# TTV Service Production Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting TTV Service Production Deployment${NC}"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not available. Please install Docker Compose.${NC}"
    exit 1
fi

# Determine Docker Compose command
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Check for NVIDIA Docker support (for GPU)
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  GPU support not available. Running in CPU mode.${NC}"
    export COMPOSE_FILE="docker-compose.yml:docker-compose.cpu.yml"
fi

# Load environment variables
if [ -f .env ]; then
    echo -e "${GREEN}📋 Loading environment variables from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}⚠️  No .env file found. Using default values.${NC}"
    echo -e "${YELLOW}    Please create .env file from .env.example for production${NC}"
fi

# Create necessary directories
echo -e "${GREEN}📁 Creating necessary directories${NC}"
mkdir -p storage cache temp logs nginx/ssl

# Generate SSL certificates if they don't exist
if [ ! -f nginx/ssl/ttv.crt ] || [ ! -f nginx/ssl/ttv.key ]; then
    echo -e "${GREEN}🔒 Generating self-signed SSL certificates${NC}"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/ttv.key \
        -out nginx/ssl/ttv.crt \
        -subj "/C=US/ST=State/L=City/O=TTV/CN=ttv.bhiv.local"
fi

# Set proper permissions
echo -e "${GREEN}🔐 Setting file permissions${NC}"
chmod 600 nginx/ssl/ttv.key
chmod 644 nginx/ssl/ttv.crt
chmod -R 755 storage cache temp logs

# Initialize database
echo -e "${GREEN}🗄️  Initializing database${NC}"
cat > scripts/init_db.sql << 'EOF'
-- TTV Service Database Initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create application user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'ttv_app') THEN
        CREATE USER ttv_app WITH PASSWORD 'ttv_app_password';
    END IF;
END
$$;

-- Grant permissions
GRANT CONNECT ON DATABASE ai_agent TO ttv_app;
GRANT USAGE ON SCHEMA public TO ttv_app;
GRANT CREATE ON SCHEMA public TO ttv_app;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ttv_jobs_user_id ON ttv_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_ttv_jobs_status ON ttv_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ttv_jobs_created_at ON ttv_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_ttv_audit_logs_user_id ON ttv_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_ttv_audit_logs_timestamp ON ttv_audit_logs(timestamp);
EOF

# Build and start services
echo -e "${GREEN}🏗️  Building TTV Service containers${NC}"
$DOCKER_COMPOSE build --no-cache

echo -e "${GREEN}🚀 Starting TTV Service stack${NC}"
$DOCKER_COMPOSE up -d

# Wait for services to be ready
echo -e "${GREEN}⏳ Waiting for services to be ready${NC}"
sleep 30

# Health checks
echo -e "${GREEN}🏥 Performing health checks${NC}"

# Check PostgreSQL
if $DOCKER_COMPOSE exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL is healthy${NC}"
else
    echo -e "${RED}❌ PostgreSQL health check failed${NC}"
fi

# Check Redis
if $DOCKER_COMPOSE exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis is healthy${NC}"
else
    echo -e "${RED}❌ Redis health check failed${NC}"
fi

# Check TTV API
if curl -f http://localhost:8002/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ TTV API is healthy${NC}"
else
    echo -e "${RED}❌ TTV API health check failed${NC}"
fi

# Check Nginx
if curl -f -k https://localhost/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Nginx proxy is healthy${NC}"
else
    echo -e "${RED}❌ Nginx proxy health check failed${NC}"
fi

# Display service URLs
echo -e "\n${GREEN}🌐 Service URLs:${NC}"
echo -e "  API: https://localhost/api/v1/ttv"
echo -e "  Health: https://localhost/health"
echo -e "  Docs: https://localhost/docs"
echo -e "  Flower: http://localhost:5555"
echo -e "  Redis Commander: http://localhost:8081 (dev mode)"

# Display next steps
echo -e "\n${GREEN}📝 Next Steps:${NC}"
echo -e "  1. Update your .env file with production values"
echo -e "  2. Configure your domain and SSL certificates"
echo -e "  3. Set up monitoring and alerting"
echo -e "  4. Configure backup strategies"
echo -e "  5. Review security settings"

echo -e "\n${GREEN}🎉 TTV Service deployment completed!${NC}"

# Show running containers
echo -e "\n${GREEN}📊 Running containers:${NC}"
$DOCKER_COMPOSE ps