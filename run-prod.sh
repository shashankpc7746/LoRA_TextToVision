#!/bin/bash
# Production Run Script for Task-6 Production Hardening
# Multi-worker Gunicorn deployment with Uvicorn workers

# Set environment variables
export APP_MODULE="adaptive_api:adaptive_app"
export WORKERS=4
export BIND_HOST="0.0.0.0"
export BIND_PORT="8001"

# Production settings
export PYTHONPATH="${PYTHONPATH}:$(pwd)/AnimateDiff"

echo "🚀 Starting Adaptive Video Generation API (Production Mode)"
echo "📍 Module: $APP_MODULE"
echo "👷 Workers: $WORKERS"
echo "🌐 Bind: $BIND_HOST:$BIND_PORT"
echo "📊 Gunicorn with Uvicorn workers for optimal async performance"
echo ""

# Start Gunicorn with Uvicorn workers
exec gunicorn \
    --workers $WORKERS \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind $BIND_HOST:$BIND_PORT \
    --timeout 300 \
    --keep-alive 75 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --reload \
    $APP_MODULE