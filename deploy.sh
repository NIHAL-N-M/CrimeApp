#!/bin/bash

# Face Recognition App Deployment Script
echo "🔍 Face Recognition Challenge - Deployment Script"
echo "================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p face_samples face_samples2 Images

# Build and start the application
echo "🚀 Building and starting the application..."
docker-compose up --build -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    echo "✅ Application is running successfully!"
    echo "🌐 Open your browser and go to: http://localhost:8501"
    echo ""
    echo "📋 Available endpoints:"
    echo "   - Main App: http://localhost:8501"
    echo "   - Health Check: http://localhost:8501/_stcore/health"
    echo ""
    echo "🛠️  Management commands:"
    echo "   - View logs: docker-compose logs -f"
    echo "   - Stop app: docker-compose down"
    echo "   - Restart app: docker-compose restart"
else
    echo "❌ Application failed to start. Check logs with: docker-compose logs"
    exit 1
fi
