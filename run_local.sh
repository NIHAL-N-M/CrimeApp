#!/bin/bash

# Face Recognition App - Local Development Script
echo "🔍 Face Recognition Challenge - Local Development"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 is installed"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 is installed"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements_web.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p face_samples face_samples2 Images

# Run the application
echo "🚀 Starting the application..."
echo "🌐 The app will be available at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

streamlit run app.py --server.port=8501 --server.address=0.0.0.0
