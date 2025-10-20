#!/usr/bin/env python3
"""
Quick start script for the Face Recognition Web Application
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'opencv-python', 'numpy', 'Pillow', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ All packages installed!")
    else:
        print("✅ All required packages are installed!")

def start_app():
    """Start the Streamlit application"""
    print("🚀 Starting Face Recognition Web Application...")
    print("🌐 The app will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    print("🔍 Face Recognition Challenge - Quick Start")
    print("=" * 50)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check and install requirements
    check_requirements()
    
    # Start the application
    start_app()
