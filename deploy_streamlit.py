#!/usr/bin/env python3
"""
Streamlit Cloud Deployment Test Script
This script tests the app locally before deploying to Streamlit Cloud
"""

import subprocess
import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✅ Pillow {PIL.__version__}")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True

def test_app_import():
    """Test if the app can be imported"""
    print("\n🔍 Testing app import...")
    
    try:
        import app
        print("✅ App imports successfully")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def check_files():
    """Check if all required files exist"""
    print("\n🔍 Checking required files...")
    
    required_files = [
        'streamlit_app.py',
        'app.py',
        'requirements.txt',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def main():
    """Main deployment test function"""
    print("🚀 Streamlit Cloud Deployment Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Test app import
    if not test_app_import():
        print("\n❌ App import test failed. Please check app.py")
        return False
    
    # Check files
    files_ok, missing = check_files()
    if not files_ok:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✅ All tests passed!")
    print("\n🚀 Ready for Streamlit Cloud deployment!")
    print("\nNext steps:")
    print("1. Push your code to GitHub")
    print("2. Go to https://share.streamlit.io")
    print("3. Connect your GitHub repository")
    print("4. Set main file to: streamlit_app.py")
    print("5. Deploy!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
