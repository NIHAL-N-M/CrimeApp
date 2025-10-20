#!/usr/bin/env python3
"""
Face Recognition Challenge - Startup Script
This script starts the face recognition application with proper error handling.
"""

import sys
import os
import subprocess
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'opencv-python',
        'numpy',
        'Pillow',
        'pymysql',
        'tkVideo',
        'streamlit',
        'Django',
        'requests',
        'beautifulsoup4'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'Pillow':
                import PIL
            elif package == 'pymysql':
                import pymysql
            elif package == 'tkVideo':
                import tkvideo
            elif package == 'streamlit':
                import streamlit
            elif package == 'Django':
                import django
            elif package == 'requests':
                import requests
            elif package == 'beautifulsoup4':
                import bs4
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip3 install -r requirements_core.txt")
        return False
    
    return True

def check_database():
    """Check if MySQL database is running and accessible"""
    try:
        import pymysql
        db = pymysql.connect(host="localhost", user="root", password="", database="criminaldb")
        db.close()
        print("Database connection successful!")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Please ensure MySQL is running and the criminaldb database exists.")
        return False

def start_application():
    """Start the main application"""
    try:
        print("Starting Face Recognition Application...")
        from main import window
        window.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        return False
    
    return True

def main():
    """Main function to start the application"""
    print("=" * 50)
    print("Face Recognition Challenge - Microsoft Engage'22")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return
    
    # Check database
    print("Checking database connection...")
    if not check_database():
        print("Please set up the database and try again.")
        return
    
    # Start application
    print("All checks passed! Starting application...")
    if start_application():
        print("Application started successfully!")
    else:
        print("Failed to start application.")

if __name__ == "__main__":
    main()

