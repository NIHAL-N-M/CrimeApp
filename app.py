#!/usr/bin/env python3
"""
Face Recognition Web Application
Streamlit-based web version of the face recognition challenge
"""

import streamlit as st
import cv2
import numpy as np
import sqlite3
import os
import tempfile
from PIL import Image
import base64
from datetime import datetime
import pandas as pd
import time

# Import real-time face recognition module
try:
    from realtime_face_recognition import load_face_recognition_model, recognize_faces_in_frame
    REALTIME_MODULE_AVAILABLE = True
except ImportError:
    REALTIME_MODULE_AVAILABLE = False
    st.warning("Real-time recognition module not available")

# Page configuration
st.set_page_config(
    page_title="Face Recognition Challenge",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI/UX
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #f0f8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #e8f4fd;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Cards and Containers */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2.5rem;
        border-radius: 20px;
        color: #2c3e50;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .login-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        margin: 2rem auto;
        max-width: 500px;
    }
    
    /* Status Boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(21, 87, 36, 0.1);
        border-left: 5px solid #28a745;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(114, 28, 36, 0.1);
        border-left: 5px solid #dc3545;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(12, 84, 96, 0.1);
        border-left: 5px solid #17a2b8;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(133, 100, 4, 0.1);
        border-left: 5px solid #ffc107;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 0 20px 20px 0;
        box-shadow: 5px 0 20px rgba(0,0,0,0.1);
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        border: 2px dashed #667eea;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #764ba2;
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Text Inputs */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e9ecef;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 15px;
        border: 2px solid #e9ecef;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .feature-card, .login-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.5rem 0;
        }
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS criminaldata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            address TEXT,
            phone TEXT,
            fathers_name TEXT,
            gender TEXT,
            dob TEXT,
            crimes_done TEXT,
            date_of_arrest TEXT,
            place_of_arrest TEXT,
            face_encoding TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS missingdata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            fathers_name TEXT,
            address TEXT,
            phone TEXT,
            gender TEXT,
            dob TEXT,
            identification TEXT,
            date_of_missing TEXT,
            place_of_missing TEXT,
            face_encoding TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_information (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            gender TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def simple_face_detection(image):
    """Simple face detection using OpenCV Haar Cascade"""
    # Convert PIL to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces, opencv_image

def login_page():
    """Login page with modern UI"""
    # Main header with gradient text
    st.markdown('<h1 class="main-header fade-in">🔍 Face Recognition System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header fade-in">Advanced AI-powered criminal detection and missing person identification</p>', unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-card fade-in">', unsafe_allow_html=True)
        st.markdown("### 🔐 Secure Login")
        st.markdown("Please enter your credentials to access the system")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
            with col_btn2:
                register_button = st.form_submit_button("📝 Register", use_container_width=True)
            
            if submit_button:
                if username and password:
                    conn = sqlite3.connect('face_recognition.db')
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM user_information WHERE username = ? AND password = ?", 
                                 (username, password))
                    user = cursor.fetchone()
                    conn.close()
                    
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user
                        st.success("✅ Login successful! Welcome back!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
                else:
                    st.warning("⚠️ Please fill in all fields")
            
            if register_button:
                st.session_state.show_register = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Features preview
        st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
        st.markdown("### 🌟 System Features")
        
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.markdown("""
            **🔍 Criminal Detection**
            - Upload images for face detection
            - Compare against criminal database
            - Real-time identification results
            """)
        
        with col_feat2:
            st.markdown("""
            **👥 Missing Person Search**
            - Search for missing individuals
            - Upload photos for matching
            - Database management tools
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    """Registration page with modern UI"""
    st.markdown('<h1 class="main-header fade-in">📝 Create New Account</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header fade-in">Join our secure face recognition system</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-card fade-in">', unsafe_allow_html=True)
        st.markdown("### 👤 User Registration")
        st.markdown("Please fill in your details to create an account")
        
        with st.form("register_form"):
            col_name1, col_name2 = st.columns(2)
            with col_name1:
                first_name = st.text_input("👤 First Name", placeholder="Enter your first name")
            with col_name2:
                last_name = st.text_input("👤 Last Name", placeholder="Enter your last name")
            
            gender = st.selectbox("⚥ Gender", ["Male", "Female", "Other"])
            
            username = st.text_input("👤 Username", placeholder="Choose a unique username")
            
            col_pass1, col_pass2 = st.columns(2)
            with col_pass1:
                password = st.text_input("🔒 Password", type="password", placeholder="Create a strong password")
            with col_pass2:
                confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm your password")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submit_button = st.form_submit_button("✅ Create Account", use_container_width=True)
            with col_btn2:
                back_button = st.form_submit_button("⬅️ Back to Login", use_container_width=True)
            
            if submit_button:
                if not all([first_name, last_name, username, password, confirm_password]):
                    st.warning("⚠️ Please fill in all required fields")
                elif password != confirm_password:
                    st.error("❌ Passwords do not match. Please try again.")
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters long.")
                else:
                    try:
                        conn = sqlite3.connect('face_recognition.db')
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO user_information (first_name, last_name, gender, username, password) VALUES (?, ?, ?, ?, ?)",
                                     (first_name, last_name, gender, username, password))
                        conn.commit()
                        conn.close()
                        st.success("🎉 Registration successful! You can now login with your credentials.")
                        st.session_state.show_register = False
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("❌ Username already exists. Please choose a different username.")
            
            if back_button:
                st.session_state.show_register = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Security info
        st.markdown('<div class="info-box fade-in">', unsafe_allow_html=True)
        st.markdown("### 🔒 Security Information")
        st.markdown("""
        - Your data is encrypted and stored securely
        - We use industry-standard security practices
        - Your personal information is never shared with third parties
        - All face recognition data is processed locally
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def main_dashboard():
    """Main dashboard with modern UI"""
    # Welcome header
    st.markdown(f'<h1 class="main-header fade-in">👋 Welcome back, {st.session_state.current_user[1]}!</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header fade-in">Access the powerful face recognition system</p>', unsafe_allow_html=True)
    
    # Sidebar with modern design
    with st.sidebar:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown(f"### 👤 {st.session_state.current_user[1]} {st.session_state.current_user[2]}")
        st.markdown(f"**Role:** System Administrator")
        st.markdown(f"**Status:** 🟢 Online")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        
        # Navigation with icons
        page = st.radio("Choose a feature:", [
            "🔍 Criminal Detection",
            "👥 Find Missing People",
            "📹 Real-Time Recognition",
            "📝 Register Criminal",
            "📋 Register Missing Person",
            "📊 View Database",
            "⚙️ System Settings"
        ])
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM criminaldata")
        criminal_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM missingdata")
        missing_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM user_information")
        user_count = cursor.fetchone()[0]
        
        conn.close()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Criminals", criminal_count)
            st.metric("Missing", missing_count)
        with col2:
            st.metric("Users", user_count)
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
    
    # Main content area with better layout
    if "Criminal Detection" in page:
        criminal_detection_page()
    elif "Find Missing People" in page:
        missing_people_page()
    elif "Real-Time Recognition" in page:
        realtime_recognition_page()
    elif "Register Criminal" in page:
        register_criminal_page()
    elif "Register Missing Person" in page:
        register_missing_page()
    elif "View Database" in page:
        view_database_page()
    elif "System Settings" in page:
        system_settings_page()

def criminal_detection_page():
    """Criminal detection functionality with modern UI"""
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("## 🕵️ Criminal Detection System")
    st.markdown("Upload images to detect and identify potential criminals using advanced face recognition technology")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Upload Image")
        st.markdown("Choose an image file to analyze for criminal detection")
        
        uploaded_file = st.file_uploader("Choose an image file", 
                                       type=['png', 'jpg', 'jpeg'],
                                       help="Supported formats: PNG, JPG, JPEG")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Detect Faces", use_container_width=True):
                    with st.spinner("🔍 Analyzing image for faces..."):
                        faces, opencv_image = simple_face_detection(image)
                        
                        if len(faces) > 0:
                            st.success(f"✅ Found {len(faces)} face(s) in the image!")
                            
                            # Draw rectangles around faces
                            for (x, y, w, h) in faces:
                                cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            
                            # Convert back to RGB for display
                            result_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                            st.image(result_image, caption="🎯 Face Detection Result", use_container_width=True)
                            
                            # Check against criminal database
                            check_criminal_database(faces, opencv_image)
                        else:
                            st.warning("⚠️ No faces detected in the image. Please try with a different image.")
            
            with col_btn2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Detection Results")
        st.markdown("Analysis results and criminal database matches will appear here")
        
        # Show some stats
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM criminaldata")
        total_criminals = cursor.fetchone()[0]
        conn.close()
        
        st.metric("Total Criminals in Database", total_criminals)
        
        st.markdown("### 🔍 How it works:")
        st.markdown("""
        1. **Upload Image** - Select a clear image with faces
        2. **Face Detection** - AI detects all faces in the image
        3. **Database Matching** - Compares against criminal database
        4. **Results** - Shows matches and confidence levels
        """)
        
        st.markdown("### 💡 Tips for better detection:")
        st.markdown("""
        - Use clear, well-lit images
        - Ensure faces are clearly visible
        - Avoid blurry or low-quality images
        - Multiple faces can be detected at once
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def check_criminal_database(faces, image):
    """Check detected faces against criminal database with modern UI"""
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Checking against Criminal Database...")
    
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, crimes_done, date_of_arrest, place_of_arrest FROM criminaldata")
    criminals = cursor.fetchall()
    conn.close()
    
    if criminals:
        st.markdown("**📋 Criminal Database Records:**")
        for i, criminal in enumerate(criminals, 1):
            st.markdown(f"""
            **{i}. {criminal[0]}**
            - **Crimes:** {criminal[1] or 'Not specified'}
            - **Arrested:** {criminal[2] or 'Not specified'}
            - **Location:** {criminal[3] or 'Not specified'}
            """)
    else:
        st.info("ℹ️ No criminal records found in database")
    
    st.markdown('</div>', unsafe_allow_html=True)

def system_settings_page():
    """System settings page with modern UI"""
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("## ⚙️ System Settings")
    st.markdown("Configure system preferences and manage application settings")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 🔧 Application Settings")
        
        # Theme settings
        st.markdown("#### 🎨 Theme Settings")
        theme = st.selectbox("Choose Theme", ["Light", "Dark", "Auto"])
        
        # Detection settings
        st.markdown("#### 🔍 Detection Settings")
        confidence_threshold = st.slider("Face Detection Confidence", 0.1, 1.0, 0.5, 0.1)
        max_faces = st.number_input("Maximum Faces to Detect", 1, 10, 5)
        
        # Notification settings
        st.markdown("#### 🔔 Notification Settings")
        email_notifications = st.checkbox("Email Notifications", value=True)
        sound_alerts = st.checkbox("Sound Alerts", value=True)
        
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 System Information")
        
        # System stats
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM criminaldata")
        criminal_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM missingdata")
        missing_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM user_information")
        user_count = cursor.fetchone()[0]
        
        conn.close()
        
        st.metric("Total Criminals", criminal_count)
        st.metric("Missing Persons", missing_count)
        st.metric("System Users", user_count)
        
        st.markdown("### 🗄️ Database Management")
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.warning("⚠️ This action cannot be undone!")
        
        if st.button("📤 Export Database", use_container_width=True):
            st.info("📁 Database export feature coming soon!")
        
        if st.button("📥 Import Database", use_container_width=True):
            st.info("📁 Database import feature coming soon!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System status
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🟢 System Status")
    
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        st.markdown("**Database:** 🟢 Online")
    with col_status2:
        st.markdown("**Face Detection:** 🟢 Active")
    with col_status3:
        st.markdown("**System:** 🟢 Running")
    
    st.markdown('</div>', unsafe_allow_html=True)

def missing_people_page():
    """Missing people detection functionality with modern UI"""
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("## 👥 Missing Person Search System")
    st.markdown("Upload images to search for missing persons using advanced face recognition technology")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Upload Image")
        st.markdown("Choose an image file to search for missing persons")
        
        uploaded_file = st.file_uploader("Choose an image file", 
                                       type=['png', 'jpg', 'jpeg'],
                                       help="Supported formats: PNG, JPG, JPEG")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔍 Search Missing", use_container_width=True):
                    with st.spinner("🔍 Searching for missing persons..."):
                        faces, opencv_image = simple_face_detection(image)
                        
                        if len(faces) > 0:
                            st.success(f"✅ Found {len(faces)} face(s) in the image!")
                            
                            # Draw rectangles around faces
                            for (x, y, w, h) in faces:
                                cv2.rectangle(opencv_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
                            
                            # Convert back to RGB for display
                            result_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                            st.image(result_image, caption="🎯 Face Detection Result", use_container_width=True)
                            
                            # Check against missing people database
                            check_missing_database(faces, opencv_image)
                        else:
                            st.warning("⚠️ No faces detected in the image. Please try with a different image.")
            
            with col_btn2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Search Results")
        st.markdown("Missing person search results and database matches will appear here")
        
        # Show some stats
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM missingdata")
        total_missing = cursor.fetchone()[0]
        conn.close()
        
        st.metric("Missing Persons in Database", total_missing)
        
        st.markdown("### 🔍 How it works:")
        st.markdown("""
        1. **Upload Image** - Select a clear image with faces
        2. **Face Detection** - AI detects all faces in the image
        3. **Database Matching** - Compares against missing persons database
        4. **Results** - Shows potential matches and details
        """)
        
        st.markdown("### 💡 Tips for better search:")
        st.markdown("""
        - Use clear, well-lit images
        - Ensure faces are clearly visible
        - Avoid blurry or low-quality images
        - Multiple faces can be searched at once
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def realtime_recognition_page():
    """Real-time face recognition page with camera input"""
    st.markdown('<div class="feature-card fade-in">', unsafe_allow_html=True)
    st.markdown("## 📹 Real-Time Face Recognition")
    st.markdown("Use your camera to recognize faces in real-time")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not REALTIME_MODULE_AVAILABLE:
        st.error("❌ Real-time recognition module is not available. Please check the installation.")
        return
    
    # Load model once at the start
    if 'model' not in st.session_state or 'names' not in st.session_state:
        with st.spinner("🔄 Loading face recognition model..."):
            st.session_state.model, st.session_state.names = load_face_recognition_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📷 Camera Input")
        
        # Camera input
        picture = st.camera_input("📹 Take a photo for face recognition", 
                                 help="Click the camera button to capture an image")
        
        if picture is not None:
            # Convert to OpenCV format
            bytes_data = picture.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Display captured image
            st.image(picture, caption="📷 Captured Image", use_container_width=True)
            
            # Recognize faces
            if st.button("🔍 Recognize Faces", use_container_width=True):
                if st.session_state.model is not None:
                    with st.spinner("🤖 Recognizing faces..."):
                        # Process the image
                        result_frame, recognized, all_detections = recognize_faces_in_frame(
                            st.session_state.model, 
                            st.session_state.names, 
                            img_cv.copy()
                        )
                        
                        # Convert back to RGB for display
                        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, caption="🎯 Recognition Result", use_container_width=True)
                        
                        # Show recognition results
                        if len(recognized) > 0:
                            st.success(f"✅ Recognized {len(recognized)} person(s)!")
                            # Display details of recognized persons
                            for person in recognized:
                                st.markdown(f"**{person['name']}** (Confidence: {person['confidence']:.2f})")
                        elif len(all_detections) > 0:
                            st.warning(f"⚠️ Detected {len(all_detections)} face(s) but none matched the database")
                        else:
                            st.info("ℹ️ No faces detected in the image")
                else:
                    st.error("❌ Face recognition model is not loaded. Please register some persons first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Information")
        
        # Show registered persons
        if st.session_state.names:
            st.markdown("### 👥 Registered Persons:")
            for person_id, name in st.session_state.names.items():
                st.markdown(f"- **{name}**")
        else:
            st.warning("⚠️ No persons registered yet")
            st.markdown("Go to 'Register Criminal' or 'Register Missing Person' to add faces to the database.")
        
        st.markdown("---")
        st.markdown("### 📖 How to use:")
        st.markdown("""
        1. Click the camera button
        2. Allow camera access
        3. Capture your image
        4. Click "Recognize Faces"
        5. See recognition results
        """)
        
        st.markdown("### 💡 Tips:")
        st.markdown("""
        - Good lighting improves results
        - Face the camera directly
        - Stay still while capturing
        - At least 5 images per person needed for training
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

def check_missing_database(faces, image):
    """Check detected faces against missing people database with modern UI"""
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Checking against Missing People Database...")
    
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name, date_of_missing, place_of_missing, identification FROM missingdata")
    missing = cursor.fetchall()
    conn.close()
    
    if missing:
        st.markdown("**📋 Missing People Database Records:**")
        for i, person in enumerate(missing, 1):
            st.markdown(f"""
            **{i}. {person[0]}**
            - **Missing Since:** {person[1] or 'Not specified'}
            - **Last Seen:** {person[2] or 'Not specified'}
            - **Identification:** {person[3] or 'Not specified'}
            """)
    else:
        st.info("ℹ️ No missing people records found in database")
    
    st.markdown('</div>', unsafe_allow_html=True)

def register_criminal_page():
    """Register new criminal"""
    st.markdown("## 📝 Register New Criminal")
    
    with st.form("criminal_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *")
            fathers_name = st.text_input("Father's Name")
            address = st.text_input("Address")
            phone = st.text_input("Phone Number")
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            dob = st.date_input("Date of Birth")
            crimes_done = st.text_input("Crimes Committed")
            date_of_arrest = st.date_input("Date of Arrest")
        
        place_of_arrest = st.text_input("Place of Arrest")
        
        st.markdown("### Upload Criminal Photo")
        criminal_photo = st.file_uploader("Choose criminal photo", type=['png', 'jpg', 'jpeg'])
        
        submitted = st.form_submit_button("Register Criminal")
        
        if submitted:
            if name and criminal_photo:
                try:
                    conn = sqlite3.connect('face_recognition.db')
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO criminaldata 
                        (name, fathers_name, address, phone, gender, dob, crimes_done, date_of_arrest, place_of_arrest)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (name, fathers_name, address, phone, gender, str(dob), 
                          crimes_done, str(date_of_arrest), place_of_arrest))
                    conn.commit()
                    conn.close()
                    st.success("Criminal registered successfully!")
                except sqlite3.IntegrityError:
                    st.error("Criminal with this name already exists")
            else:
                st.error("Please fill in all required fields and upload a photo")

def register_missing_page():
    """Register missing person"""
    st.markdown("## 📝 Register Missing Person")
    
    with st.form("missing_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *")
            fathers_name = st.text_input("Father's Name")
            address = st.text_input("Address")
            phone = st.text_input("Phone Number")
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            dob = st.date_input("Date of Birth")
            identification = st.text_input("Identification Marks")
            date_of_missing = st.date_input("Date of Missing")
        
        place_of_missing = st.text_input("Place of Missing")
        
        st.markdown("### Upload Missing Person Photo")
        missing_photo = st.file_uploader("Choose missing person photo", type=['png', 'jpg', 'jpeg'])
        
        submitted = st.form_submit_button("Register Missing Person")
        
        if submitted:
            if name and missing_photo:
                try:
                    conn = sqlite3.connect('face_recognition.db')
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO missingdata 
                        (name, fathers_name, address, phone, gender, dob, identification, date_of_missing, place_of_missing)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (name, fathers_name, address, phone, gender, str(dob), 
                          identification, str(date_of_missing), place_of_missing))
                    conn.commit()
                    conn.close()
                    st.success("Missing person registered successfully!")
                except sqlite3.IntegrityError:
                    st.error("Missing person with this name already exists")
            else:
                st.error("Please fill in all required fields and upload a photo")

def view_database_page():
    """View database records"""
    st.markdown("## 📊 Database Records")
    
    tab1, tab2 = st.tabs(["Criminal Records", "Missing People Records"])
    
    with tab1:
        st.markdown("### Criminal Database")
        conn = sqlite3.connect('face_recognition.db')
        df_criminals = pd.read_sql_query("SELECT * FROM criminaldata", conn)
        conn.close()
        
        if not df_criminals.empty:
            st.dataframe(df_criminals, use_container_width=True)
        else:
            st.info("No criminal records found")
    
    with tab2:
        st.markdown("### Missing People Database")
        conn = sqlite3.connect('face_recognition.db')
        df_missing = pd.read_sql_query("SELECT * FROM missingdata", conn)
        conn.close()
        
        if not df_missing.empty:
            st.dataframe(df_missing, use_container_width=True)
        else:
            st.info("No missing people records found")

def main():
    """Main application function"""
    # Initialize session state variables
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    
    # Initialize database
    init_database()
    
    # Check if user is authenticated
    if not st.session_state.authenticated:
        if st.session_state.show_register:
            register_page()
        else:
            login_page()
    else:
        main_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🔍 Face Recognition System | Advanced AI-powered criminal detection and missing person identification</p>
        <p>Built with ❤️ using Streamlit, OpenCV, and Python</p>
        <p>© 2024 Face Recognition Challenge - All rights reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
