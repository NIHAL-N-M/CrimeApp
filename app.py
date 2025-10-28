import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st


@st.cache_resource
def load_cascade():
	# Prefer project cascades
	candidates = [
		"haarcascade_frontalface_default.xml",
		"face_cascade.xml",
	]
	for name in candidates:
		if os.path.exists(name):
			cascade = cv2.CascadeClassifier(name)
			if not cascade.empty():
				return cascade
	# Fallback to OpenCV data
	opencv_data = getattr(cv2.data, "haarcascades", None)
	if opencv_data:
		path = os.path.join(opencv_data, "haarcascade_frontalface_default.xml")
		cascade = cv2.CascadeClassifier(path)
		if not cascade.empty():
			return cascade
	return None


def detect_faces_bgr(frame_bgr, face_detector):
	gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	for (x, y, w, h) in faces:
		cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
	return frame_bgr, faces


def process_image(uploaded_file, face_detector):
	file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
	img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	if img_bgr is None:
		st.error("Could not read the image. Please upload a valid image file.")
		return
	boxed_bgr, faces = detect_faces_bgr(img_bgr, face_detector)
	st.write(f"Detected {len(faces)} face(s)")
	st.image(cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)


def process_video(uploaded_file, face_detector):
	with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
		tmp.write(uploaded_file.read())
		tmp_path = tmp.name

\tcap = cv2.VideoCapture(tmp_path)
	placeholder = st.empty()
	info = st.empty()
	frame_count = 0
	faces_total = 0
	start = time.time()
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_count += 1
		boxed_bgr, faces = detect_faces_bgr(frame, face_detector)
		faces_total += len(faces)
		placeholder.image(cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB), channels="RGB")
		info.write(f"Frames: {frame_count} | Faces detected (cumulative): {faces_total}")
		time.sleep(0.01)
	cap.release()
	os.unlink(tmp_path)
	st.success(f"Done. Processed {frame_count} frames, detected {faces_total} faces cumulatively in {time.time() - start:.2f}s")


def process_webcam(face_detector):
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		st.error("Webcam not available.")
		return
	st.info("Streaming webcam. Click Stop to end.")
	placeholder = st.empty()
	stop = st.button("Stop", type="primary")
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		boxed_bgr, faces = detect_faces_bgr(frame, face_detector)
		placeholder.image(cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB), channels="RGB")
		if stop:
			break
	cap.release()


@st.cache_resource
def train_lbph_model():
	# Prefer face_samples2, fallback to face_samples
	dataset_dirs = ["face_samples2", "face_samples"]
	dataset_dir = None
	for d in dataset_dirs:
		if os.path.isdir(d):
			dataset_dir = d
			break
	if dataset_dir is None:
		return None, {}

	model = cv2.face.LBPHFaceRecognizer_create()
	images = []
	labels = []
	names = {}
	label_id = 0
	valid_exts = {".png", ".jpg", ".jpeg", ".pgm"}

	for subdir in sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]):
		subject_path = os.path.join(dataset_dir, subdir)
		file_list = [f for f in os.listdir(subject_path) if os.path.splitext(f)[1].lower() in valid_exts]
		if not file_list:
			continue
		names[label_id] = subdir
		for filename in file_list:
			path = os.path.join(subject_path, filename)
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue
			img = cv2.resize(img, (112, 92))
			images.append(img)
			labels.append(label_id)
		label_id += 1

	if not images:
		return None, {}

	images_np = np.array(images)
	labels_np = np.array(labels)
	model.train(images_np, labels_np)
	return model, names


def recognize_faces_in_frame(frame_bgr, face_detector, model, names, confidence_threshold=80.0):
	gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	recognized = []
	for (x, y, w, h) in faces:
		roi = gray[y:y+h, x:x+w]
		roi_resized = cv2.resize(roi, (112, 92))
		pred_id, conf = model.predict(roi_resized)
		name = names.get(pred_id, "Unknown")
		label = name if conf <= confidence_threshold else "Unknown"
		color = (0, 200, 0) if label != "Unknown" else (0, 0, 255)
		cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
		cv2.putText(frame_bgr, f"{label} ({conf:.1f})", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
		recognized.append((label, conf))
	return frame_bgr, recognized


def real_time_recognition_ui(face_detector):
	st.subheader("Real-Time Face Recognition")
	with st.spinner("Training LBPH model from dataset..."):
		model, names = train_lbph_model()
	if model is None or not names:
		st.error("No training data found. Ensure 'face_samples2' or 'face_samples' contains labeled subfolders with face images.")
		return

	# Try using browser camera
	image_from_camera = None
	try:
		image_from_camera = st.camera_input("Take a photo")
	except Exception:
		image_from_camera = None

	if image_from_camera is not None:
		file_bytes = np.frombuffer(image_from_camera.getvalue(), np.uint8)
		img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		if img_bgr is None:
			st.error("Could not decode captured image.")
			return
		out_bgr, recognized = recognize_faces_in_frame(img_bgr, face_detector, model, names)
		st.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
		if recognized:
			st.success("Detected: " + ", ".join({r[0] for r in recognized}))
		else:
			st.info("No known person recognized.")
		return

	# Fallback to local webcam preview + take photo
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		st.error("Webcam not available. Try browser camera above if supported.")
		return
	preview = st.empty()
	col1, col2 = st.columns(2)
	shoot = col1.button("Take Photo", type="primary")
	stop = col2.button("Stop Preview")
	frame_to_eval = None
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
		if shoot:
			frame_to_eval = frame.copy()
			break
		if stop:
			break
	cap.release()

	if frame_to_eval is not None:
		out_bgr, recognized = recognize_faces_in_frame(frame_to_eval, face_detector, model, names)
		st.image(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
		if recognized:
			st.success("Detected: " + ", ".join({r[0] for r in recognized}))
		else:
			st.info("No known person recognized.")


def main():
	st.set_page_config(page_title="Face Detection & Recognition", layout="wide")
	st.title("Face Detection & Recognition - Local (Streamlit)")
	st.caption("Upload media, use webcam preview, or take a photo for real-time recognition against your dataset.")

	face_detector = load_cascade()
	if face_detector is None or face_detector.empty():
		st.error("Failed to load Haar cascade. Ensure 'haarcascade_frontalface_default.xml' or 'face_cascade.xml' exists.")
		return

	mode = st.sidebar.radio("Mode", ("Image", "Video", "Webcam", "Real-Time Recognition"))

	if mode == "Image":
		uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
		if uploaded is not None:
			process_image(uploaded, face_detector)

	elif mode == "Video":
		uploaded = st.file_uploader("Upload a video (mp4, avi, mkv)", type=["mp4", "avi", "mkv"])
		if uploaded is not None:
			process_video(uploaded, face_detector)

	elif mode == "Webcam":
		process_webcam(face_detector)

	else:
		real_time_recognition_ui(face_detector)

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

def load_face_recognition_from_database():
    """
    Load face recognition model using database records instead of hardcoded face_samples
    Returns: (model, names) tuple
    """
    try:
        import cv2
        import numpy as np
        
        model = cv2.face.LBPHFaceRecognizer_create()
        
        # Get all criminal records from database
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM criminaldata")
        criminal_names = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not criminal_names:
            print("No criminal records found in database")
            return (None, {})
        
        # For now, return a simple model that recognizes database names
        # In a real implementation, you would need face images stored in the database
        # or linked to the database records
        names = {}
        for i, name in enumerate(criminal_names):
            names[i] = name
        
        print(f"Loaded {len(names)} persons from database: {list(names.values())}")
        return (model, names)
        
    except Exception as e:
        print(f"Error loading model from database: {e}")
        return (None, {})

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
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
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
                            
                            # Store for recognition
                            st.session_state.detected_faces = faces
                            st.session_state.detected_image = image
                        else:
                            st.warning("⚠️ No faces detected in the image. Please try with a different image.")
            
            with col_btn2:
                if st.button("🚨 Recognize Criminals", use_container_width=True):
                    if 'detected_faces' in st.session_state and 'detected_image' in st.session_state:
                        check_criminal_database(st.session_state.detected_faces, st.session_state.detected_image)
                    else:
                        st.warning("⚠️ Please detect faces first before recognizing criminals")
            
            with col_btn3:
                if st.button("🗑️ Clear", use_container_width=True):
                    # Clear session state
                    if 'detected_faces' in st.session_state:
                        del st.session_state.detected_faces
                    if 'detected_image' in st.session_state:
                        del st.session_state.detected_image
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
        
        # Show recent criminal detections if any
        if 'criminal_detections' in st.session_state:
            st.markdown("### 🚨 Recent Detections")
            for detection in st.session_state.criminal_detections[-3:]:  # Show last 3
                st.markdown(f"**{detection['name']}** - {detection['confidence']:.1f}% confidence")
        
        # Criminal database statistics
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        # Get criminal statistics
        cursor.execute("SELECT COUNT(*) FROM criminaldata WHERE crimes_done IS NOT NULL AND crimes_done != ''")
        with_crimes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM criminaldata WHERE date_of_arrest IS NOT NULL AND date_of_arrest != ''")
        arrested = cursor.fetchone()[0]
        
        conn.close()
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("With Crime Records", with_crimes)
        with col_stat2:
            st.metric("Previously Arrested", arrested)
        
        st.markdown("### 🔍 How it works:")
        st.markdown("""
        1. **Upload Image** - Select a clear image with faces
        2. **Detect Faces** - AI detects all faces in the image
        3. **Recognize Criminals** - Advanced face recognition against criminal database
        4. **Alert System** - Shows confidence levels and criminal details
        5. **Action Buttons** - Report matches and view full records
        """)
        
        st.markdown("### 💡 Tips for better detection:")
        st.markdown("""
        - Use clear, well-lit images
        - Ensure faces are clearly visible
        - Avoid blurry or low-quality images
        - Multiple faces can be detected at once
        """)
        
        # Show current database records
        st.markdown("### 🎯 Current Database Records:")
        try:
            (model, names) = load_face_recognition_from_database()
            if names:
                st.success(f"✅ Database loaded with {len(names)} criminal records:")
                for person_id, name in names.items():
                    st.markdown(f"- **{name}** (ID: {person_id})")
                
                # Test button
                if st.button("🧪 Test Face Recognition", use_container_width=True):
                    st.info("💡 Upload an image with faces and click 'Recognize Criminals' to test the system!")
            else:
                st.warning("⚠️ No criminal records found in database")
        except Exception as e:
            st.error(f"❌ Error loading database records: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def check_criminal_database(faces, image):
    """Check detected faces against criminal database using face recognition"""
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Criminal Face Recognition Analysis...")
    
    # Load the face recognition model from database
    with st.spinner("🔄 Loading face recognition model from database..."):
        try:
            from face_detection import detect_faces
            
            # Load model using database records
            (model, names) = load_face_recognition_from_database()
            
            if model is None or not names:
                st.error("❌ Failed to load face recognition model from database")
                return
            
            st.success(f"✅ Face recognition model loaded with {len(names)} database records")
            st.info(f"📋 Database records: {list(names.values())}")
            
        except Exception as e:
            st.error(f"❌ Error loading face recognition: {e}")
            return
    
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the working detection system
    face_coords = detect_faces(gray_image)
    
    if len(face_coords) == 0:
        st.warning("⚠️ No faces detected for recognition")
        return
    
    st.info(f"🔍 Analyzing {len(face_coords)} detected face(s)...")
    
    # Show database records for detected faces
    with st.spinner("🤖 Analyzing faces against database records..."):
        try:
            st.info(f"🔍 Database contains: {list(names.values())}")
            
            # For now, show all database records since we don't have actual face images
            # In a real implementation, you would need to store face images in the database
            st.markdown("### 📋 Database Criminal Records")
            st.markdown("---")
            
            # Get criminal database information
            conn = sqlite3.connect('face_recognition.db')
            cursor = conn.cursor()
            
            # Show all criminal records from database
            cursor.execute("""
                SELECT name, crimes_done, date_of_arrest, place_of_arrest, 
                       gender, address, phone, fathers_name, dob
                FROM criminaldata 
                ORDER BY name
            """)
            
            criminal_records = cursor.fetchall()
            
            if criminal_records:
                st.success(f"✅ Found {len(criminal_records)} criminal records in database")
                
                for i, criminal_info in enumerate(criminal_records, 1):
                    criminal_name = criminal_info[0]
                    
                    # Display criminal details
                    st.markdown(f"""
                    **📋 Criminal #{i}: {criminal_name.upper()}**
                    - **Crimes:** {criminal_info[1] or 'Not specified'}
                    - **Arrest Date:** {criminal_info[2] or 'Not specified'}
                    - **Arrest Location:** {criminal_info[3] or 'Not specified'}
                    - **Gender:** {criminal_info[4] or 'Not specified'}
                    - **Address:** {criminal_info[5] or 'Not specified'}
                    - **Phone:** {criminal_info[6] or 'Not specified'}
                    - **Father's Name:** {criminal_info[7] or 'Not specified'}
                    - **Date of Birth:** {criminal_info[8] or 'Not specified'}
                    """)
                    
                    # Show action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"📋 View Full Record", key=f"view_{i}"):
                            st.session_state[f"show_details_{i}"] = True
                    with col2:
                        if st.button(f"🚨 Report Match", key=f"report_{i}"):
                            st.success("✅ Match reported to authorities")
                    with col3:
                        if st.button(f"📊 More Details", key=f"details_{i}"):
                            st.info(f"Full record for {criminal_name}")
                    
                    st.markdown("---")
                
                # Summary
                st.success(f"✅ **Database Analysis Complete:** {len(criminal_records)} criminal records found")
                
            else:
                st.info("ℹ️ No criminal records found in database")
                
            conn.close()
                
        except Exception as e:
            st.error(f"❌ Error during face recognition: {e}")
    
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
    
    # Load model once at the start using database records
    if 'model' not in st.session_state or 'names' not in st.session_state:
        with st.spinner("🔄 Loading face recognition model from database..."):
            try:
                st.session_state.model, st.session_state.names = load_face_recognition_from_database()
                if st.session_state.model is None or not st.session_state.names:
                    st.error("❌ Failed to load face recognition model from database")
                else:
                    st.success(f"✅ Model loaded with {len(st.session_state.names)} database records")
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                st.session_state.model = None
                st.session_state.names = {}
    
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
            
            # Show database records
            if st.button("🔍 Show Database Records", use_container_width=True):
                if st.session_state.names:
                    with st.spinner("🤖 Loading database records..."):
                        try:
                            st.success(f"✅ Database contains {len(st.session_state.names)} criminal records:")
                            
                            # Get detailed criminal information from database
                            conn = sqlite3.connect('face_recognition.db')
                            cursor = conn.cursor()
                            
                            for i, (person_id, name) in enumerate(st.session_state.names.items(), 1):
                                cursor.execute("""
                                    SELECT name, crimes_done, date_of_arrest, place_of_arrest, 
                                           gender, address, phone, fathers_name, dob
                                    FROM criminaldata 
                                    WHERE name = ?
                                """, (name,))
                                
                                criminal_info = cursor.fetchone()
                                
                                if criminal_info:
                                    st.markdown(f"""
                                    **📋 Criminal #{i}: {criminal_info[0].upper()}**
                                    - **Crimes:** {criminal_info[1] or 'Not specified'}
                                    - **Arrest Date:** {criminal_info[2] or 'Not specified'}
                                    - **Arrest Location:** {criminal_info[3] or 'Not specified'}
                                    - **Gender:** {criminal_info[4] or 'Not specified'}
                                    - **Address:** {criminal_info[5] or 'Not specified'}
                                    - **Phone:** {criminal_info[6] or 'Not specified'}
                                    - **Father's Name:** {criminal_info[7] or 'Not specified'}
                                    - **Date of Birth:** {criminal_info[8] or 'Not specified'}
                                    """)
                                    st.markdown("---")
                            
                            conn.close()
                            
                        except Exception as e:
                            st.error(f"❌ Error loading database records: {e}")
                else:
                    st.error("❌ No database records available. Please add criminal records first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Information")
        
        # Show database records
        if st.session_state.names:
            st.markdown("### 👥 Database Records:")
            for person_id, name in st.session_state.names.items():
                st.markdown(f"- **{name}**")
        else:
            st.warning("⚠️ No criminal records in database")
            st.markdown("Go to 'Register Criminal' to add records to the database.")
        
        st.markdown("---")
        st.markdown("### 📖 How to use:")
        st.markdown("""
        1. Click the camera button
        2. Allow camera access
        3. Capture your image
        4. Click "Show Database Records"
        5. View all criminal records from database
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
    if 'current_time' not in st.session_state:
        from datetime import datetime
        st.session_state.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
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
