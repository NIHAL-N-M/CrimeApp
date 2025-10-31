import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
import pymysql
import pandas as pd


@st.cache_resource
def load_cascade():
	candidates = [
		"haarcascade_frontalface_default.xml",
		"face_cascade.xml",
	]
	for name in candidates:
		if os.path.exists(name):
			c = cv2.CascadeClassifier(name)
			if not c.empty():
				return c
	data = getattr(cv2.data, "haarcascades", None)
	if data:
		c = cv2.CascadeClassifier(os.path.join(data, "haarcascade_frontalface_default.xml"))
		if not c.empty():
			return c
	return None


@st.cache_resource
def train_criminal_lbph_model():
	# Train LBPH on face_samples (criminal dataset) to match Criminal Detection System
	dataset_dir = "face_samples"
	if not os.path.isdir(dataset_dir):
		return None, {}
	# Ensure OpenCV contrib (cv2.face) is available
	if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
		return None, {}
        model = cv2.face.LBPHFaceRecognizer_create()
	images = []
	labels = []
        names = {}
	label_id = 0
	valid_exts = {".png", ".jpg", ".jpeg", ".pgm"}
	for subdir in sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]):
		person_dir = os.path.join(dataset_dir, subdir)
		files = [f for f in os.listdir(person_dir) if os.path.splitext(f)[1].lower() in valid_exts]
		if not files:
			continue
		names[label_id] = subdir
		for f in files:
			path = os.path.join(person_dir, f)
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue
			img = cv2.resize(img, (112, 92))
			images.append(img)
			labels.append(label_id)
		label_id += 1
	if not images:
		return None, {}
	model.train(np.array(images), np.array(labels))
	return model, names


def recognize_frame_with_model(frame_bgr, face_detector, model, names, confidence_threshold=80.0):
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


def image_mode(face_detector):
	uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
	if uploaded is None:
		return
	file_bytes = np.frombuffer(uploaded.read(), np.uint8)
	img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	if img is None:
		st.error("Could not read image.")
		return
	st.info("Detecting faces (no recognition in this mode)...")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)


def real_time_recognition(face_detector):
	with st.spinner("Training LBPH model on criminal dataset (face_samples)..."):
		model, names = train_criminal_lbph_model()
	if model is None:
		st.error("LBPH model (cv2.face) is unavailable. Ensure opencv-contrib-python is installed and not overridden by opencv-python.")
		return
	if not names:
		st.error("No training data found in 'face_samples'. Ensure it contains subfolders per person with face images.")
		return
	# Prefer browser camera
	image_from_camera = None
	try:
		image_from_camera = st.camera_input("Take a photo for recognition")
	except Exception:
		image_from_camera = None
	if image_from_camera is not None:
		file_bytes = np.frombuffer(image_from_camera.getvalue(), np.uint8)
		img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		if img is None:
			st.error("Could not decode captured image.")
			return
		out, rec = recognize_frame_with_model(img, face_detector, model, names)
		st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
		if rec:
			st.success("Detected: " + ", ".join({r[0] for r in rec}))
                    else:
			st.info("No known person recognized.")
		return
	# Fallback to local webcam preview and shoot
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		st.error("Webcam not available. Try using the browser camera above if supported.")
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
		out, rec = recognize_frame_with_model(frame_to_eval, face_detector, model, names)
		st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
		if rec:
			st.success("Detected: " + ", ".join({r[0] for r in rec}))
                else:
			st.info("No known person recognized.")


def main():
	st.set_page_config(page_title="Criminal Detection - Streamlit", layout="wide")
	st.title("Criminal Detection - Real-Time Recognition")
	st.caption("Uses the same LBPH model trained on 'face_samples' as the desktop app.")
	face_detector = load_cascade()
	if face_detector is None or face_detector.empty():
		st.error("Failed to load Haar cascade. Make sure the XML file is present.")
		return

	st.sidebar.subheader("Modes")
	mode = st.sidebar.radio("Mode", ("Real-Time Recognition", "Image (Detect Only)", "Criminal Database"))
	if mode == "Real-Time Recognition":
		real_time_recognition(face_detector)
	else:
		if mode == "Image (Detect Only)":
			image_mode(face_detector)
                else:
			criminal_database_ui()


def criminal_database_ui():
	st.subheader("Criminal Database (MySQL)")
        col1, col2 = st.columns(2)
        with col1:
		host = st.text_input("Host", value="localhost")
		user = st.text_input("User", value="root")
		password = st.text_input("Password", value="", type="password")
		database = st.text_input("Database", value="criminaldb")
		connect = st.button("Connect")
	
	if 'db_params' not in st.session_state:
		st.session_state.db_params = None

	if connect:
		st.session_state.db_params = dict(host=host, user=user, password=password, database=database)

	params = st.session_state.db_params
	if not params:
		st.info("Enter connection details and click Connect to view records.")
                return
            
	try:
		conn = pymysql.connect(**params)
        except Exception as e:
		st.error(f"Could not connect to database: {e}")
            return
    
	with conn:
		try:
			df = pd.read_sql("SELECT `Criminal-ID` as criminal_id, `Name` as name, `Address` as address, `Phone` as phone FROM criminaldata", conn)
        except Exception as e:
			st.error(f"Failed to load records. Ensure you imported table.sql. Error: {e}")
        return
    
	st.success(f"Loaded {len(df)} records from criminaldata")
	st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
	st.subheader("Add Training Images to face_samples")
	st.caption("Pick a criminal name (or type a new one) and upload face images to improve recognition.")

	existing_names = sorted([d for d in os.listdir('face_samples')]) if os.path.isdir('face_samples') else []
	name_choice = st.selectbox("Select existing name", options=[""] + existing_names)
	new_name = st.text_input("Or enter a new name")
	final_name = new_name.strip() if new_name.strip() else name_choice.strip()
	uploads = st.file_uploader("Upload face images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
	save_btn = st.button("Save Images")

	if save_btn:
		if not final_name:
			st.error("Please select or enter a name.")
			return
		if not uploads:
			st.error("Please upload at least one image.")
			return
		person_dir = os.path.join('face_samples', final_name)
		os.makedirs(person_dir, exist_ok=True)
		saved = 0
		for i, up in enumerate(uploads, start=1):
			data = np.frombuffer(up.read(), np.uint8)
			img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue
			img = cv2.resize(img, (112, 92))
			cv2.imwrite(os.path.join(person_dir, f"{int(time.time())}_{i}.png"), img)
			saved += 1
		st.success(f"Saved {saved} image(s) to {person_dir}")


