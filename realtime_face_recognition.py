#!/usr/bin/env python3
"""
Real-time Face Recognition Module
Provides functions for real-time face detection and recognition
Uses the original working face recognition system
"""

import cv2
import numpy as np
import os

# Use the original face cascade that was working
face_cascade = cv2.CascadeClassifier('face_cascade.xml')

def load_face_recognition_model():
    """
    Load the trained face recognition model using the original working system
    Returns: (model, names) tuple
    """
    try:
        model = cv2.face.LBPHFaceRecognizer_create()
        fn_dir = 'face_samples'
        
        # Use the original training method that was working
        (images, labels, names, id) = ([], [], {}, 0)
        
        # Walk through face_samples directory (same as original)
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                names[id] = subdir  # Store person name
                subjectpath = os.path.join(fn_dir, subdir)
                
                # Load all images for this person (using color images like original)
                for filename in os.listdir(subjectpath):
                    f_name, f_extension = os.path.splitext(filename)
                    
                    # Skip non-image files
                    if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.pgm']:
                        print("Skipping " + filename + ", wrong file type")
                        continue
                    
                    path = os.path.join(subjectpath, filename)
                    img = cv2.imread(path)  # Read as color image (like original)
                    
                    if img is not None:
                        images.append(img)
                        labels.append(id)
                id += 1
        
        # Train the model (same as original)
        if len(images) > 0:
            images = np.array(images)
            labels = np.array(labels)
            model.train(images, labels)
            print(f"Model trained with {len(images)} images for {len(names)} persons")
            print(f"Names dictionary: {names}")
            return (model, names)
        else:
            print("No training images found")
            return (None, {})
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return (None, {})


def recognize_faces_in_frame(model, names, frame):
    """
    Recognize faces in a video frame using the original working system
    
    Args:
        model: Trained face recognition model
        names: Dictionary mapping person IDs to names
        frame: Video frame (BGR format)
    
    Returns:
        (frame_with_boxes, recognized_persons, all_detections)
    """
    if model is None:
        return frame, [], []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use original face detection parameters
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    recognized_persons = []
    all_detections = []
    
    # Process each detected face using original recognition logic
    for (x, y, w, h) in faces:
        all_detections.append({'position': (x, y, w, h)})
        
        # Extract face region (same as original)
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to model input size (same as original)
        face_resized = cv2.resize(face_roi, (112, 92))
        
        # Predict using original method
        try:
            (prediction, confidence) = model.predict(face_resized)
            print(f"Prediction: label={prediction}, confidence={confidence}, names={names}")
            
            # Use original confidence threshold (< 95)
            if confidence < 95 and prediction in names:
                person_name = names[prediction]
                recognized_persons.append({
                    'name': person_name.capitalize(),
                    'confidence': confidence,
                    'position': (x, y, w, h)
                })
                
                # Draw green box for recognized person (same as original)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Add name label
                cv2.putText(frame, person_name.capitalize(), (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Draw red box for unrecognized person (same as original)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Unknown', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                print(f"Face detected but not recognized: confidence={confidence}, threshold=95")
        except Exception as e:
            print(f"Error predicting: {e}")
            # Draw yellow box for error
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    return frame, recognized_persons, all_detections


def detect_faces_simple(frame):
    """
    Simple face detection without recognition
    Returns: (frame_with_boxes, face_count)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return frame, len(faces)

