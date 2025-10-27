#!/usr/bin/env python3
"""
Real-time Face Recognition Module
Provides functions for real-time face detection and recognition
"""

import cv2
import numpy as np
import os

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_face_recognition_model():
    """
    Load the trained face recognition model
    Returns: (model, names) tuple
    """
    try:
        model = cv2.face.LBPHFaceRecognizer_create()
        fn_dir = 'face_samples'
        
        images, labels, names = [], [], {}
        person_id = 0
        
        # Walk through face_samples directory
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                names[person_id] = subdir  # Store person name
                subjectpath = os.path.join(fn_dir, subdir)
                
                # Load all images for this person
                for filename in os.listdir(subjectpath):
                    f_name, f_extension = os.path.splitext(filename)
                    
                    # Skip non-image files
                    if f_extension.lower() not in ['.png', '.jpg', '.jpeg', '.pgm']:
                        continue
                    
                    path = os.path.join(subjectpath, filename)
                    img = cv2.imread(path, 0)  # Read as grayscale
                    
                    if img is not None:
                        images.append(img)
                        labels.append(person_id)
                
                person_id += 1
        
        # Train the model
        if len(images) > 0:
            images = np.array(images)
            labels = np.array(labels)
            model.train(images, labels)
            return (model, names)
        else:
            return (None, {})
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return (None, {})


def recognize_faces_in_frame(model, names, frame):
    """
    Recognize faces in a video frame
    
    Args:
        model: Trained face recognition model
        names: Dictionary mapping person IDs to names
        frame: Video frame (BGR format)
    
    Returns:
        (frame_with_boxes, recognized_persons)
    """
    if model is None:
        return frame, []
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with more sensitive parameters
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    recognized_persons = []
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to model input size
        face_resized = cv2.resize(face_roi, (112, 92))
        
        # Predict
        try:
            label, confidence = model.predict(face_resized)
            print(f"Prediction: label={label}, confidence={confidence}, names={names}")
            
            # Lower confidence = better match (LBPH returns 0 for perfect match)
            # Use more lenient threshold for better recognition
            if confidence < 100 and label in names:
                person_name = names[label]
                recognized_persons.append({
                    'name': person_name,
                    'confidence': confidence,
                    'position': (x, y, w, h)
                })
                
                # Draw green box for recognized person
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Add name label
                cv2.putText(frame, person_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Draw red box for unrecognized person
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Unknown', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error predicting: {e}")
            # Draw yellow box for error
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    
    return frame, recognized_persons


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

