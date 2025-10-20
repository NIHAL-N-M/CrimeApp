# webcam.py
import cv2
import tkinter as tk
from tkinter import messagebox
import threading
import time

def web():
    """Webcam function for live face detection"""
    try:
        # Try to open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam. Please check if webcam is connected.")
            return
            
        # Create window for webcam
        cv2.namedWindow('Live Face Detection', cv2.WINDOW_NORMAL)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            messagebox.showerror("Error", "Could not load face cascade classifier.")
            cap.release()
            return
        
        print("Webcam started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display frame
            cv2.imshow('Live Face Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

