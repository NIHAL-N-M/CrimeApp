# home2_simple.py - Simplified version for missing people detection
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import os
from face_detection import detect_faces

def home2():
    """Simplified missing people detection interface"""
    root = tk.Toplevel()
    root.title("Find Missing People")
    root.geometry("800x600")
    root.configure(bg="#051729")
    
    # Title
    title_label = tk.Label(root, text="Find Missing People System", 
                          font="Helvetica 20 bold", fg="white", bg="#051729")
    title_label.pack(pady=20)
    
    # Instructions
    instructions = tk.Label(root, text="This feature helps find missing people using face recognition", 
                           font="Verdana 12", fg="white", bg="#051729")
    instructions.pack(pady=10)
    
    # Image selection button
    def select_image():
        filetypes = [("Images", "*.jpg *.jpeg *.png")]
        path = filedialog.askopenfilename(title="Choose an image", filetypes=filetypes)
        
        if path:
            try:
                # Load and display image
                img = cv2.imread(path)
                if img is not None:
                    # Convert to RGB for display
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = detect_faces(gray)
                    
                    # Draw rectangles around detected faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    messagebox.showinfo("Success", f"Found {len(faces)} face(s) in the image")
                else:
                    messagebox.showerror("Error", "Could not load the image")
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
    
    # Buttons
    select_btn = tk.Button(root, text="Select Image", command=select_image,
                          font="Verdana 14 bold", width=15, fg="white", bg="#000000",
                          pady=10, bd=0, highlightthickness=2, highlightbackground="white")
    select_btn.pack(pady=20)
    
    # Close button
    close_btn = tk.Button(root, text="Close", command=root.destroy,
                         font="Verdana 12", width=10, fg="white", bg="#666666",
                         pady=5, bd=0)
    close_btn.pack(pady=20)
    
    root.mainloop()

