# main_login.py
import tkinter as tk
from tkinter import messagebox
import pymysql
from login import login_function
from register import register_function

def mainfunction():
    """Main function to handle login/signup window"""
    login_window = tk.Toplevel()
    login_window.title("Login/Sign-Up")
    login_window.geometry("400x300")
    login_window.configure(bg="#051729")
    
    # Title
    title_label = tk.Label(login_window, text="Welcome to Face Recognition System", 
                          font="Helvetica 16 bold", fg="white", bg="#051729")
    title_label.pack(pady=20)
    
    # Login Button
    login_btn = tk.Button(login_window, text="Login", command=login_function,
                         font="Verdana 14 bold", width=15, fg="white", bg="#000000",
                         pady=10, bd=0, highlightthickness=2, highlightbackground="white")
    login_btn.pack(pady=20)
    
    # Register Button
    register_btn = tk.Button(login_window, text="Sign Up", command=register_function,
                           font="Verdana 14 bold", width=15, fg="white", bg="#000000",
                           pady=10, bd=0, highlightthickness=2, highlightbackground="white")
    register_btn.pack(pady=20)
    
    # Close Button
    close_btn = tk.Button(login_window, text="Close", command=login_window.destroy,
                         font="Verdana 12", width=10, fg="white", bg="#666666",
                         pady=5, bd=0)
    close_btn.pack(pady=20)

