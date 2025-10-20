# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** (if you haven't already)
2. **Upload all files** to your GitHub repository
3. **Ensure these files are in your repo:**
   - `streamlit_app.py` (main entry point)
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (configuration)

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with your GitHub account**
3. **Click "New app"**
4. **Fill in the details:**
   - **Repository:** Select your GitHub repository
   - **Branch:** `main` or `master`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** Choose a custom URL (optional)

5. **Click "Deploy!"**

### Step 3: Wait for Deployment

- Streamlit will automatically install dependencies
- Build and deploy your app
- You'll get a live URL like: `https://your-app-name.streamlit.app`

## 📁 Required Files Structure

```
your-repo/
├── streamlit_app.py          # Main entry point
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── face_samples/            # Sample images (optional)
├── face_samples2/           # Sample images (optional)
└── Images/                  # App images (optional)
```

## 🔧 Configuration Details

### streamlit_app.py
```python
#!/usr/bin/env python3
from app import main

if __name__ == "__main__":
    main()
```

### requirements.txt
```
streamlit>=1.28.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
Pillow>=9.0.0
pandas>=1.5.0
```

### .streamlit/config.toml
```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#f0f2f6"
secondaryBackgroundColor = "#ffffff"
textColor = "#262730"
font = "sans serif"
```

## 🚨 Troubleshooting

### Common Issues:

1. **Import Error:**
   - Make sure `streamlit_app.py` imports from `app.py` correctly
   - Check that all dependencies are in `requirements.txt`

2. **File Not Found:**
   - Ensure all required files are in the repository
   - Check file paths are correct

3. **Database Issues:**
   - SQLite database will be created automatically
   - No external database setup required

4. **Deployment Fails:**
   - Check the logs in Streamlit Cloud dashboard
   - Verify all dependencies are compatible

## 🌐 Access Your App

Once deployed, your app will be available at:
- **URL:** `https://your-app-name.streamlit.app`
- **Default Login:** `mansi` / `mansi0904`

## 📊 Features Available

- ✅ **Criminal Detection** - Upload images to detect criminals
- ✅ **Missing Person Search** - Search for missing persons
- ✅ **Database Management** - View and manage records
- ✅ **User Authentication** - Secure login system
- ✅ **Modern UI** - Beautiful, responsive interface

## 🔄 Updates

To update your app:
1. Push changes to your GitHub repository
2. Streamlit Cloud will automatically redeploy
3. Your app will be updated in a few minutes

## 💡 Tips

- **Free Tier:** Streamlit Cloud offers free hosting
- **Custom Domain:** Available in paid plans
- **Private Repos:** Supported in paid plans
- **Environment Variables:** Can be set in Streamlit Cloud dashboard

---

**Ready to deploy? Follow the steps above and your face recognition app will be live on Streamlit Cloud! 🚀**
