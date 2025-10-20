#!/bin/bash

# GitHub Setup Script for Streamlit Cloud Deployment
echo "🚀 Setting up GitHub repository for Streamlit Cloud deployment"
echo "=============================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️ No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Face Recognition App - Ready for Streamlit Cloud deployment

Features:
- Modern UI/UX with gradient design
- Criminal detection system
- Missing person search
- User authentication
- Database management
- System settings
- Mobile responsive design

Ready for Streamlit Cloud deployment!"
    echo "✅ Changes committed"
fi

# Show status
echo ""
echo "📊 Git Status:"
git status

echo ""
echo "🎯 Next Steps:"
echo "1. Create a new repository on GitHub (github.com/new)"
echo "2. Copy the repository URL"
echo "3. Run these commands:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Go to https://share.streamlit.io"
echo "5. Connect your GitHub repository"
echo "6. Set main file to: streamlit_app.py"
echo "7. Deploy!"
echo ""
echo "🌐 Your app will be live at: https://your-app-name.streamlit.app"
