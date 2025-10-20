# 🚀 Face Recognition App - Hosting Guide

This guide provides multiple options to host your Face Recognition Challenge application.

## 📋 Prerequisites

- Python 3.8+ (for local development)
- Docker & Docker Compose (for containerized deployment)
- Git (for version control)

## 🌐 Hosting Options

### Option 1: Local Development (Easiest)

**Perfect for testing and development**

1. **Clone and setup:**
   ```bash
   git clone <your-repo-url>
   cd Face-recognition-Challenge-Engage-22-main
   chmod +x run_local.sh
   ./run_local.sh
   ```

2. **Access the app:**
   - Open your browser and go to `http://localhost:8501`
   - Default login: `mansi` / `mansi0904`

### Option 2: Docker Deployment (Recommended)

**Best for production and easy deployment**

1. **Quick start:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

2. **Manual Docker setup:**
   ```bash
   # Build the image
   docker build -t face-recognition-app .
   
   # Run the container
   docker run -p 8501:8501 -v $(pwd)/face_recognition.db:/app/face_recognition.db face-recognition-app
   ```

3. **Using Docker Compose:**
   ```bash
   docker-compose up --build -d
   ```

### Option 3: Cloud Deployment

#### A. Heroku (Free Tier Available)

1. **Create Heroku app:**
   ```bash
   # Install Heroku CLI first
   heroku create your-face-recognition-app
   ```

2. **Create Procfile:**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy face recognition app"
   git push heroku main
   ```

#### B. Railway

1. **Connect your GitHub repo to Railway**
2. **Railway will automatically detect the Streamlit app**
3. **Deploy with one click**

#### C. Streamlit Cloud (Free)

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub repo**
4. **Deploy automatically**

#### D. Google Cloud Run

1. **Build and push to Google Container Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/face-recognition-app
   gcloud run deploy --image gcr.io/PROJECT-ID/face-recognition-app --platform managed --region us-central1 --allow-unauthenticated
   ```

#### E. AWS Elastic Beanstalk

1. **Create `Dockerrun.aws.json`:**
   ```json
   {
     "AWSEBDockerrunVersion": "1",
     "Image": {
       "Name": "your-account.dkr.ecr.region.amazonaws.com/face-recognition-app",
       "Update": "true"
     },
     "Ports": [
       {
         "ContainerPort": "8501"
       }
     ]
   }
   ```

2. **Deploy using EB CLI:**
   ```bash
   eb init
   eb create production
   eb deploy
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file for configuration:

```env
# Database
DATABASE_URL=sqlite:///face_recognition.db

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Security (for production)
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Database Setup

The app uses SQLite by default (no external database required). For production with high traffic, consider:

- **PostgreSQL** (recommended for production)
- **MySQL** (original requirement)
- **SQLite** (default, good for small to medium apps)

## 📊 Monitoring & Maintenance

### Health Checks

- **Local:** `http://localhost:8501/_stcore/health`
- **Docker:** `curl http://localhost:8501/_stcore/health`

### Logs

```bash
# Docker logs
docker-compose logs -f

# Local logs
# Check terminal output when running locally
```

### Backup

```bash
# Backup database
cp face_recognition.db face_recognition_backup_$(date +%Y%m%d).db

# Backup with Docker
docker cp container_name:/app/face_recognition.db ./backup/
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Kill process using port 8501
   lsof -ti:8501 | xargs kill -9
   ```

2. **Docker build fails:**
   ```bash
   # Clean Docker cache
   docker system prune -a
   docker-compose build --no-cache
   ```

3. **Permission denied:**
   ```bash
   chmod +x *.sh
   ```

4. **Database connection issues:**
   - Check if SQLite file exists
   - Verify file permissions
   - Check disk space

### Performance Optimization

1. **For production:**
   - Use a reverse proxy (nginx)
   - Enable HTTPS
   - Set up proper logging
   - Use a production database

2. **For scaling:**
   - Use Kubernetes for orchestration
   - Implement load balancing
   - Use CDN for static assets

## 📈 Scaling Options

### Horizontal Scaling

- **Kubernetes:** Deploy multiple replicas
- **Docker Swarm:** Use Docker's built-in orchestration
- **Load Balancer:** Distribute traffic across instances

### Vertical Scaling

- **Increase container resources**
- **Use more powerful cloud instances**
- **Optimize database queries**

## 🔒 Security Considerations

1. **Enable HTTPS** in production
2. **Use environment variables** for sensitive data
3. **Implement proper authentication**
4. **Regular security updates**
5. **Database encryption** for sensitive data

## 📞 Support

If you encounter issues:

1. Check the logs first
2. Verify all dependencies are installed
3. Ensure ports are not blocked
4. Check system resources (CPU, memory, disk)

## 🎯 Quick Start Commands

```bash
# Local development
./run_local.sh

# Docker deployment
./deploy.sh

# Stop everything
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart
```

---

**Happy Hosting! 🚀**
