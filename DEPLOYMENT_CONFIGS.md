# 🛠️ Deployment Configurations & Environment Setup Guide

This guide details all environment setups, dependency specifications, container configurations, and cloud deployment procedures for the **Vehicle CO₂ Emissions Machine Learning Project**.

---

## 💻 1. Environment Setup (Local / Server)

### Prerequisites
* **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04/22.04 LTS recommended)
* **Python Version:** Python `3.10`, `3.11`, or `3.12` (Python 3.13 supported)

### Step-by-Step Environment Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/Hariomrk/ml_deploy.git
cd ml_deploy
```

#### 2. Create an Isolated Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Production Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Model Training & Serialization
```bash
python train.py
```
*Generates `co2_model.pkl` and `feature_options.json`.*

---

## 📦 2. Dependency Specification ([requirements.txt](file:///d:/COLLEGE/ml_deployment/requirements.txt))

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.30.0
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0
```

---

## 🐳 3. Containerized Deployment (Docker)

### A. Dockerfile Setup ([Dockerfile](file:///d:/COLLEGE/ml_deployment/Dockerfile))
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

# Run training to build artifacts if not present
RUN python train.py

EXPOSE 8000 8501

HEALTHCHECK CMD curl --fail http://localhost:8000/api/status || exit 1

ENTRYPOINT ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### B. Build and Run Container
```bash
# Build the Docker image
docker build -t co2-predictor:latest .

# Run REST API on port 8000
docker run -d -p 8000:8000 --name co2_api co2-predictor:latest
```

### C. Multi-Container Orchestration ([docker-compose.yml](file:///d:/COLLEGE/ml_deployment/docker-compose.yml))
To launch both the **FastAPI REST API** and **Streamlit Dashboard** concurrently:
```bash
docker-compose up -d
```
* REST API: `http://localhost:8000`
* Streamlit UI: `http://localhost:8501`

---

## ☁️ 4. Cloud Platform Deployments

### Option A: Render.com (Free Web Service)
1. Push this repository to GitHub.
2. Sign in to [Render.com](https://render.com/) and click **New + -> Web Service**.
3. Select your repository `Hariomrk/ml_deploy`.
4. Render will auto-detect settings from [render.yaml](file:///d:/COLLEGE/ml_deployment/render.yaml):
   * **Build Command:** `pip install -r requirements.txt && python train.py`
   * **Start Command:** `uvicorn app_api:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.

### Option B: Streamlit Community Cloud (1-Click UI Hosting)
1. Visit [share.streamlit.io](https://share.streamlit.io/).
2. Select repository: `Hariomrk/ml_deploy`.
3. Set main file path: `app.py`.
4. Click **Deploy**.

### Option C: Linux VM (Ubuntu / AWS EC2 / DigitalOcean) via Systemd
Create a systemd service file `/etc/systemd/system/co2-api.service`:
```ini
[Unit]
Description=EcoTrack Vehicle CO2 REST API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ml_deploy
ExecStart=/home/ubuntu/ml_deploy/venv/bin/uvicorn app_api:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable co2-api
sudo systemctl start co2-api
```
