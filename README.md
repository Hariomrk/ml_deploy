# 🌿 Vehicle CO2 Emissions Predictor - ML Deployment

A complete, production-ready Machine Learning deployment project built using the Canadian Vehicle CO2 Emissions dataset. It predicts a vehicle's **CO2 emissions in grams per kilometer (g/km)** based on engine capacity, cylinders, transmission, and fuel type.

---

## 📁 Project Structure

```text
ml_deployment/
│
├── CO2 Emissions_Canada.csv    # Original training dataset (7,385 records)
├── FINAL_PROJECT (2).ipynb     # Exploratory data analysis & model experiments
│
├── train.py                    # Standalone model training & export pipeline
├── co2_model.pkl               # Serialized trained scikit-learn Pipeline
├── feature_options.json        # Extracted categorical options & metrics
│
├── app.py                      # Streamlit Interactive Web Application
├── app_api.py                  # FastAPI REST API (Swagger docs at /docs)
├── test_prediction.py          # Quick CLI prediction verification test
│
├── requirements.txt            # Python dependencies
├── Procfile                    # Cloud platform deployment configuration (Render/Heroku)
├── Dockerfile                  # Containerized deployment configuration
└── README.md                   # Complete documentation & deployment guide
```

---

## 🚀 Model Details & Performance

- **Algorithm:** Random Forest Regressor with `ColumnTransformer` (One-Hot Encoding for categories, numerical passthrough).
- **Features Used:**
  - `Make` (e.g., Toyota, Ford, BMW)
  - `Model` (e.g., Camry, F-150, 330i)
  - `Vehicle Class` (e.g., Compact, Mid-Size, SUV - Small)
  - `Engine Size(L)` (e.g., 2.0L, 3.5L)
  - `Cylinders` (e.g., 4, 6, 8)
  - `Transmission` (e.g., AS6, M6, AV)
  - `Fuel Type` (e.g., X = Regular, Z = Premium, D = Diesel, E = Ethanol)
- **Target:** `CO2 Emissions(g/km)`
- **Evaluation Metrics (on 20% unseen test set):**
  - **$R^2$ Score:** `0.9650` (96.5% variance explained)
  - **Mean Absolute Error (MAE):** `7.55 g/km`
  - **Root Mean Squared Error (RMSE):** `11.24 g/km`

---

## 💻 1. How to Run Locally

### Step 1: Install Dependencies
Open your terminal inside this folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train the Model
If you ever update the CSV dataset, retrain with:
```bash
python train.py
```

### Step 3: Launch the Streamlit Web App
```bash
streamlit run app.py
```
This opens your browser at `http://localhost:8501` featuring:
- Single vehicle prediction with dynamic Make/Model selection.
- Visual emission rating badge (Eco-friendly / Moderate / High emission).
- Batch CSV file upload & download of predicted emissions.
- Dataset insights & model performance tab.

### Step 4: (Alternative) Launch the FastAPI REST API
```bash
uvicorn app_api:app --reload --port 8000
```
- Open `http://localhost:8000/docs` to test the interactive Swagger API documentation.
- Endpoint: `POST /predict`

---

## 🌐 2. How to Deploy Online (Free Options)

### Option A: Streamlit Community Cloud (Easiest & 100% Free)
1. Initialize a git repository and push this folder to **GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Vehicle CO2 Emissions Predictor deployment"
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and main file path (`app.py`).
5. Click **"Deploy"**. Your app will be live with a public URL in 2 minutes!

---

### Option B: Hugging Face Spaces (Free Cloud Hosting)
1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and create a new Space.
2. Select **Streamlit** as the Space SDK.
3. Clone your space repository or upload `app.py`, `train.py`, `co2_model.pkl`, `feature_options.json`, `CO2 Emissions_Canada.csv`, and `requirements.txt`.
4. Commit changes. Hugging Face will automatically build and launch your application for free.

---

### Option C: Render.com (Web Service / API / Docker)
1. Push your repository to GitHub.
2. Go to **[render.com](https://render.com/)** and create a **New Web Service**.
3. Connect your GitHub repository.
4. Set:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt && python train.py`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Click **"Create Web Service"**.

---

### Option D: Docker Container
To build and run anywhere with Docker:
```bash
docker build -t co2-predictor .
docker run -p 8501:8501 co2-predictor
```
Open `http://localhost:8501` in your browser.
