from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import json
import os

app = FastAPI(
    title="EcoTrack Vehicle CO2 Emissions REST API",
    description="High-performance REST API for predicting vehicle tailpipe greenhouse gas emissions (g/km) using a Random Forest Scikit-Learn Pipeline.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for external web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "co2_model.pkl"
OPTIONS_PATH = "feature_options.json"
HTML_PATH = os.path.join("templates", "index.html")

class VehicleInput(BaseModel):
    make: str = Field(..., example="TOYOTA", description="Manufacturer name (e.g. TOYOTA, FORD, BMW)")
    model: str = Field(..., example="CAMRY", description="Vehicle model name")
    vehicle_class: str = Field(..., example="MID-SIZE", description="Vehicle classification (e.g. COMPACT, MID-SIZE, SUV - SMALL)")
    engine_size: float = Field(..., ge=0.5, le=10.0, example=2.5, description="Engine displacement in Liters")
    cylinders: int = Field(..., ge=2, le=16, example=4, description="Number of engine cylinders")
    transmission: str = Field(..., example="AS8", description="Transmission code (e.g. AS6, AS8, AV, M6)")
    fuel_type: str = Field(..., example="X", description="Fuel type code (X: Regular, Z: Premium, D: Diesel, E: Ethanol, N: Natural Gas)")

class BatchVehicleInput(BaseModel):
    vehicles: List[VehicleInput]

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Trained model 'co2_model.pkl' not found. Please run 'python train.py' to generate the model artifact."
        )
    return joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse, summary="Interactive Web Frontend")
def root_ui():
    """Serves the responsive HTML/CSS/Tailwind/JS frontend application with light yellow and green gradient."""
    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>EcoTrack Vehicle CO2 Emissions REST API is Running</h1><p>Visit <a href='/docs'>/docs</a> for Swagger documentation.</p>")

@app.get("/api/status", summary="Health Check and Metadata")
def health_check():
    """Returns the API health status and model availability."""
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "online",
        "service": "EcoTrack Vehicle CO2 Predictor REST API",
        "model_loaded": model_loaded,
        "endpoints": {
            "web_ui": "/",
            "swagger_docs": "/docs",
            "redoc_specs": "/redoc",
            "predict_single": "/api/predict",
            "predict_batch": "/api/predict/batch",
            "available_features": "/api/features"
        }
    }

@app.get("/api/features", summary="List Available Categorical Features")
def list_features():
    """Returns all available manufacturers, classes, transmissions, and fuel types."""
    if os.path.exists(OPTIONS_PATH):
        with open(OPTIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "feature_options.json not found"}

@app.post("/api/predict", summary="Predict Single Vehicle CO2 Emissions")
def predict_single(vehicle: VehicleInput):
    """
    Accepts vehicle specifications and returns:
    - Predicted CO2 tailpipe emissions (g/km)
    - Eco Rating Grade (A+, A, B, C, D)
    - Estimated Annual CO2 Tonnage (assuming 15,000 km/year)
    - Trees required to offset annual emissions
    - Percentage variance from Canadian national fleet average (250 g/km)
    """
    pipeline = get_model()
    try:
        input_data = pd.DataFrame([{
            'Make': vehicle.make.upper().strip(),
            'Model': vehicle.model.upper().strip(),
            'Vehicle Class': vehicle.vehicle_class.upper().strip(),
            'Engine Size(L)': float(vehicle.engine_size),
            'Cylinders': int(vehicle.cylinders),
            'Transmission': vehicle.transmission.upper().strip(),
            'Fuel Type': vehicle.fuel_type.upper().strip()
        }])

        pred = float(pipeline.predict(input_data)[0])

        # Eco Grade and category
        if pred < 140:
            grade = "A+"
            category = "Ultra Clean / Eco Champion"
        elif pred < 180:
            grade = "A"
            category = "Eco-Friendly (Low Emission)"
        elif pred <= 240:
            grade = "B"
            category = "Moderate (Average Fleet)"
        elif pred <= 300:
            grade = "C"
            category = "High Emission Vehicle"
        else:
            grade = "D"
            category = "Very High (Gas Guzzler)"

        annual_kg = (pred * 15000) / 1000
        annual_tonnes = round(annual_kg / 1000, 2)
        trees_needed = int(np.ceil(annual_kg / 22))
        canadian_avg = 250.0
        pct_diff = round(((pred - canadian_avg) / canadian_avg) * 100, 1)

        return {
            "status": "success",
            "predicted_co2_emission": round(pred, 1),
            "unit": "g/km",
            "eco_grade": grade,
            "impact_category": category,
            "annual_co2_tonnes": annual_tonnes,
            "trees_needed_to_offset_per_year": trees_needed,
            "variance_vs_national_fleet_avg_percent": pct_diff,
            "vehicle_specifications": vehicle.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/predict/batch", summary="Batch Fleet CO2 Predictions")
def predict_batch(batch: BatchVehicleInput):
    """Processes a fleet of multiple vehicles and returns individual predictions with fleet-wide averages."""
    pipeline = get_model()
    try:
        rows = []
        for v in batch.vehicles:
            rows.append({
                'Make': v.make.upper().strip(),
                'Model': v.model.upper().strip(),
                'Vehicle Class': v.vehicle_class.upper().strip(),
                'Engine Size(L)': float(v.engine_size),
                'Cylinders': int(v.cylinders),
                'Transmission': v.transmission.upper().strip(),
                'Fuel Type': v.fuel_type.upper().strip()
            })
        
        df = pd.DataFrame(rows)
        preds = pipeline.predict(df)

        results = []
        for i, pred_val in enumerate(preds):
            pred = float(pred_val)
            annual_tonnes = round(((pred * 15000) / 1000) / 1000, 2)
            results.append({
                "vehicle": rows[i],
                "predicted_co2_emission": round(pred, 1),
                "annual_co2_tonnes": annual_tonnes
            })

        return {
            "status": "success",
            "total_vehicles_processed": len(results),
            "fleet_average_co2_g_km": round(float(preds.mean()), 1),
            "cleanest_vehicle_co2_g_km": round(float(preds.min()), 1),
            "highest_emitting_vehicle_co2_g_km": round(float(preds.max()), 1),
            "predictions": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)
