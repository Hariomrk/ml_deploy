from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Vehicle CO2 Emissions Prediction API",
    description="REST API for predicting vehicle CO2 emissions based on vehicle specifications.",
    version="1.0.0"
)

MODEL_PATH = "co2_model.pkl"

class VehicleSpec(BaseModel):
    make: str = Field(..., example="TOYOTA", description="Manufacturer name")
    model: str = Field(..., example="CAMRY", description="Vehicle model name")
    vehicle_class: str = Field(..., example="MID-SIZE", description="Vehicle classification")
    engine_size: float = Field(..., example=2.5, description="Engine size in Liters")
    cylinders: int = Field(..., example=4, description="Number of engine cylinders")
    transmission: str = Field(..., example="AS8", description="Transmission code (e.g. AS6, M6, AV)")
    fuel_type: str = Field(..., example="X", description="Fuel type code (X, Z, D, E, N)")

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Vehicle CO2 Emissions Predictor",
        "docs_url": "/docs",
        "endpoint": "/predict"
    }

@app.post("/predict")
def predict_emission(vehicle: VehicleSpec):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model is not trained yet. Run 'python train.py' first."
        )

    try:
        pipeline = joblib.load(MODEL_PATH)
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

        if pred < 170:
            category = "Eco-Friendly (Low Emission)"
        elif pred <= 250:
            category = "Moderate Emission"
        else:
            category = "High Emission"

        return {
            "predicted_co2_emission": round(pred, 2),
            "unit": "g/km",
            "impact_category": category,
            "vehicle_details": vehicle.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
