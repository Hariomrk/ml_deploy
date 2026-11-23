# 📡 Vehicle CO₂ Emissions Prediction REST API Documentation

A production-ready REST API built with **FastAPI** that serves inference from a scikit-learn **Random Forest Regressor Pipeline** ($R^2 = 0.9650$, $\text{MAE} = 7.55\text{ g/km}$).

---

## 🚀 Quick Start

### 1. Launch the API Server
```bash
uvicorn app_api:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Interactive ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Web UI with Tailwind CSS:** [http://localhost:8000/](http://localhost:8000/)

---

## 📋 Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web UI frontend (HTML, Tailwind CSS, JS with Yellow/Green gradient) |
| `GET` | `/api/status` | Healthcheck and API operational status |
| `GET` | `/api/features` | Available manufacturers, classes, transmissions, and fuels |
| `POST` | `/api/predict` | Single vehicle prediction + environmental metrics |
| `POST` | `/api/predict/batch` | Batch fleet processing with summary statistics |

---

## 🔍 Detailed Endpoint Reference

### 1. `POST /api/predict`
Predicts tailpipe CO₂ emissions for a single vehicle.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Body Schema
```json
{
  "make": "TOYOTA",
  "model": "CAMRY",
  "vehicle_class": "MID-SIZE",
  "engine_size": 2.5,
  "cylinders": 4,
  "transmission": "AS8",
  "fuel_type": "X"
}
```

#### Field Specifications
| Field | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `make` | string | Yes | Vehicle manufacturer | `"TOYOTA"` |
| `model` | string | Yes | Vehicle model name | `"CAMRY"` |
| `vehicle_class` | string | Yes | Body class | `"MID-SIZE"` |
| `engine_size` | float | Yes | Displacement in Liters (0.5 to 10.0) | `2.5` |
| `cylinders` | integer | Yes | Number of cylinders (2 to 16) | `4` |
| `transmission` | string | Yes | Transmission code (`AS6`, `AS8`, `AV`, `M6`) | `"AS8"` |
| `fuel_type` | string | Yes | Fuel code (`X`: Regular, `Z`: Premium, `D`: Diesel, `E`: E85) | `"X"` |

#### Successful Response (`200 OK`)
```json
{
  "status": "success",
  "predicted_co2_emission": 173.8,
  "unit": "g/km",
  "eco_grade": "A",
  "impact_category": "Eco-Friendly (Low Emission)",
  "annual_co2_tonnes": 2.61,
  "trees_needed_to_offset_per_year": 119,
  "variance_vs_national_fleet_avg_percent": -30.5,
  "vehicle_specifications": {
    "make": "TOYOTA",
    "model": "CAMRY",
    "vehicle_class": "MID-SIZE",
    "engine_size": 2.5,
    "cylinders": 4,
    "transmission": "AS8",
    "fuel_type": "X"
  }
}
```

---

### 2. `POST /api/predict/batch`
Predicts emissions across a fleet of vehicles simultaneously.

#### Request Body Schema
```json
{
  "vehicles": [
    {
      "make": "TOYOTA",
      "model": "CAMRY",
      "vehicle_class": "MID-SIZE",
      "engine_size": 2.5,
      "cylinders": 4,
      "transmission": "AS8",
      "fuel_type": "X"
    },
    {
      "make": "FORD",
      "model": "F-150",
      "vehicle_class": "PICKUP TRUCK - STANDARD",
      "engine_size": 3.3,
      "cylinders": 6,
      "transmission": "AS6",
      "fuel_type": "X"
    }
  ]
}
```

#### Successful Response (`200 OK`)
```json
{
  "status": "success",
  "total_vehicles_processed": 2,
  "fleet_average_co2_g_km": 216.3,
  "cleanest_vehicle_co2_g_km": 173.8,
  "highest_emitting_vehicle_co2_g_km": 258.8,
  "predictions": [
    {
      "vehicle": { ... },
      "predicted_co2_emission": 173.8,
      "annual_co2_tonnes": 2.61
    },
    {
      "vehicle": { ... },
      "predicted_co2_emission": 258.8,
      "annual_co2_tonnes": 3.88
    }
  ]
}
```

---

## 💻 Developer Code Snippets

### Python (`requests`)
```python
import requests

url = "http://localhost:8000/api/predict"
payload = {
    "make": "TOYOTA",
    "model": "CAMRY",
    "vehicle_class": "MID-SIZE",
    "engine_size": 2.5,
    "cylinders": 4,
    "transmission": "AS8",
    "fuel_type": "X"
}

response = requests.post(url, json=payload)
data = response.json()
print(f"Predicted CO2: {data['predicted_co2_emission']} {data['unit']}")
print(f"Eco Grade: {data['eco_grade']} ({data['impact_category']})")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "HONDA",
    "model": "CR-V AWD",
    "vehicle_class": "SUV - SMALL",
    "engine_size": 1.5,
    "cylinders": 4,
    "transmission": "AV",
    "fuel_type": "X"
  }'
```

### JavaScript / Browser (`fetch`)
```javascript
const vehicleData = {
  make: "BMW",
  model: "330i xDRIVE SEDAN",
  vehicle_class: "COMPACT",
  engine_size: 2.0,
  cylinders: 4,
  transmission: "AS8",
  fuel_type: "Z"
};

fetch("http://localhost:8000/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(vehicleData)
})
  .then(res => res.json())
  .then(result => {
    console.log(`Prediction: ${result.predicted_co2_emission} g/km`);
  });
```

---

## ⚠️ Error Responses

| HTTP Code | Description | Example Cause |
| :--- | :--- | :--- |
| `422 Unprocessable Entity` | Schema validation error | Passing string for `engine_size` instead of float |
| `503 Service Unavailable` | Model file missing | `co2_model.pkl` not found (run `python train.py`) |
| `500 Internal Server Error` | Unexpected processing failure | Pipeline exception |
