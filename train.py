import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATASET_PATH = "CO2 Emissions_Canada.csv"
MODEL_PATH = "co2_model.pkl"
OPTIONS_PATH = "feature_options.json"

fuel_descriptions = {
    "X": "Regular Gasoline",
    "Z": "Premium Gasoline",
    "D": "Diesel",
    "E": "Ethanol (E85)",
    "N": "Natural Gas"
}

transmission_descriptions = {
    "A": "Automatic",
    "AM": "Automated Manual",
    "AS": "Automatic with Select Shift",
    "AV": "Continuously Variable (CVT)",
    "M": "Manual"
}

def describe_transmission(code):
    prefix = ""
    gears = ""
    for char in code:
        if char.isalpha():
            prefix += char
        elif char.isdigit():
            gears += char
    type_name = transmission_descriptions.get(prefix, prefix)
    if gears:
        return f"{code} - {type_name} ({gears} gears)"
    return f"{code} - {type_name}"

def train_and_save():
    print("Loading dataset from:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded. Total records: {len(df)}")

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    features = [
        'Make',
        'Model',
        'Vehicle Class',
        'Engine Size(L)',
        'Cylinders',
        'Transmission',
        'Fuel Type'
    ]
    target = 'CO2 Emissions(g/km)'

    X = df[features]
    y = df[target]

    categorical_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
    numerical_features = ['Engine Size(L)', 'Cylinders']

    # Preprocessor matches the notebook
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('numerical', 'passthrough', numerical_features)
        ]
    )

    # Best performing model from notebook: Random Forest Regressor
    model_pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Splitting dataset into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Training Random Forest Regressor Pipeline...")
    model_pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Performance on Test Set ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE:      {mae:.2f} g/km")
    print(f"RMSE:     {rmse:.2f} g/km")

    # Save trained model
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"\nModel pipeline saved to: {MODEL_PATH}")

    # Extract and save feature options for web UI
    makes = sorted(df['Make'].dropna().unique().tolist())
    vehicle_classes = sorted(df['Vehicle Class'].dropna().unique().tolist())
    transmissions = sorted(df['Transmission'].dropna().unique().tolist())
    fuel_types = sorted(df['Fuel Type'].dropna().unique().tolist())

    make_to_models = {}
    for make, group in df.groupby('Make'):
        make_to_models[make] = sorted(group['Model'].dropna().unique().tolist())

    options = {
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2)
        },
        "features": features,
        "makes": makes,
        "make_to_models": make_to_models,
        "vehicle_classes": vehicle_classes,
        "transmissions": [
            {"code": t, "label": describe_transmission(t)} for t in transmissions
        ],
        "fuel_types": [
            {"code": ft, "label": f"{ft} - {fuel_descriptions.get(ft, ft)}"} for ft in fuel_types
        ],
        "engine_size_range": {
            "min": float(df['Engine Size(L)'].min()),
            "max": float(df['Engine Size(L)'].max()),
            "default": 2.0
        },
        "cylinders_options": sorted([int(c) for c in df['Cylinders'].dropna().unique()])
    }

    with open(OPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)

    print(f"Feature options and metrics saved to: {OPTIONS_PATH}")
    return model_pipeline

if __name__ == "__main__":
    train_and_save()
