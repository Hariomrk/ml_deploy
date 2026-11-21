import joblib
import pandas as pd

model = joblib.load("co2_model.pkl")
test_car = pd.DataFrame([{
    'Make': 'TOYOTA',
    'Model': 'CAMRY',
    'Vehicle Class': 'MID-SIZE',
    'Engine Size(L)': 2.5,
    'Cylinders': 4,
    'Transmission': 'AS8',
    'Fuel Type': 'X'
}])

prediction = model.predict(test_car)
print("Prediction test successful!")
print(f"Sample Input: Toyota Camry 2.5L 4 Cyl")
print(f"Predicted CO2 Emission: {prediction[0]:.2f} g/km")
