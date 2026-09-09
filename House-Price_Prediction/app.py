#Using Fastapi to build a web application for house price prediction.add()
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import joblib
# Load the trained model and scaler 
model = joblib.load("artifacts/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
app = FastAPI(title="California House Price Prediction API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
    
# =========================
# Request schema
# =========================

class PredictionRequest(BaseModel):
    data: Dict[str, float]
    
# =========================
# Home
# =========================

@app.get("/")

def home():
    return {"message": "Welcome to the California House Price Prediction API!"}


# =========================
# Prediction API
# =========================

@app.post("/predict_api")

def predict_api(request: PredictionRequest):
    data = request.data
    print("Input:", data)
    #Convert input --> numpy array
    input_data = np.array(list(data.values())).reshape(1,-1) 
    
    #Scaling
    
    new_data = scaler.transform(input_data)
    
    #Prediction
    output = model.predict(new_data)
    prediction = float(output[0])
    print("Prediction:", prediction)
    
    return {
        "prediction": prediction
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    data = request.data
    print("Input:", data)
    #Convert input --> numpy array
    input_data = np.array(list(data.values())).reshape(1,-1) 
    
    #Scaling
    
    new_data = scaler.transform(input_data)
    
    #Prediction
    output = model.predict(new_data)
    prediction = float(output[0])
    print("Prediction:", prediction)
    
    return {
        "prediction": prediction
    }