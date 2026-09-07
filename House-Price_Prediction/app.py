#Using Fastapi to build a web application for house price prediction.add()
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict


app = FastAPI(title="California House Price Prediction API", version="1.0")

# Load the trained model and scaler 
with open("artifacts/best_model.pkl","rb") as f:
    model = pickle.load(f)
    
with open("artifacts/scaler.pkl","rb") as f:
    scaler = pickle.load(f)
    
# =========================
# Request schema
# =========================

    