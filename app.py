# app.py
from fastapi import FastAPI
from pydantic import BaseModel # used for data validation
import joblib
import numpy as np


class InputData(BaseModel):
    # can have multiple fields
    features: list[float]

class MetaData(BaseModel):
    pass

app = FastAPI()

# Load the trained model at startup
model = joblib.load("model.pkl")


@app.get("/") 
# "/" is root path.
# Register following function as a GET/ function.
def read_root():
    # simple health check
    return {"message": "ML model inference API is running"}


@app.post("/predict")
def predict(data: InputData):
    """
    FastAPI automatically transform JSON body to a InputData object
    Expects JSON like:
    {
        "features": [0.1, -0.2, 1.5, 0.3, -0.7]
    }
    """
    # Convert list to 2D array: shape (1, n_features)
    X = np.array([data.features])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()

    return {
        "input_features": data.features,
        "prediction": int(pred),
        "probabilities": proba,
    }
