from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Iris Classification API")

MODEL_PATH = "model/iris_model.pkl"
model = joblib.load(MODEL_PATH)

columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

class IrisFeatures(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Iris ML API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(iris: IrisFeatures):
    data = pd.DataFrame([iris.features], columns=columns)
    prediction = model.predict(data)
    return {"predicted_species": str(prediction[0])}
