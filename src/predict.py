import joblib
import pandas as pd

MODEL_PATH = "model/iris_model.pkl"

def predict_iris(features):
    model = joblib.load(MODEL_PATH)
    columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
    data = pd.DataFrame([features], columns=columns)
    prediction = model.predict(data)
    return prediction[0]

if __name__ == "__main__":
    sample_flower = [5.1, 3.5, 1.4, 0.2]
    result = predict_iris(sample_flower)
    print("Iris Features:")
    print(sample_flower)
    print("\nPredicted Species:", result)
