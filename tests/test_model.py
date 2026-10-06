import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.predict import predict_iris

def test_prediction():
    sample_flower = [5.1, 3.5, 1.4, 0.2]
    prediction = predict_iris(sample_flower)
    assert prediction in ["setosa", "versicolor", "virginica"]
