import joblib
import os
import random
from langchain_core.tools import tool
from utils.preprocessing import build_feature_array

_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Model", "house_price_model.joblib")
_house_price_predictor = joblib.load(_model_path)


@tool
def predict_property_price(features: dict) -> dict:
    # 1. Convert human inputs into a strict array of numbers
    mathematical_array = build_feature_array(features)
    
    # 2. Get the prediction
    predicted_value = float(_house_price_predictor.predict(mathematical_array)[0])

    # 3. Create a safety buffer (±10%) so we don't claim exact perfection
    return {
        "predicted_price": round(predicted_value, 2),
        "price_range": {
            "low":  round(predicted_value * 0.90, 2),
            "high": round(predicted_value * 1.10, 2),
        }
    }


@tool
def get_comparable_properties(location: str, price: float, size: float) -> list:
    similar_houses = []
    
    # Generate 3 fake houses
    for i in range(3):
        similar_houses.append({
            "id":           f"COMP-{i + 1}",
            "location":     location,
            # Randomize the price by ±12%
            "price":        round(price * random.uniform(0.88, 1.12), 2),
            # Randomize the square footage by ±10%
            "size_sqft":    round(size  * random.uniform(0.90, 1.10), 0),
            "sold_days_ago": random.randint(10, 90),
        })
        
    return similar_houses
