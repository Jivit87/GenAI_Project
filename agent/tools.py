import joblib
import os
import random
from langchain_core.tools import tool
from utils.preprocessing import build_feature_array

# Load the ML model 
_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Model", "house_price_model.joblib")
_model = joblib.load(_model_path)


@tool
def predict_property_price(features: dict) -> dict:
    """
    Predicts the property price using the trained ML model.
    Also returns a ±10% confidence price range.
    """
    scaled = build_feature_array(features)
    price = float(_model.predict(scaled)[0])

    return {
        "predicted_price": round(price, 2),
        "price_range": {
            "low":  round(price * 0.90, 2),
            "high": round(price * 1.10, 2),
        }
    }


@tool
def get_comparable_properties(location: str, price: float, size: float) -> list:
    """
    Returns 3 synthetic comparable properties near the given price and size.
    In a real app this would call a listings API (e.g. Zillow, 99acres).
    """
    comps = []
    for i in range(3):
        comps.append({
            "id":           f"COMP-{i + 1}",
            "location":     location,
            "price":        round(price * random.uniform(0.88, 1.12), 2),
            "size_sqft":    round(size  * random.uniform(0.90, 1.10), 0),
            "sold_days_ago": random.randint(10, 90),
        })
    return comps
