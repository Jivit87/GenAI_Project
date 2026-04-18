import joblib
import os
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from utils.preprocessing import build_feature_array

_base_dir = os.path.dirname(os.path.dirname(__file__))
_model_path = os.path.join(_base_dir, "Model", "house_price_model.joblib")
_dataset_path = os.path.join(_base_dir, "Data", "houseDataset.csv")

_house_price_predictor = joblib.load(_model_path)
# Load dataset once and cache it
_df = pd.read_csv(_dataset_path)

@tool
def predict_property_price(features: dict) -> dict:
    """Predicts the market price of a property using the trained ML model."""
    mathematical_array = build_feature_array(features)
    predicted_value = float(_house_price_predictor.predict(mathematical_array)[0])

    return {
        "predicted_price": round(predicted_value, 2),
        "price_range": {
            "low":  round(predicted_value * 0.90, 2),
            "high": round(predicted_value * 1.10, 2),
        }
    }

@tool
def get_comparable_properties(latitude: float, longitude: float, price: float, bedrooms: int) -> list:
    """Finds real comparable properties from the historical dataset."""
    # 1. Filter by bedroom count first
    matches = _df[_df['No of Bedrooms'] == bedrooms].copy()
    
    # 2. Filter by price proximity (±25%)
    price_min, price_max = price * 0.75, price * 1.25
    matches = matches[(matches['Sale Price'] >= price_min) & (matches['Sale Price'] <= price_max)]
    
    if matches.empty:
        # Fallback to just price if bedrooms filter is too restrictive
        matches = _df[(_df['Sale Price'] >= price_min) & (_df['Sale Price'] <= price_max)].copy()

    # 3. Calculate distance from target location
    matches['dist'] = np.sqrt(
        (matches['Latitude'] - latitude)**2 + 
        (matches['Longitude'] - longitude)**2
    )
    
    # 4. Take the top 3 closest matches
    top_matches = matches.sort_values('dist').head(3)
    
    results = []
    for _, row in top_matches.iterrows():
        results.append({
            "id":           str(row['ID']),
            "price":        float(row['Sale Price']),
            "size_sqft":    float(row['Flat Area (in Sqft)']),
            "bedrooms":     int(row['No of Bedrooms']),
            "distance_score": round(float(row['dist']), 4)
        })
        
    return results
