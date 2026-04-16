import numpy as np
import joblib
import os

# Load scaler and feature column order once when this file is imported
_base = os.path.dirname(os.path.dirname(__file__))
_scaler          = joblib.load(os.path.join(_base, "Model", "scaler.joblib"))
_feature_columns = joblib.load(os.path.join(_base, "Model", "feature_columns.joblib"))


def build_feature_array(features):
    """
    Converts a flat dictionary of user inputs into a scaled numpy array
    ready for the ML model. Uses the same column order the model was trained on.

    Expected keys in `features`:
        num_bedrooms, num_bathrooms, total_flat_area, total_lot_area,
        num_floors, is_waterfront (0 or 1), overall_grade,
        area_excl_basement, basement_area, age_years, year_renovated,
        latitude, longitude, living_area_renovated, lot_area_renovated,
        condition ('Bad'/'Fair'/'Good'/'Excellent'/'Okay'),
        times_visited ('None'/'Once'/'Twice'/'Thrice'/'Four')
    """
    condition    = features.get("condition", "Good")
    times_visited = features.get("times_visited", "None")

    # Build a dict matching each training column to its value
    row = {
        "No of Bedrooms":                           features.get("num_bedrooms", 3),
        "No of Bathrooms":                          features.get("num_bathrooms", 2),
        "Flat Area (in Sqft)":                      features.get("total_flat_area", 1500),
        "Lot Area (in Sqft)":                       features.get("total_lot_area", 5000),
        "No of Floors":                             features.get("num_floors", 1),
        "Waterfront View":                          features.get("is_waterfront", 0),
        "Overall Grade":                            features.get("overall_grade", 7),
        "Area of the House from Basement (in Sqft)": features.get("area_excl_basement", 1500),
        "Basement Area (in Sqft)":                  features.get("basement_area", 0),
        "Age of House (in Years)":                  features.get("age_years", 30),
        "Renovated Year":                           features.get("year_renovated", 0),
        "Latitude":                                 features.get("latitude", 47.5112),
        "Longitude":                                features.get("longitude", -122.257),
        "Living Area after Renovation (in Sqft)":   features.get("living_area_renovated", 1500),
        "Lot Area after Renovation (in Sqft)":      features.get("lot_area_renovated", 5000),
        # One-hot encoded condition (drop_first=True removed 'Bad')
        "Condition of the House_Excellent": 1 if condition == "Excellent" else 0,
        "Condition of the House_Fair":      1 if condition == "Fair"      else 0,
        "Condition of the House_Good":      1 if condition == "Good"      else 0,
        "Condition of the House_Okay":      1 if condition == "Okay"      else 0,
        # One-hot encoded visits (drop_first=True removed 'Four')
        "No of Times Visited_Once":   1 if times_visited == "Once"   else 0,
        "No of Times Visited_Thrice": 1 if times_visited == "Thrice" else 0,
        "No of Times Visited_Twice":  1 if times_visited == "Twice"  else 0,
    }

    # Arrange values in the exact column order the scaler/model expects
    raw = np.array([row[col] for col in _feature_columns]).reshape(1, -1)
    return _scaler.transform(raw)
