import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv("Data/houseDataset.csv")
df = df.drop(columns=["ID", "Date House was Sold", "Zipcode"])

num_cols = ["No of Bathrooms", "Flat Area (in Sqft)", "Lot Area (in Sqft)", "Area of the House from Basement (in Sqft)", "Latitude", "Longitude", "Living Area after Renovation (in Sqft)"]
cat_cols = ["Waterfront View", "Condition of the House", "No of Times Visited"]

df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

df["Waterfront View"] = LabelEncoder().fit_transform(df["Waterfront View"])
df = pd.get_dummies(df, columns=["Condition of the House", "No of Times Visited"], drop_first=True)

X = df.drop(columns=["Sale Price"])
y = df["Sale Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Training small Random Forest model (max_depth=12, n_estimators=60)...")
model = RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=2, random_state=42)
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)
print(f"Model R² score: {score:.4f}")

os.makedirs("Model", exist_ok=True)
joblib.dump(model,  "Model/house_price_model.joblib")
joblib.dump(scaler, "Model/scaler.joblib")
joblib.dump(X.columns.tolist(), "Model/feature_columns.joblib")

print("Saved smaller model successfully.")
