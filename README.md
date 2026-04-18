---
title: Real Estate Prediction
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

---
title: GenAI Real Estate Advisor
emoji: 🏡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🏠 House Sale Price Prediction

A machine learning project that predicts house sale prices using the **King County House Sales Dataset**. The project compares **Linear Regression** and **Random Forest Regression** models, with Random Forest achieving the best performance (**R² = 0.89**).

---

## 🚀 Live Demo
**Hosted on Hugging Face Spaces:** [Link to your Space](https://huggingface.co/spaces/parthrajsingh/real-estate-prediction)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## 🔍 Overview

This Jupyter Notebook (`GenAI_MidSem.ipynb`) performs an end-to-end machine learning pipeline for predicting house sale prices. It covers:

1. **Data Loading & Exploration** — Understanding the dataset structure and statistics
2. **Data Preprocessing** — Handling missing values, encoding categorical features, and feature selection
3. **Model Training & Evaluation** — Comparing Linear Regression vs. Random Forest Regression
4. **Model Selection** — Automatically selecting the best-performing model based on R² score

---

## 📊 Dataset

| Property | Details |
|---|---|
| **File** | `Data/houseDataset.csv` |
| **Rows** | 21,609 |
| **Columns** | 21 |
| **Target Variable** | `Sale Price` |

### Features

| Feature | Type | Description |
|---|---|---|
| ID | int64 | Unique identifier for each house |
| Date House was Sold | object | Date of sale |
| Sale Price | int64 | **Target** — Price the house was sold for |
| No of Bedrooms | int64 | Number of bedrooms |
| No of Bathrooms | float64 | Number of bathrooms |
| Flat Area (in Sqft) | float64 | Living area in square feet |
| Lot Area (in Sqft) | float64 | Lot size in square feet |
| No of Floors | float64 | Number of floors |
| Waterfront View | object | Whether the property has a waterfront view (Yes/No) |
| No of Times Visited | object | Number of times the property was visited |
| Condition of the House | object | Overall condition rating |
| Overall Grade | int64 | Overall grade given to the house |
| Area of the House from Basement (in Sqft) | float64 | Total area above basement |
| Basement Area (in Sqft) | int64 | Basement area in square feet |
| Age of House (in Years) | int64 | Age of the house |
| Renovated Year | int64 | Year of renovation (0 if never renovated) |
| Zipcode | float64 | ZIP code location |
| Latitude | float64 | Latitude coordinate |
| Longitude | float64 | Longitude coordinate |
| Living Area after Renovation (in Sqft) | float64 | Living area post-renovation |
| Lot Area after Renovation (in Sqft) | int64 | Lot area post-renovation |

---

## ⚙️ Project Workflow

### 1. Data Loading & Exploration
- Load dataset from CSV using Pandas
- Display dataset shape, first/last rows, summary statistics (`describe()`), data types (`info()`), and missing value counts (`isnull().sum()`)

### 2. Data Preprocessing

#### Missing Value Handling
- **Numerical columns** (Bathrooms, Flat Area, Lot Area, etc.) → Imputed with **median** using `SimpleImputer`
- **Categorical columns** (Zipcode, Waterfront View, Condition, Times Visited) → Imputed with **most frequent** value using `SimpleImputer`

#### Feature Engineering
- **Label Encoding** — `Waterfront View` (Yes/No → 1/0) using `LabelEncoder`
- **One-Hot Encoding** — `Condition of the House` and `No of Times Visited` using `pd.get_dummies(drop_first=True)`
- **Dropped Columns** — `ID`, `Date House was Sold`, `Zipcode` (non-predictive features)

### 3. Train-Test Split
- **80% Training** / **20% Testing** with `random_state=42` for reproducibility

### 4. Model Training & Evaluation
- **Linear Regression** — Baseline model
- **Random Forest Regressor** — Ensemble model with `n_estimators=200`

### 5. Model Selection
- Automatically selects the model with the higher R² score

---

## 📈 Results

| Metric | Linear Regression | Random Forest |
|---|---|---|
| **MAE** | 93,815.26 | 56,430.32 |
| **RMSE** | 125,725.51 | 83,673.56 |
| **R² Score** | 0.7523 | **0.8903** ✅ |

> **Best Model: Random Forest Regressor** — Selected automatically based on the highest R² score.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **Python 3.11** | Programming language |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation and analysis |
| **scikit-learn** | Machine learning models, preprocessing, and evaluation |

### Key scikit-learn Modules Used
- `sklearn.impute.SimpleImputer` — Missing value imputation
- `sklearn.preprocessing.LabelEncoder` — Label encoding for categorical features
- `sklearn.model_selection.train_test_split` — Splitting data into train/test sets
- `sklearn.linear_model.LinearRegression` — Linear Regression model
- `sklearn.ensemble.RandomForestRegressor` — Random Forest model
- `sklearn.metrics` — MAE, MSE, R² score evaluation

---

## 📁 Project Structure

```
GenAI_Project/
├── Data/
│   └── houseDataset.csv       # House sales dataset (21,609 records)
├── GenAI_MidSem.ipynb         # Main Jupyter Notebook
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GenAI_Project

# Install dependencies
pip install numpy pandas scikit-learn jupyter

# Launch the notebook
jupyter notebook GenAI_MidSem.ipynb
```

### Running on Google Colab
1. Upload `GenAI_MidSem.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `Data/houseDataset.csv` to the Colab runtime
3. Run all cells sequentially

---

## 📝 License

This project is for educational purposes (Mid-Semester Examination).
