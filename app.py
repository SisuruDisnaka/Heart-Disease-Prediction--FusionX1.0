# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)


PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_PATH = "models/knn_tuned_model.pkl"  # or knn_heart_disease_model.pkl

# Load artifacts at startup
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    preprocessor = None
    print(f"Warning: could not load preprocessor: {e}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: could not load model: {e}")

# Expected input fields and their types (strings -> categorical / booleans; numbers -> numeric)
FIELD_ORDER = [
    "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Exercise Habits",
    "Smoking", "Family Heart Disease", "Diabetes", "BMI", "High Blood Pressure",
    "Low HDL Cholesterol", "High LDL Cholesterol", "Alcohol Consumption", "Stress Level",
    "Sleep Hours", "Sugar Consumption", "Triglyceride Level", "Fasting Blood Sugar",
    "CRP Level", "Homocysteine Level"
]

# parse booleans like Yes/No, True/False
def parse_bool(value):
    if value is None:
        return np.nan
    v = str(value).strip().lower()
    if v in ("yes", "y", "true", "1"): return "Yes"
    if v in ("no", "n", "false", "0"): return "No"
    return value

# convert numeric input safely
def parse_float(value):
    try:
        if value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if preprocessor is None or model is None:
        flash("Model or preprocessor not loaded on server. Check server logs.")
        return redirect(url_for('index'))

    # Collect form values into ordered dict
    data = {}
    for field in FIELD_ORDER:
        raw = request.form.get(field)
        data[field] = raw

    # Build DataFrame row with proper dtypes
    # Map boolean-like fields to Yes/No
    bool_fields = [
        "Smoking", "Family Heart Disease", "Diabetes", "High Blood Pressure",
        "Low HDL Cholesterol", "High LDL Cholesterol", "Alcohol Consumption"
    ]

    numeric_fields = [
        "Age", "Blood Pressure", "BMI", "Triglyceride Level", "Fasting Blood Sugar",
        "CRP Level", "Homocysteine Level", "Sleep Hours", "Cholesterol Level"
    ]

    row = {}
    for k, v in data.items():
        if k in bool_fields:
            row[k] = parse_bool(v)
        elif k in numeric_fields:
            row[k] = parse_float(v)
        else:
            # ordinal/nominal strings: keep as-is
            row[k] = v if v != "" else np.nan

    df = pd.DataFrame([row])

    # Optional: quick validation
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        flash(f"Warning: {missing_count} missing/invalid fields detected. The model will attempt to impute if preprocessor supports it.")

    # Transform using preprocessor then predict
    try:
        X_proc = preprocessor.transform(df)
    except Exception as e:
        flash(f"Preprocessor transformation failed: {e}")
        return redirect(url_for('index'))

    # If preprocessor returns sparse matrix
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    try:
        proba = model.predict_proba(X_proc)[0]
        pred = model.predict(X_proc)[0]
    except Exception:
        # Some KNN pickles may not have predict_proba; fall back to predict only
        pred = model.predict(X_proc)[0]
        proba = None

    # Prepare human readable output
    result = {
        "prediction": str(pred),
        "probability": None if proba is None else max(proba).item()
    }

    return render_template("result.html", result=result, input=row)

if __name__ == '__main__':
    app.run(debug=True)
