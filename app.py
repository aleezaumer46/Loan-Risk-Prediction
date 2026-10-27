import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load dataset for categorical values
df = pd.read_csv("Loan Prediction.csv", encoding="latin1")

# Get unique values for dropdowns
professions = sorted(df["Profession"].unique())
cities = sorted(df["CITY"].unique())
states = sorted(df["STATE"].unique())
house_ownerships = sorted(df["House_Ownership"].unique())
car_ownerships = sorted(df["Car_Ownership"].unique())
marital_statuses = sorted(df["Married/Single"].unique())

# Load trained artifacts
model = load_model("loan_risk_ann.keras")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


@app.route("/")
def home():
    return render_template(
        "index.html",
        professions=professions,
        cities=cities,
        states=states,
        house_ownerships=house_ownerships,
        car_ownerships=car_ownerships,
        marital_statuses=marital_statuses
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        income = int(request.form["Income"])
        age = int(request.form["Age"])
        experience = int(request.form["Experience"])
        married = request.form["Married/Single"]
        house = request.form["House_Ownership"]
        car = request.form["Car_Ownership"]
        profession = request.form["Profession"]
        city = request.form["CITY"]
        state = request.form["STATE"]
        current_job = int(request.form["CURRENT_JOB_YRS"])
        current_house = int(request.form["CURRENT_HOUSE_YRS"])

        form_data = {
            "Income": income,
            "Age": age,
            "Experience": experience,
            "Married/Single": married,
            "House_Ownership": house,
            "Car_Ownership": car,
            "Profession": profession,
            "CITY": city,
            "STATE": state,
            "CURRENT_JOB_YRS": current_job,
            "CURRENT_HOUSE_YRS": current_house
        }

        # Create DataFrame
        data = pd.DataFrame([form_data])

        # Encode categorical features
        categorical_cols = [
            "Married/Single",
            "House_Ownership",
            "Car_Ownership",
            "Profession",
            "CITY",
            "STATE"
        ]

        for col in categorical_cols:
            data[col] = label_encoders[col].transform(data[col])

        # Scale features
        data_scaled = scaler.transform(data)

        # Predict with ANN
        prob = float(model.predict(data_scaled, verbose=0)[0][0])

        is_high_risk = prob >= 0.5
        prediction = "High Risk" if is_high_risk else "Low Risk"
        risk_percentage = round(prob * 100, 2)
        confidence = round((prob if is_high_risk else (1.0 - prob)) * 100, 2)

        # Policy tiering
        if prob < 0.15:
            risk_tier = "Ultra Low Risk"
            recommendation = "Fast-track approval with prime interest rate."
        elif prob < 0.35:
            risk_tier = "Low Risk"
            recommendation = "Standard automated approval with standard terms."
        elif prob < 0.50:
            risk_tier = "Moderate Risk"
            recommendation = "Manual review recommended. Require collateral proof."
        elif prob < 0.70:
            risk_tier = "High Risk"
            recommendation = "High risk of default. Cap loan amount and verify income."
        else:
            risk_tier = "Critical Risk"
            recommendation = "Reject application or require strict cash collateral."

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            risk_percentage=risk_percentage,
            probability=round(prob, 4),
            risk_tier=risk_tier,
            recommendation=recommendation,
            professions=professions,
            cities=cities,
            states=states,
            house_ownerships=house_ownerships,
            car_ownerships=car_ownerships,
            marital_statuses=marital_statuses,
            form_values=form_data
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e),
            professions=professions,
            cities=cities,
            states=states,
            house_ownerships=house_ownerships,
            car_ownerships=car_ownerships,
            marital_statuses=marital_statuses
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
