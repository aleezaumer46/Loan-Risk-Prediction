"""
Phase 13: Customer Application Testing Suite
Evaluates at least 10 distinct customer profiles against the trained ANN Model.
Records Customer Input, Prediction, Probability, and Financial Explanation.
"""

import json
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# 10 Diverse Customer Test Cases (including both Low Risk and High Risk profiles)
TEST_CUSTOMERS = [
    {
        "customer_id": "CUST-001",
        "description": "Young entry-level mechanical engineer, single, rented home, low tenure",
        "data": {
            "Income": 1303834,
            "Age": 23,
            "Experience": 3,
            "Married/Single": "single",
            "House_Ownership": "rented",
            "Car_Ownership": "no",
            "Profession": "Mechanical_engineer",
            "CITY": "Rewa",
            "STATE": "Madhya_Pradesh",
            "CURRENT_JOB_YRS": 3,
            "CURRENT_HOUSE_YRS": 13
        }
    },
    {
        "customer_id": "CUST-002",
        "description": "Established corporate physician, single, rented home, high income, long tenure",
        "data": {
            "Income": 7574516,
            "Age": 40,
            "Experience": 10,
            "Married/Single": "single",
            "House_Ownership": "rented",
            "Car_Ownership": "no",
            "Profession": "Physician",
            "CITY": "Bhopal",
            "STATE": "Madhya_Pradesh",
            "CURRENT_JOB_YRS": 9,
            "CURRENT_HOUSE_YRS": 13
        }
    },
    {
        "customer_id": "CUST-003",
        "description": "Senior Software Developer, married, owned house, car owner, high stability",
        "data": {
            "Income": 8945000,
            "Age": 45,
            "Experience": 18,
            "Married/Single": "married",
            "House_Ownership": "owned",
            "Car_Ownership": "yes",
            "Profession": "Software_Developer",
            "CITY": "Bangalore",
            "STATE": "Karnataka",
            "CURRENT_JOB_YRS": 12,
            "CURRENT_HOUSE_YRS": 14
        }
    },
    {
        "customer_id": "CUST-004",
        "description": "High-Risk Customer: Young Secretary, single, rented house, no car, elevated credit risk",
        "data": {
            "Income": 6868118,
            "Age": 25,
            "Experience": 16,
            "Married/Single": "single",
            "House_Ownership": "rented",
            "Car_Ownership": "no",
            "Profession": "Secretary",
            "CITY": "Danapur",
            "STATE": "Bihar",
            "CURRENT_JOB_YRS": 13,
            "CURRENT_HOUSE_YRS": 13
        }
    },
    {
        "customer_id": "CUST-005",
        "description": "High-Risk Customer: Single Librarian, rented apartment, no vehicle, elevated risk score",
        "data": {
            "Income": 6325118,
            "Age": 21,
            "Experience": 13,
            "Married/Single": "single",
            "House_Ownership": "rented",
            "Car_Ownership": "no",
            "Profession": "Librarian",
            "CITY": "Burhanpur",
            "STATE": "Madhya_Pradesh",
            "CURRENT_JOB_YRS": 11,
            "CURRENT_HOUSE_YRS": 12
        }
    },
    {
        "customer_id": "CUST-006",
        "description": "High-Risk Customer: Technical Writer, single, rented housing, high regional default correlation",
        "data": {
            "Income": 9861298,
            "Age": 59,
            "Experience": 16,
            "Married/Single": "single",
            "House_Ownership": "rented",
            "Car_Ownership": "no",
            "Profession": "Technical_writer",
            "CITY": "North_Dumdum",
            "STATE": "West_Bengal",
            "CURRENT_JOB_YRS": 9,
            "CURRENT_HOUSE_YRS": 11
        }
    },
    {
        "customer_id": "CUST-007",
        "description": "Senior Civil Servant, married, owned property, long residential stability",
        "data": {
            "Income": 6200000,
            "Age": 54,
            "Experience": 20,
            "Married/Single": "married",
            "House_Ownership": "owned",
            "Car_Ownership": "yes",
            "Profession": "Civil_servant",
            "CITY": "New_Delhi",
            "STATE": "Delhi",
            "CURRENT_JOB_YRS": 14,
            "CURRENT_HOUSE_YRS": 14
        }
    },
    {
        "customer_id": "CUST-008",
        "description": "Chartered Accountant, married, owned home, high income, established client base",
        "data": {
            "Income": 9200000,
            "Age": 48,
            "Experience": 16,
            "Married/Single": "married",
            "House_Ownership": "owned",
            "Car_Ownership": "yes",
            "Profession": "Chartered_Accountant",
            "CITY": "Ahmedabad",
            "STATE": "Gujarat",
            "CURRENT_JOB_YRS": 11,
            "CURRENT_HOUSE_YRS": 14
        }
    },
    {
        "customer_id": "CUST-009",
        "description": "Entry-level Hotel Manager, single, rented, low tenure, high mobility",
        "data": {
            "Income": 1850000,
            "Age": 25,
            "Experience": 3,
            "Married/Single": "single",
            "House_Ownership": "rented",
            "Car_Ownership": "no",
            "Profession": "Hotel_Manager",
            "CITY": "Jaipur",
            "STATE": "Rajasthan",
            "CURRENT_JOB_YRS": 2,
            "CURRENT_HOUSE_YRS": 10
        }
    },
    {
        "customer_id": "CUST-010",
        "description": "Experienced Police Officer, married, owned home, steady municipal career",
        "data": {
            "Income": 4800000,
            "Age": 43,
            "Experience": 15,
            "Married/Single": "married",
            "House_Ownership": "owned",
            "Car_Ownership": "yes",
            "Profession": "Police_officer",
            "CITY": "Lucknow",
            "STATE": "Uttar_Pradesh",
            "CURRENT_JOB_YRS": 10,
            "CURRENT_HOUSE_YRS": 12
        }
    }
]


def run_tests():
    print("=" * 80)
    print("PHASE 13: LOAN RISK PREDICTION SYSTEM - 10 CUSTOMER EMPIRICAL TEST SUITE")
    print("=" * 80)

    model = load_model("loan_risk_ann.keras")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")

    results = []

    for idx, item in enumerate(TEST_CUSTOMERS, 1):
        cid = item["customer_id"]
        desc = item["description"]
        raw_data = item["data"].copy()

        df_row = pd.DataFrame([raw_data])

        # Categorical encoding
        cat_cols = ["Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]
        for col in cat_cols:
            df_row[col] = label_encoders[col].transform(df_row[col])

        # Scaling
        scaled_row = scaler.transform(df_row)

        # Inference
        prob = float(model.predict(scaled_row, verbose=0)[0][0])
        prediction = "High Risk" if prob >= 0.5 else "Low Risk"
        risk_pct = round(prob * 100, 2)
        confidence = round((prob if prob >= 0.5 else (1.0 - prob)) * 100, 2)

        # Generate financial explanation
        if prob < 0.15:
            explanation = f"Substantial repayment probability ({100-risk_pct:.1f}%). Low debt-to-income risk; recommended for instant credit approval."
        elif prob < 0.35:
            explanation = f"Adequate repayment capacity ({100-risk_pct:.1f}%). Stable income stream with manageable default risk."
        elif prob < 0.50:
            explanation = f"Borderline credit profile with {risk_pct}% default likelihood. Recommend collateral or co-borrower requirement."
        else:
            explanation = f"Elevated default risk ({risk_pct}%). High statistical default probability; recommend strict manual underwriting, guarantor, or rejection."

        results.append({
            "test_number": idx,
            "customer_id": cid,
            "description": desc,
            "income": raw_data["Income"],
            "age": raw_data["Age"],
            "experience": raw_data["Experience"],
            "profession": raw_data["Profession"],
            "city": raw_data["CITY"],
            "prediction": prediction,
            "probability": round(prob, 4),
            "risk_percentage": risk_pct,
            "confidence": confidence,
            "explanation": explanation
        })

        print(f"\n[Test Case #{idx}] Customer ID: {cid}")
        print(f"Profile: {desc}")
        print(f"Key Features: Age={raw_data['Age']}, Income=Rs.{raw_data['Income']:,}, Exp={raw_data['Experience']} yrs, Job_Tenure={raw_data['CURRENT_JOB_YRS']} yrs, City={raw_data['CITY']}")
        print(f"Prediction: {prediction} | Default Probability: {risk_pct}% | Confidence: {confidence}%")
        print(f"Financial Reasoning: {explanation}")

    # Save to JSON
    with open("documentation/test_results_10_customers.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("All 10 customer test cases processed successfully. Results saved to documentation/test_results_10_customers.json")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
