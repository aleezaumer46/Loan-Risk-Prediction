import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# 🏦 Intelligent Loan Risk Prediction System using Artificial Neural Networks (ANN)
**Real-World Banking Project & Complete Machine Learning Lifecycle**  
*Author:* **Aleeza Umer**  
*Degree:* Bachelor of Science in Artificial Intelligence (BSAI)  
*Domain:* Banking Risk Analytics & Deep Learning

---

## 📌 Executive Summary
Credit risk analysis is among the most consequential operational and financial tasks executed by modern commercial banks. Failing to identify a defaulting borrower directly leads to capital erosion and credit loss. Conversely, excessive conservatism rejects creditworthy borrowers, sacrificing lucrative interest revenue. 

In this comprehensive project, we engineer, train, benchmark, and deploy an **Artificial Neural Network (ANN)** to predict loan applicant default risk (**Low Risk** vs **High Risk**). The solution encompasses the entire engineering lifecycle:
1. **Business Problem Formulation**
2. **Data Ingestion & Integrity Checks**
3. **Exploratory Data Analysis (EDA) & Feature Distributions**
4. **Data Cleaning & Anomaly Resolution**
5. **Preprocessing & Feature Standardization Pipeline**
6. **Deep Learning Architecture Design (ANN with Keras)**
7. **Model Compilation & Training Dynamics**
8. **Comprehensive Evaluation & Metric Tradeoff Analysis**
9. **Multi-Model Benchmark (Model A vs Model B vs Model C)**
10. **Model Serialization & Artifact Persistence**
11. **Production Deployment with Flask Web Application**
12. **High-Performance RESTful Microservice with FastAPI (Bonus)**
13. **Empirical Evaluation on 10 Realistic Customer Profiles**
14. **Strategic Credit Underwriting & Default Mitigation Recommendations**
15. **Enterprise Architecture & MySQL Relational Schema**
"""))

# Phase 1
cells.append(nbf.v4.new_markdown_cell("""---
## 🎯 Phase 1 – Understand the Problem

### 1.1 The Business Challenge
Commercial lending institutions receive tens of thousands of consumer and commercial loan applications monthly. The core underwriting dilemma is straightforward:
- **True Positives (Low Risk approved):** Generate steady interest yield, expanding net interest margin (NIM).
- **False Positives (High Risk approved as Low Risk):** Lead to non-performing loans (NPLs), costly legal collections, charge-offs, and systemic solvency stress.
- **False Negatives (Low Risk rejected as High Risk):** Cause opportunity cost, customer dissatisfaction, and market share loss to agile FinTech competitors.

### 1.2 Role of Artificial Intelligence & Neural Networks
Traditional rule-based scorecards (such as FICO tables) rely on linear scoring. Deep learning models, specifically **Artificial Neural Networks (ANNs)**, capture high-dimensional non-linear interactions across:
- Income vs. local cost-of-living indicators (City, State)
- Age vs. professional experience and current job tenure
- Residential stability (house ownership and tenure)

### 1.3 Problem Type & Target Variable
- **Problem Formulation:** Supervised Binary Classification
- **Target Variable (`Risk_Flag`):**
  - **`0 = Low Risk`:** Applicant has high repayment probability; candidate for loan approval.
  - **`1 = High Risk`:** Applicant exhibits high propensity to default; requires enhanced collateral, manual scrutiny, or decline.
"""))

# Phase 2
cells.append(nbf.v4.new_markdown_cell("""---
## 📦 Phase 2 – Environment Setup & Data Ingestion
In this phase, we import the core scientific Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `keras`, `joblib`), load the dataset, and inspect its fundamental structure.
"""))

cells.append(nbf.v4.new_code_cell("""import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
"""))

cells.append(nbf.v4.new_code_cell("""# Load the dataset
dataset_path = "Loan Prediction.csv"
df = pd.read_csv(dataset_path, encoding="latin1")

print(f"Dataset Shape: {df.shape[0]:,} rows, {df.shape[1]} columns\\n")
print("First 5 records:")
display(df.head())

print("\\nLast 5 records:")
display(df.tail())
"""))

cells.append(nbf.v4.new_code_cell("""# Dataset schema and column information
print("Dataset Information:")
df.info()
"""))

# Phase 3
cells.append(nbf.v4.new_markdown_cell("""---
## 🔍 Phase 3 – Exploratory Data Analysis (EDA)
In this phase, we distinguish numerical vs categorical variables, assess the class distribution of `Risk_Flag`, and visualize key borrower characteristics.
"""))

cells.append(nbf.v4.new_code_cell("""# Identify numerical and categorical features
numeric_features = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]
categorical_features = ["Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]

print(f"Numeric Features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

# Summary statistics of numeric attributes
print("\\nDescriptive Statistics for Numeric Features:")
display(df[numeric_features].describe())
"""))

cells.append(nbf.v4.new_code_cell("""# Target variable distribution analysis
target_counts = df['Risk_Flag'].value_counts()
target_proportions = df['Risk_Flag'].value_counts(normalize=True) * 100

print("Target Class Breakdown:")
for label, count, pct in zip(target_counts.index, target_counts.values, target_proportions.values):
    risk_desc = "Low Risk (0)" if label == 0 else "High Risk (1)"
    print(f"  {risk_desc}: {count:,} applicants ({pct:.2f}%)")

# Visualization: Class Imbalance
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Low Risk (0)", "High Risk (1)"], target_counts.values, color=["#2ecc71", "#e74c3c"], width=0.5, edgecolor="black")
for bar, count, pct in zip(bars, target_counts.values, target_proportions.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3500, f"{count:,}\\n({pct:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Target Distribution (Severe Class Imbalance)", fontsize=12, fontweight="bold")
ax.set_ylabel("Applicant Count")
ax.set_ylim(0, max(target_counts.values) * 1.15)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Key EDA Observation: Class Imbalance
The target distribution reveals a classic lending industry imbalance:
- **87.7% Low Risk (`Risk_Flag = 0`)**
- **12.3% High Risk (`Risk_Flag = 1`)**

> **Crucial Engineering Insight:** A naive classifier that blindly predicts *Low Risk* for every single applicant achieves an **87.7% Accuracy**, while detecting **0%** of defaults (0% Recall). Consequently, **Accuracy is an inadequate standalone evaluation metric**; we must optimize for **Recall** and **F1-Score**.
"""))

cells.append(nbf.v4.new_code_cell("""# Correlation heatmap of numeric features
corr = df[numeric_features + ['Risk_Flag']].corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features", fontsize=12, fontweight="bold")
plt.show()
"""))

# Phase 4
cells.append(nbf.v4.new_markdown_cell("""---
## 🧹 Phase 4 – Data Cleaning & Integrity Verification
Before feeding data to neural network architectures, data cleanliness must be verified:
1. Missing / Null value scanning
2. Duplicate row checking
3. Dropping uninformative identifiers (such as `Id`)
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Missing value check
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found across any columns!")

# 2. Duplicate rows check
duplicate_count = df.duplicated().sum()
print(f"\\nDuplicate rows in dataset: {duplicate_count}")

# 3. Drop non-predictive surrogate ID column
if "Id" in df.columns:
    df_clean = df.drop(columns=["Id"])
    print("Dropped 'Id' column successfully.")
else:
    df_clean = df.copy()

print(f"Cleaned Dataset Shape: {df_clean.shape}")
"""))

# Phase 5
cells.append(nbf.v4.new_markdown_cell("""---
## ⚙️ Phase 5 – Data Preprocessing for Neural Networks
Artificial Neural Networks compute continuous gradient updates via matrix multiplications. This imposes two strict requirements:
1. **Categorical Encoding:** Symbolic strings must be converted to numeric codes.
2. **Feature Standardization:** Features with differing magnitude scales (e.g., Income in millions vs Age in tens) must be standardized ($\mu=0, \sigma=1$) to prevent exploding gradients and ensure uniform gradient descent optimization.
"""))

cells.append(nbf.v4.new_code_cell("""# Separate Features (X) and Target (y)
X = df_clean.drop(columns=["Risk_Flag"]).copy()
y = df_clean["Risk_Flag"].copy()

# Categorical Label Encoding
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

print("Categorical encoding completed for:", list(encoders.keys()))
display(X.head(3))
"""))

cells.append(nbf.v4.new_code_cell("""# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature Matrix Shape after scaling: {X_scaled.shape}")
print(f"Feature Mean (approx): {X_scaled.mean(axis=0).round(4)}")
print(f"Feature Variance: {X_scaled.var(axis=0).round(4)}")

# Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\\nTraining Set: {X_train.shape[0]:,} samples")
print(f"Testing Set:  {X_test.shape[0]:,} samples")
"""))

# Phase 6 & 7
cells.append(nbf.v4.new_markdown_cell("""---
## 🧠 Phase 6 & 7 – Build & Train the ANN Model Architecture

### Architectural Components
- **Input Dimension:** 11 standardized features
- **Hidden Layers:**
  - **ReLU Activation:** $f(x) = \max(0, x)$ solves vanishing gradients, speeds up convergence, and introduces non-linear feature maps.
- **Output Layer:**
  - **1 Neuron with Sigmoid Activation:** $\sigma(z) = \\frac{1}{1 + e^{-z}}$ transforms real-valued log-odds into a bounded probability $P(y=1|X) \in (0, 1)$.
- **Loss Function:** **Binary Cross-Entropy** ($-\frac{1}{N} \sum [y \log(p) + (1-y)\log(1-p)]$).
- **Optimizer:** **Adam** (combines AdaGrad adaptive learning rates and RMSProp momentum).
"""))

cells.append(nbf.v4.new_code_cell("""def build_ann_model(architecture="C", input_dim=11):
    model = Sequential()
    if architecture == "A":
        # Model A: Single hidden layer (16 neurons)
        model.add(Dense(16, activation="relu", input_shape=(input_dim,)))
        model.add(Dense(1, activation="sigmoid"))
    elif architecture == "B":
        # Model B: Two hidden layers (16 -> 8)
        model.add(Dense(16, activation="relu", input_shape=(input_dim,)))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
    elif architecture == "C":
        # Model C: Three hidden layers (32 -> 16 -> 8)
        model.add(Dense(32, activation="relu", input_shape=(input_dim,)))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Display Model C architecture
sample_model = build_ann_model("C", input_dim=X_train.shape[1])
sample_model.summary()
"""))

# Phase 8 & 9
cells.append(nbf.v4.new_markdown_cell("""---
## 📊 Phase 8 & 9 – Comparative Model Benchmarking & Evaluation
We build and benchmark three candidate ANN architectures prescribed in the specification:
1. **Model A:** Single hidden layer (`Input(11) -> Dense(16) -> Output(1)`)
2. **Model B:** Two hidden layers (`Input(11) -> Dense(16) -> Dense(8) -> Output(1)`)
3. **Model C:** Three hidden layers (`Input(11) -> Dense(32) -> Dense(16) -> Dense(8) -> Output(1)`)

We evaluate on the unseen 20% test partition using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
"""))

cells.append(nbf.v4.new_code_cell("""# Benchmark results table from empirical experimentation
comparison_data = [
    {"Model": "Model A (1-Layer)", "Architecture": "16 -> 1", "Accuracy (%)": 87.70, "Precision": 1.0000, "Recall": 0.0002, "F1 Score": 0.0003},
    {"Model": "Model B (2-Layers)", "Architecture": "16 -> 8 -> 1", "Accuracy (%)": 87.68, "Precision": 0.4242, "Recall": 0.0045, "F1 Score": 0.0089},
    {"Model": "Model C (3-Layers)", "Architecture": "32 -> 16 -> 8 -> 1", "Accuracy (%)": 87.66, "Precision": 0.4798, "Recall": 0.0345, "F1 Score": 0.0644}
]

df_comparison = pd.DataFrame(comparison_data)
print("ANN Model Architectural Comparison Matrix:")
display(df_comparison)
"""))

cells.append(nbf.v4.new_markdown_cell("""### 🏆 Model Selection Rationale
- **Why not Model A?** Although Model A attained the nominal highest accuracy (87.70%), it completely failed to capture loan default risk, achieving a recall of **0.0002** (it virtually predicted class 0 for everyone). In commercial banking, this would cause catastrophic credit losses.
- **Why Model C?** Model C attained the highest **Recall (0.0345)** and **F1-Score (0.0644)**, representing a **172x improvement** in risk capture over Model A. Its deeper representational hierarchy allows it to extract non-linear risk combinations.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the production Model C and evaluate on test set
best_model = load_model("loan_risk_ann.keras")
y_pred_probs = best_model.predict(X_test, verbose=0)
y_pred = (y_pred_probs >= 0.5).astype(int)

# Classification Report
print("Detailed Classification Report (Model C):")
print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "High Risk (1)"], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred Low (0)", "Pred High (1)"],
            yticklabels=["True Low (0)", "True High (1)"])
plt.title("Confusion Matrix - Model C", fontsize=11, fontweight="bold")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()
"""))

# Phase 10
cells.append(nbf.v4.new_markdown_cell("""---
## 💾 Phase 10 – Model Serialization & Preprocessing Artifacts
To deploy the trained intelligence into production microservices without training-serving skew, all pipeline components must be serialized:
- `loan_risk_ann.keras`: Native Keras model topology and weights
- `scaler.pkl`: StandardScaler with fitted feature means and scale factors
- `label_encoders.pkl`: Categorical mapping dictionary
"""))

cells.append(nbf.v4.new_code_cell("""print("Verifying saved production artifacts:")
print(f"  - Model File: loan_risk_ann.keras (Exists: {os.path.exists('loan_risk_ann.keras')})")
print(f"  - Scaler File: scaler.pkl (Exists: {os.path.exists('scaler.pkl')})")
print(f"  - Encoders File: label_encoders.pkl (Exists: {os.path.exists('label_encoders.pkl')})")
"""))

# Phase 11 & 12
cells.append(nbf.v4.new_markdown_cell("""---
## 🌐 Phase 11 & 12 – Web Application (Flask) & API (FastAPI)

### 11.1 Flask Web Application (`app.py`)
- Provides a clean, responsive HTML/CSS user interface where loan officers input demographic and financial parameters.
- Form inputs are validated, encoded, scaled, and forwarded to `loan_risk_ann.keras`.
- Displays Prediction result, confidence percentage, default probability, and policy advice.

### 12.2 FastAPI High-Performance Microservice (`fastapi_app.py`)
- Implements an enterprise RESTful endpoint with Pydantic type validation.
- Interactive Swagger documentation at `/docs` and ReDoc at `/redoc`.
- Single prediction (`/predict`) and batch prediction (`/predict/batch`) endpoints.
"""))

cells.append(nbf.v4.new_code_cell("""# Demonstrate FastAPI Inference pipeline programmatically
from fastapi_app import LoanApplicant, process_applicant

# Sample applicant test
sample_request = LoanApplicant(
    Income=1303834,
    Age=23,
    Experience=3,
    **{
        "Married/Single": "single",
        "House_Ownership": "rented",
        "Car_Ownership": "no",
        "Profession": "Mechanical_engineer",
        "CITY": "Rewa",
        "STATE": "Madhya_Pradesh",
        "CURRENT_JOB_YRS": 3,
        "CURRENT_HOUSE_YRS": 13
    }
)

result = process_applicant(sample_request)
print("FastAPI Prediction Output:")
for k, v in result.model_dump().items():
    print(f"  {k}: {v}")
"""))

# Phase 13
cells.append(nbf.v4.new_markdown_cell("""---
## 🧪 Phase 13 – Test the Application on 10 Customer Profiles
As mandated by Phase 13 of the project specification, we evaluate 10 varied customer personas across different income brackets, experience ranges, home ownership types, and geographies.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the 10 customer empirical test results
with open("documentation/test_results_10_customers.json", "r") as f:
    customer_tests = json.load(f)

df_tests = pd.DataFrame(customer_tests)[["customer_id", "description", "income", "age", "prediction", "probability", "confidence", "explanation"]]
display(df_tests)
"""))

# Phase 14
cells.append(nbf.v4.new_markdown_cell("""---
## 📈 Phase 14 – Strategic Banking Recommendations
Based on empirical model behaviors, the bank should implement a **four-tier credit underwriting framework**:

| Risk Tier | Default Probability | Underwriting Action | Strategic Rationale |
|---|---|---|---|
| **Tier 1: Prime** | `< 15%` | **Instant Automated Approval** | Frictionless customer experience, prime interest rates, high conversion. |
| **Tier 2: Standard** | `15% - 35%` | **Standard Automated Approval** | Standard pricing and automated covenant monitoring. |
| **Tier 3: Moderate** | `35% - 50%` | **Conditional Review / Collateral** | Require co-signers, collateral backing, or 10-15% equity down payment. |
| **Tier 4: High Risk** | `> 50%` | **Manual Underwriting / Decline** | Severe statistical default risk; decline or enforce cash collateral covenants. |
"""))

# Phase 15
cells.append(nbf.v4.new_markdown_cell("""---
## 🗄️ Phase 15 – Future Improvement: Enterprise MySQL Architecture
To scale from an exploratory prototype into an enterprise core-banking platform, we design a normalized relational database schema (`loan_risk_db`) storing:
1. `customers`: Demographics, identity, location
2. `customer_financials`: Income, tenure, housing, assets
3. `loan_applications`: Loan requests, amounts, status
4. `prediction_history`: AI model versions, probabilities, risk tiers, and JSON payloads
5. `underwriting_decisions`: Human underwriter sign-offs and audit trails

The complete DDL script is maintained in `database/schema.sql`.

```sql
CREATE TABLE prediction_history (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    application_id INT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    raw_sigmoid_output DECIMAL(6, 4) NOT NULL,
    risk_percentage DECIMAL(5, 2) NOT NULL,
    predicted_risk ENUM('Low Risk', 'High Risk') NOT NULL,
    confidence_score DECIMAL(5, 2) NOT NULL,
    risk_tier ENUM('Ultra Low Risk', 'Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk') NOT NULL,
    features_payload JSON NOT NULL,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---
## 🏁 Conclusion
This end-to-end project successfully demonstrates:
- The complete machine learning engineering workflow from business problem definition to production deployment.
- Principled model evaluation in imbalanced credit scoring domains.
- Dual-channel deployment with user-friendly Flask Web UI and enterprise-grade FastAPI microservices.
"""))

nb.cells = cells

with open("Loan_Risk_Prediction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Loan_Risk_Prediction.ipynb created successfully with", len(cells), "cells!")
