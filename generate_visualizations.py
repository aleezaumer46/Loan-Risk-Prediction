import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from tensorflow.keras.models import load_model

os.makedirs("images", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

print("1. Loading dataset...")
df = pd.read_csv("Loan Prediction.csv", encoding="latin1")

# --- Chart 1: Target Distribution ---
print("Generating Target Distribution...")
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df["Risk_Flag"].value_counts()
pcts = df["Risk_Flag"].value_counts(normalize=True) * 100
bars = ax.bar(["Low Risk (0)", "High Risk (1)"], counts.values, color=["#2ecc71", "#e74c3c"], width=0.5, edgecolor="black")
for bar, count, pct in zip(bars, counts.values, pcts.values):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 3000, f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
ax.set_title("Loan Risk Flag Distribution (Severe Class Imbalance)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Number of Applicants")
ax.set_ylim(0, max(counts.values) * 1.15)
plt.savefig("images/target_distribution.png", dpi=200)
plt.close()

# --- Chart 2: Feature Distributions ---
print("Generating Numeric Features Distribution...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.histplot(data=df, x="Income", hue="Risk_Flag", kde=True, ax=axes[0], palette=["#2ecc71", "#e74c3c"], bins=30, alpha=0.5)
axes[0].set_title("Income Distribution by Risk", fontweight="bold")
axes[0].set_xlabel("Income (INR)")

sns.histplot(data=df, x="Age", hue="Risk_Flag", kde=True, ax=axes[1], palette=["#2ecc71", "#e74c3c"], bins=25, alpha=0.5)
axes[1].set_title("Age Distribution by Risk", fontweight="bold")
axes[1].set_xlabel("Age (Years)")

sns.histplot(data=df, x="Experience", hue="Risk_Flag", kde=True, ax=axes[2], palette=["#2ecc71", "#e74c3c"], bins=20, alpha=0.5)
axes[2].set_title("Work Experience Distribution by Risk", fontweight="bold")
axes[2].set_xlabel("Experience (Years)")
plt.savefig("images/numeric_distributions.png", dpi=200)
plt.close()

# --- Chart 3: Correlation Heatmap ---
print("Generating Correlation Heatmap...")
numeric_cols = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS", "Risk_Flag"]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, ax=ax, square=True, linewidths=0.5)
ax.set_title("Correlation Heatmap of Numerical Features", fontsize=13, fontweight="bold", pad=12)
plt.savefig("images/correlation_matrix.png", dpi=200)
plt.close()

# --- Preprocessing for Model Evaluation ---
print("Preparing evaluation data...")
X = df.drop(columns=["Id", "Risk_Flag"])
y = df["Risk_Flag"]

le = joblib.load("label_encoders.pkl")
for col in ["Married/Single", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]:
    X[col] = le[col].transform(X[col])

scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Evaluate Model C
print("Evaluating Model C...")
model_c = load_model("loan_risk_ann.keras")
y_pred_prob = model_c.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

# --- Chart 4: Confusion Matrix ---
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Predicted Low Risk (0)", "Predicted High Risk (1)"],
            yticklabels=["Actual Low Risk (0)", "Actual High Risk (1)"])
ax.set_title("Confusion Matrix - Selected ANN Model (Model C)", fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("True Risk Label")
ax.set_xlabel("Predicted Risk Label")
plt.savefig("images/confusion_matrix.png", dpi=200)
plt.close()

# --- Chart 5: ROC Curve ---
print("Generating ROC Curve...")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"Model C (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="lower right")
plt.savefig("images/roc_curve.png", dpi=200)
plt.close()

# --- Chart 6: Model Comparison Bar Chart ---
print("Generating Model Comparison Chart...")
models = ["Model A\n(1-Layer)", "Model B\n(2-Layers)", "Model C\n(3-Layers)"]
metrics = {
    "Accuracy (%)": [87.70, 87.68, 87.66],
    "Precision": [1.0000, 0.4242, 0.4798],
    "Recall": [0.0002, 0.0045, 0.0345],
    "F1 Score": [0.0003, 0.0089, 0.0644]
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot Accuracy
x = np.arange(len(models))
width = 0.35
axes[0].bar(x, metrics["Accuracy (%)"], width=0.45, color=["#3498db", "#2980b9", "#1f618d"], edgecolor="black")
axes[0].set_ylim(85, 90)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Model Accuracy Comparison", fontweight="bold")
for i, v in enumerate(metrics["Accuracy (%)"]):
    axes[0].text(i, v + 0.15, f"{v:.2f}%", ha="center", fontweight="bold")

# Plot F1 Score and Recall comparison
width = 0.25
axes[1].bar(x - width/2, metrics["Recall"], width, label="Recall", color="#e67e22", edgecolor="black")
axes[1].bar(x + width/2, metrics["F1 Score"], width, label="F1 Score", color="#27ae60", edgecolor="black")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_ylabel("Metric Score")
axes[1].set_title("Recall & F1 Score (Risk Detection Capability)", fontweight="bold")
axes[1].legend()
for i, (r, f) in enumerate(zip(metrics["Recall"], metrics["F1 Score"])):
    axes[1].text(i - width/2, r + 0.002, f"{r:.4f}", ha="center", fontsize=9, rotation=45)
    axes[1].text(i + width/2, f + 0.002, f"{f:.4f}", ha="center", fontsize=9, rotation=45)
axes[1].set_ylim(0, 0.08)

plt.savefig("images/models_comparison.png", dpi=200)
plt.close()

print("All charts generated and saved to images/ successfully!")
