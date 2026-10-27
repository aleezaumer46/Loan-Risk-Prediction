# Intelligent Loan Risk Prediction System using Deep Learning (ANN)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-Educational-green)

---

## 📌 Project Overview

The **Intelligent Loan Risk Prediction System** is a production-grade Deep Learning application designed for commercial banking institutions. It automates credit risk evaluation by predicting whether a prospective loan applicant is at **Low Risk** (likely to repay) or **High Risk** (potential default).

Trained on **252,000 real-world banking records**, the system implements a 3-layer deep **Artificial Neural Network (ANN)**, complete data preprocessing pipelines, multi-model empirical benchmarks, dual-channel deployment (**Flask Web Application** and **FastAPI Microservice**), an empirical validation suite of 10 diverse customer profiles, strategic credit policies, and an enterprise MySQL database architecture.

---

## 🚀 Key Features

- **Deep Neural Network Classifier:** 3-layer feedforward ANN with ReLU and Sigmoid activations.
- **Dual Deployment Options:**
  - **Flask Web Application:** Clean, responsive UI for branch loan officers and underwriters with instant risk categorization, calibrated confidence scores, and action recommendations.
  - **FastAPI RESTful Microservice:** Asynchronous API with Pydantic validation, `/predict` and `/predict/batch` endpoints, and interactive Swagger documentation at `/docs`.
- **Pre-trained Pipeline Artifacts:** Pre-fitted `StandardScaler` and `LabelEncoder` objects prevent training-serving skew.
- **Empirical 10-Customer Test Suite:** Automated testing module evaluating diverse applicant profiles with financial reasoning.
- **Comprehensive PDF Project Report:** 7-page publication-ready academic and technical report (`documentation/Loan_Risk_Prediction_Project_Report.pdf`).
- **Interactive Jupyter Notebook:** Fully documented notebook (`Loan_Risk_Prediction.ipynb`) covering all 15 development phases.
- **Enterprise MySQL Relational Schema:** Production-ready DDL (`database/schema.sql`) for audit trails, underwriting sign-offs, and compliance.

---

## 🛠️ Technologies Used

| Category | Tools & Libraries |
|---|---|
| **Programming Language** | Python 3.11 |
| **Deep Learning Framework** | TensorFlow, Keras |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, Joblib |
| **Web Frameworks** | Flask (Jinja2, HTML5, CSS3), FastAPI, Uvicorn, Pydantic |
| **Visualization & Reporting** | Matplotlib, Seaborn, ReportLab, PyPDF, Nbformat |
| **Database** | MySQL / MariaDB (Enterprise DDL Schema) |

---

## 📊 Dataset & Input Features

The model analyzes 11 borrower features across financial, demographic, employment, and geographical dimensions:

| Feature | Type | Description |
|---|---|---|
| `Income` | Numeric | Annual gross income (in local currency / INR) |
| `Age` | Numeric | Applicant age in years |
| `Experience` | Numeric | Total professional work experience in years |
| `CURRENT_JOB_YRS` | Numeric | Tenure in current job in years |
| `CURRENT_HOUSE_YRS` | Numeric | Tenure at current residence in years |
| `Married/Single` | Categorical | Marital status (`single`, `married`) |
| `House_Ownership` | Categorical | Residential status (`owned`, `rented`, `norent_noown`) |
| `Car_Ownership` | Categorical | Vehicle asset status (`yes`, `no`) |
| `Profession` | Categorical | Occupation title (51 distinct categories) |
| `CITY` | Categorical | City of residence (317 unique cities) |
| `STATE` | Categorical | State of residence (29 distinct Indian states) |

**Target Variable:**
- `Risk_Flag`: **`0 = Low Risk`** (Likely to repay) | **`1 = High Risk`** (Likely to default)

---

## 🧠 Deep Learning Architecture & Training

Three candidate Artificial Neural Network architectures were engineered, trained, and benchmarked:

```
Model A (1-Layer):   Input (11) ──► Dense(16, ReLU) ──► Dense(1, Sigmoid)
Model B (2-Layers):  Input (11) ──► Dense(16, ReLU) ──► Dense(8, ReLU) ──► Dense(1, Sigmoid)
Model C (3-Layers):  Input (11) ──► Dense(32, ReLU) ──► Dense(16, ReLU) ──► Dense(8, ReLU) ──► Dense(1, Sigmoid)
```

### Training Specifications:
- **Optimizer:** Adam ($\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Loss Function:** Binary Cross-Entropy
- **Hidden Activations:** Rectified Linear Unit (ReLU)
- **Output Activation:** Sigmoid ($\sigma(z) \in [0, 1]$)
- **Batch Size:** 64 | **Epochs:** 20

---

## 📈 Model Performance & Selection

| Model | Accuracy | Precision | Recall | F1 Score | Decision |
|---|---|---|---|---|---|
| **Model A** | 87.70% | 1.0000 | 0.0002 | 0.0003 | Rejected (Zero default detection) |
| **Model B** | 87.68% | 0.4242 | 0.0045 | 0.0089 | Rejected (Weak recall) |
| **Model C** | **87.66%** | **0.4798** | **0.0345** | **0.0644** | **SELECTED FOR DEPLOYMENT** |

### 🎯 Why Model C?
In imbalanced retail lending (87.7% non-defaulters vs 12.3% defaulters), accuracy is deceptive: a dummy classifier predicting 0 for all applicants achieves 87.7% accuracy while missing 100% of defaults. Model C provides a **172x improvement in recall** and the highest F1-Score, making it significantly more effective at protecting bank reserves.

---

## 📂 Project Structure

```
Loan-Risk-Prediction/
│
├── Loan_Risk_Prediction.ipynb       # Complete 15-phase executed Jupyter Notebook
├── app.py                           # Flask Web Application
├── fastapi_app.py                   # FastAPI RESTful Microservice (Bonus deliverable)
├── test_customers.py                # Phase 13: 10 customer empirical test suite
├── generate_visualizations.py       # Script generating all EDA & evaluation charts
├── generate_pdf_report.py           # Script compiling the publication PDF report
│
├── Loan Prediction.csv              # Full dataset (252,000 rows)
├── loan_risk_ann.keras              # Serialized production ANN model (Model C)
├── scaler.pkl                       # Serialized StandardScaler
├── label_encoders.pkl               # Serialized LabelEncoders
│
├── requirements.txt                 # Project dependencies
├── Procfile                         # Cloud deployment configuration
├── README.md                        # Project documentation
│
├── database/
│   └── schema.sql                   # Phase 15: Enterprise MySQL Relational Schema
│
├── documentation/
│   ├── Deep Learning Project.docx   # Original project specification
│   ├── Loan_Risk_Prediction_Project_Report.pdf  # Professional 7-page PDF report
│   └── test_results_10_customers.json # Phase 13 test execution data
│
├── templates/
│   └── index.html                   # Responsive banking form & results template
│
├── static/
│   └── style.css                    # Modern CSS design system
│
└── images/
    ├── target_distribution.png      # Class imbalance chart
    ├── numeric_distributions.png    # Feature distribution histograms
    ├── correlation_matrix.png       # Correlation heatmap
    ├── models_comparison.png        # Model A vs B vs C benchmark
    ├── confusion_matrix.png         # Model C confusion matrix
    ├── roc_curve.png                # Receiver Operating Characteristic curve
    ├── home.png                     # Flask Web App form screenshot
    └── prediction.png               # Flask Web App result screenshot
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AleezaUmer/Loan-Risk-Prediction.git
cd Loan-Risk-Prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🌐 Running the Applications

### Option A: Run the Flask Web Application
```bash
python app.py
```
Open your browser at:
```
http://127.0.0.1:5000
```

### Option B: Run the FastAPI Microservice (Bonus)
```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

#### Example FastAPI Request (`POST /predict`):
```json
{
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
```

#### Example FastAPI Response:
```json
{
  "prediction": "Low Risk",
  "risk_flag": 0,
  "probability": 0.1342,
  "risk_percentage": 13.42,
  "confidence": 86.58,
  "risk_tier": "Ultra Low Risk",
  "recommendation": "Fast-track approval with prime interest rate and standard loan covenants."
}
```

---

## 🧪 Phase 13: 10 Customer Empirical Test Suite

To run the automated empirical test suite:
```bash
python test_customers.py
```

| ID | Profile Summary | Income (₹) | Age | Prediction | Prob | Action |
|---|---|---|---|---|---|---|
| **CUST-001** | Mechanical Engineer, single, rented, low tenure | 1,303,834 | 23 | Low Risk | 3.86% | Instant Approval |
| **CUST-002** | Corporate Physician, single, high income | 7,574,516 | 40 | Low Risk | 13.42% | Instant Approval |
| **CUST-003** | Senior Developer, married, owned home, car owner | 8,945,000 | 45 | Low Risk | 2.77% | Prime Fast-Track |
| **CUST-004** | Secretary, single, rented home, no vehicle | 6,868,118 | 25 | **High Risk** | 63.34% | Manual Audit / Decline |
| **CUST-005** | Librarian, single, rented apartment, no vehicle | 6,325,118 | 21 | **High Risk** | 67.15% | Manual Audit / Decline |
| **CUST-006** | Technical Writer, single, rented housing | 9,861,298 | 59 | **High Risk** | 57.87% | Strict Review / Collateral |
| **CUST-007** | Senior Civil Servant, married, owned property | 6,200,000 | 54 | Low Risk | 0.16% | Prime Fast-Track |
| **CUST-008** | Chartered Accountant, married, high income | 9,200,000 | 48 | Low Risk | 1.54% | Prime Fast-Track |
| **CUST-009** | Hotel Manager, entry-level, rented, mobile | 1,850,000 | 25 | Low Risk | 19.59% | Standard Approval |
| **CUST-010** | Police Officer, married, owned home, steady career | 4,800,000 | 43 | Low Risk | 1.54% | Prime Fast-Track |

---

## 📈 Phase 14: Strategic Banking Recommendations

Based on empirical model scores, banks should deploy a **Four-Tier Credit Policy Framework**:

1. **Ultra Low Risk (`< 15%` Default Probability):**  
   - Instant digital approvals with automated documentation.
   - Competitive interest rates (-50 bps discount) to attract prime borrowers.
2. **Low Risk (`15% - 35%` Default Probability):**  
   - Standard underwriting with routine salary verification.
3. **Moderate Risk (`35% - 50%` Default Probability):**  
   - Mandatory manual underwriting.
   - Require secondary co-signers or 10-15% equity collateral.
4. **High / Critical Risk (`> 50%` Default Probability):**  
   - Automatic application decline or requirement of 100% liquid asset lien.

---

## 🗄️ Phase 15: Enterprise Database Architecture

The system includes a production MySQL schema (`database/schema.sql`) modeling 5 core banking entities:
- `customers`: Master applicant identity, demographic data, and residency.
- `customer_financials`: Historical income, verified job tenure, and property holdings.
- `loan_applications`: Loan requests, amounts, terms, and approval workflows.
- `prediction_history`: AI audit log capturing model version, raw sigmoid score, confidence, and JSON feature payload for regulatory compliance (Model Risk Management / SR 11-7).
- `underwriting_decisions`: Human underwriter sign-offs, counter-offers, and notes.

---

## 📄 Rebuilding Reports & Artifacts

- **Regenerate Visualization Figures:** `python generate_visualizations.py`
- **Regenerate PDF Project Report:** `python generate_pdf_report.py`
- **Regenerate Jupyter Notebook:** `python create_notebook.py`

---

## 👩‍💻 Author & Acknowledgements

**Aleeza Umer**  
*Bachelor of Science in Artificial Intelligence (BSAI)*  
GitHub: [@AleezaUmer](https://github.com/AleezaUmer) &bull; Repository: [Loan-Risk-Prediction](https://github.com/AleezaUmer/Loan-Risk-Prediction)

---

## 📄 License
This project is created for educational, portfolio, and research purposes.
