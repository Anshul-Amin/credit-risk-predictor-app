# 🏦 Credit Risk Prediction System

A production-ready Credit Risk Prediction System focusing on **Responsible and Explainable AI**.

## Overview
This system evaluates applicants for credit default risk based on demographic and financial data. We've enhanced the UCI German Credit Dataset, implemented a robust XGBoost model with Optuna optimization, and prioritized two critical aspects of modern ML:
1. **Explainability (XAI):** Clear feature attribution using SHAP.
2. **Fairness:** A dedicated fairness audit assessing demographic parity across Gender and Age.

## Project Structure
- `data/`: Processed data and fairness metrics output.
- `models/`: Serialized model and preprocessor (`model.joblib`).
- `src/`: Core Python modules for data pipeline, training, and auditing.
  - `preprocessing.py`: Ingests and merges target labels, performs imputation.
  - `train.py`: Trains an XGBoost classifier with Optuna Bayesian Search.
  - `fairness_audit.py`: Monitors the model for disparate impact against unprivileged demographic groups.
- `app/`: Production Streamlit application featuring a Credit Score Simulator, SHAP explanations, and Fairness Dashboard.

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute Pipeline
```bash
python src/preprocessing.py
python src/train.py
python src/fairness_audit.py
```

### 3. Run Application
```bash
streamlit run app/streamlit_app.py
```

## Explainability & Fairness
- **SHAP Integration:** The Streamlit app renders a SHAP waterfall plot explaining exactly *why* an applicant received their specific risk score.
- **Fairness Metrics:** We calculate the Demographic Parity Ratio for `Sex` (Female vs. Male) and `Age` (≤25 vs. >25) to ensure equitable lending behavior.
