import pandas as pd
import numpy as np
import joblib

def calculate_demographic_parity_ratio(y_pred, protected_attr, priv_val, unpriv_val):
    # P(Y_pred = 1 | protected_attr = unprivileged) / P(Y_pred = 1 | protected_attr = privileged)
    # Notice our classes: 0 = Good (Positive outcome), 1 = Bad (Negative outcome)
    # So we want P(Y_pred = 0 | unpriv) / P(Y_pred = 0 | priv)
    
    priv_outcomes = y_pred[protected_attr == priv_val]
    unpriv_outcomes = y_pred[protected_attr == unpriv_val]
    
    if len(priv_outcomes) == 0 or len(unpriv_outcomes) == 0:
        return np.nan
        
    prob_priv_good = np.mean(priv_outcomes == 0)
    prob_unpriv_good = np.mean(unpriv_outcomes == 0)
    
    if prob_priv_good == 0:
        return 0.0
        
    return prob_unpriv_good / prob_priv_good

def audit_fairness():
    print("Loading test dataset and model...")
    df = pd.read_csv("data/test_data.csv")
    
    # Ensure Job is string
    df['Job'] = df['Job'].astype(str)
    
    X_test = df.drop(["Risk", "Risk_label"], axis=1)
    y_true = df["Risk_label"]
    
    model = joblib.load("models/model.joblib")
    y_pred = model.predict(X_test)
    
    print("\n--- Fairness Audit ---")
    
    # 1. Sex
    # 'male' vs 'female'
    protected_attr_sex = df['Sex']
    dp_sex = calculate_demographic_parity_ratio(y_pred, protected_attr_sex, 'male', 'female')
    print(f"Demographic Parity Ratio (Sex: Female vs Male): {dp_sex:.3f}")
    
    # 2. Age
    # Binary threshold: > 25 (priv) vs <= 25 (unpriv)
    protected_attr_age = np.where(df['Age'] > 25, 'older', 'younger')
    dp_age = calculate_demographic_parity_ratio(y_pred, protected_attr_age, 'older', 'younger')
    print(f"Demographic Parity Ratio (Age: <=25 vs >25): {dp_age:.3f}")
    
    # Save these metrics to a file so Streamlit can read them
    metrics = {
        "dp_sex": dp_sex,
        "dp_age": dp_age
    }
    
    import json
    with open("data/fairness_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("\nFairness metrics saved to data/fairness_metrics.json")

if __name__ == "__main__":
    audit_fairness()
