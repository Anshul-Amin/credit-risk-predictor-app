import pandas as pd
import numpy as np
import requests
import io
import os

def load_data():
    local_path = "../data/german_credit_data.csv"
    if not os.path.exists(local_path):
        local_path = "data/german_credit_data.csv"
        
    df = pd.read_csv(local_path, index_col=0)
    
    # Download the labels from UCI
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    response = requests.get(uci_url)
    
    # The UCI dataset space separated
    uci_df = pd.read_csv(io.StringIO(response.text), sep=" ", header=None)
    
    # Target column is the last one (index 20)
    # 1 is Good, 2 is Bad
    target = uci_df[20].map({1: 'Good', 2: 'Bad'})
    
    # Join the target
    df['Risk'] = target.values
    
    return df

def preprocess_data(df):
    print("Missing values before imputation:")
    print(df.isnull().sum())
    
    # Impute missing values with 'Unknown'
    df['Saving accounts'] = df['Saving accounts'].fillna('Unknown')
    df['Checking account'] = df['Checking account'].fillna('Unknown')
    
    # Also handle 'NA' strings if they were loaded as strings
    df['Saving accounts'] = df['Saving accounts'].replace('NA', 'Unknown')
    df['Checking account'] = df['Checking account'].replace('NA', 'Unknown')
    
    print("\nMissing values after imputation:")
    print(df.isnull().sum())
    
    # Convert Target to numeric: Good = 0, Bad = 1 for XGBoost
    df['Risk_label'] = df['Risk'].map({'Good': 0, 'Bad': 1})
    
    return df

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    
    print("\nPreprocessing data...")
    processed_data = preprocess_data(data)
    
    # Create target directory if not exists
    os.makedirs("data", exist_ok=True)
    if os.path.exists("../data"):
        out_path = "../data/processed_german_credit.csv"
    else:
        out_path = "data/processed_german_credit.csv"
        
    processed_data.to_csv(out_path, index=False)
    print(f"\nProcessed data saved to {out_path}")
    print("\nClass distribution:")
    print(processed_data['Risk'].value_counts(normalize=True))
