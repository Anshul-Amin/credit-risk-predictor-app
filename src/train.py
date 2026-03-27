import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import optuna
import joblib
import os

def load_data():
    return pd.read_csv("data/processed_german_credit.csv")

def build_preprocessor():
    numeric_features = ["Age", "Credit amount", "Duration"]
    categorical_features = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_features),
        ]
    )
    return preprocessor

def objective(trial, X, y, preprocessor):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "scale_pos_weight": y.value_counts()[0] / (y.value_counts()[1] + 1e-6) # good/bad -> class imbalance 70/30 (bad is 1)
    }

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**params))
    ])

    score = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()
    return score

def train_and_save():
    df = load_data()
    X = df.drop(["Risk", "Risk_label"], axis=1)
    y = df["Risk_label"] # 0 = Good, 1 = Bad

    # Ensure Job is string or category if it contains categories
    X['Job'] = X['Job'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = build_preprocessor()

    # Optuna optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, preprocessor), n_trials=15)

    print("Best hyperparameters: ", study.best_params)

    # Train final model
    best_params = study.best_params
    best_params.update({
        "random_state": 42,
        "eval_metric": "logloss",
        "scale_pos_weight": y_train.value_counts()[0] / (y_train.value_counts()[1] + 1e-6)
    })

    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**best_params))
    ])

    final_model.fit(X_train, y_train)

    # Evaluate
    roc_score = cross_val_score(final_model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"Train ROC AUC: {roc_score:.4f}")
    
    # Save model and test set for fairness audit
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/model.joblib")
    print("Model saved to models/model.joblib")
    
    # Save the test set for SHAP / Fairness evaluation
    test_df = X_test.copy()
    test_df["Risk"] = df.loc[X_test.index, "Risk"]
    test_df["Risk_label"] = y_test
    test_df.to_csv("data/test_data.csv", index=False)
    print("Test data saved to data/test_data.csv")

if __name__ == "__main__":
    train_and_save()
