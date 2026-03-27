import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="wide")

# Constants
MODEL_PATH = "models/model.joblib"
METRICS_PATH = "data/fairness_metrics.json"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

model = load_model()
metrics = load_metrics()

st.title("🏦 Credit Risk Prediction System")
st.markdown("An end-to-end system focusing on **Responsible and Explainable AI**.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Credit Score Simulator", "🔍 Explainability (SHAP)", "⚖️ Fairness Dashboard"])

# Default inputs
if 'input_data' not in st.session_state:
    st.session_state.input_data = pd.DataFrame([{
        "Age": 30, "Sex": "male", "Job": "2", "Housing": "own",
        "Saving accounts": "little", "Checking account": "moderate",
        "Credit amount": 5000, "Duration": 24, "Purpose": "car"
    }])

if 'prediction_prob' not in st.session_state:
    st.session_state.prediction_prob = None

with tab1:
    st.header("Simulator")
    st.markdown("Enter applicant details to predict credit risk. A 'Good' outcome means the applicant is low risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        job_options = ["0", "1", "2", "3"]
        job = st.selectbox("Job Type (0-3)", job_options, index=2)
        
    with col2:
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "Unknown"])
        checking = st.selectbox("Checking account", ["little", "moderate", "rich", "Unknown"])
        
    with col3:
        amount = st.number_input("Credit amount", min_value=100, max_value=50000, value=5000, step=100)
        duration = st.number_input("Duration (months)", min_value=1, max_value=120, value=24)
        purpose = st.selectbox("Purpose", ["car", "furniture/equipment", "radio/TV", "domestic appliances", "repairs", "education", "business", "vacation/others"])
        
    if st.button("Predict Risk", type="primary"):
        input_dict = {
            "Age": [age], "Sex": [sex], "Job": [str(job)], "Housing": [housing],
            "Saving accounts": [saving], "Checking account": [checking],
            "Credit amount": [amount], "Duration": [duration], "Purpose": [purpose]
        }
        st.session_state.input_data = pd.DataFrame(input_dict)
        
        if model is not None:
            proba = model.predict_proba(st.session_state.input_data)[0]
            # 0 is Good, 1 is Bad
            prob_good = proba[0]
            prob_bad = proba[1]
            st.session_state.prediction_prob = (prob_good, prob_bad)
            
            if prob_good >= 0.5:
                st.success(f"**Approved!** Probability of Good Credit: {prob_good:.1%}")
            else:
                st.error(f"**Denied.** Probability of Bad Credit: {prob_bad:.1%}")
        else:
            st.warning("Model not found. Please train the model first.")

with tab2:
    st.header("Why was I denied/approved?")
    st.markdown("We use SHAP (SHapley Additive exPlanations) to explain feature contributions towards the decision.")
    
    if st.session_state.prediction_prob is None:
        st.info("Please make a prediction in the Simulator tab first.")
    elif model is not None:
        st.write("Generating SHAP explanation...")
        preprocessor = model.named_steps['preprocessor']
        xgb_model = model.named_steps['classifier']
        
        # Transform the single input
        X_trans = preprocessor.transform(st.session_state.input_data)
        
        # Get feature names
        num_features = ["Age", "Credit amount", "Duration"]
        cat_features = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
        cat_encoder = preprocessor.named_transformers_['cat']
        encoded_cat_features = cat_encoder.get_feature_names_out(cat_features)
        feature_names = num_features + list(encoded_cat_features)
        
        # Explainer
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_trans)
        # For a single sample
        shap_values_single = shap_values[0]
        
        # Plotly/Matplotlib Waterfall
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # We need an Explanation object for the waterfall plot
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
            
        explanation = shap.Explanation(values=shap_values_single, 
                                       base_values=base_value, 
                                       data=X_trans[0], 
                                       feature_names=feature_names)
                                       
        shap.waterfall_plot(explanation, show=False)
        st.pyplot(fig)
        
        st.caption("A positive SHAP value pushes the model towards predicting 'Bad Credit' (Class 1). A negative value pushes towards 'Good Credit' (Class 0).")

with tab3:
    st.header("Fairness Audit Dashboard")
    st.markdown("This dashboard tracks **Demographic Parity Ratios** calculated on the test dataset. A ratio close to 1.0 indicates fair outcomes across groups.")
    
    if metrics is None:
        st.warning("Fairness metrics not found. Run `src/fairness_audit.py`.")
    else:
        colA, colB = st.columns(2)
        with colA:
            st.metric("Parity Ratio (Female vs Male)", f"{metrics['dp_sex']:.3f}", 
                        delta="Ideal: 1.0", delta_color="off")
            st.info("Ratio of 'Good' predictions for females compared to males.")
            
        with colB:
            st.metric("Parity Ratio (Age <=25 vs >25)", f"{metrics['dp_age']:.3f}",
                        delta="Ideal: 1.0", delta_color="off")
            st.info("Ratio of 'Good' predictions for younger applicants (<=25) compared to older.")

        # Progress bars as visualize
        st.markdown("### Visual Overview")
        st.progress(min(1.0, metrics['dp_sex']), text="Female/Male Parity")
        st.progress(min(1.0, metrics['dp_age']), text="Young/Old Parity")
