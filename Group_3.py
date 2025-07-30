import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(page_title="Loan Default Predictor - Group 3", layout="wide")

# ----------------------------------------
# LOAD TRAINED ARTIFACTS
# ----------------------------------------
model = joblib.load("gradient_boost_model.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

# ----------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Predictor", "About"],
        icons=["house", "activity", "info-circle"],
        default_index=1,
        styles={
            "container": {"padding": "0!important", "background-color": "#f9f9f9"},
            "icon": {"color": "#3b82f6", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#3b82f6", "color": "white"},
        }
    )

# ----------------------------------------
# HOME SECTION
# ----------------------------------------
if selected == "Home":
    st.title("📊 Welcome to Group 3's Loan Default Prediction App")
    st.markdown("""
    This Streamlit app uses a trained Gradient Boosting model to predict the likelihood of loan default based on customer attributes.
    
    Use the *Predictor* tab to input details and get a real-time prediction.

    ---
    """)

# ----------------------------------------
# ABOUT SECTION
# ----------------------------------------
elif selected == "About":
    st.title("ℹ About This App")
    st.markdown("""
    - *Project:* Loan Default Classification  
    - *Model Used:* Gradient Boosting Classifier  
    - *Team:* Group 3  
    - *Data Source:* credit_risk.csv (cleaned)  
    - *Target Variable:* Loan Status (0 = No Default, 1 = Default)

    ---
    """)

# ----------------------------------------
# PREDICTOR SECTION
# ----------------------------------------
elif selected == "Predictor":
    st.title("📈 Loan Default Predictor")

    st.markdown("### Enter applicant details:")

    # Example input fields — adjust based on your actual X_columns list
    Age = st.slider("Age", 18, 70, 30)
    Income = st.number_input("Monthly Income", min_value=0.0, step=100.0)
    Emp_length = st.number_input("Employment Length (years)", min_value=0.0, step=1.0)
    Amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
    Rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
    Percent_income = st.number_input("Percent of Income", min_value=0.0, step=0.01)
    Cred_length = st.slider("Credit History Length (years)", 0, 40, 5)

    Home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    Intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    Default = st.selectbox("Previous Default", ["Y", "N"])

    # Preprocess inputs
    input_dict = {
        "Age": Age,
        "Income": Income,
        "Home": Home,
        "Emp_length": Emp_length,
        "Intent": Intent,
        "Amount": Amount,
        "Rate": Rate,
        "Percent_income": Percent_income,
        "Cred_length": Cred_length,
        "Default": Default,
    }

    input_df = pd.DataFrame([input_dict])

    # Label encoding (ensure consistent ordering from training)
    encode_order = {
        "Home": {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3},
        "Intent": {"DEBTCONSOLIDATION": 0, "EDUCATION": 1, "HOMEIMPROVEMENT": 2, "MEDICAL": 3, "PERSONAL": 4, "VENTURE": 5},
        "Default": {"N": 0, "Y": 1}
    }

    for col, mapping in encode_order.items():
        input_df[col] = input_df[col].map(mapping)

    # Reorder columns and scale
    input_df = input_df[X_columns]
    input_scaled = scaler.transform(input_df)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][prediction]

        if prediction == 1:
            st.error(f"⚠ The applicant is likely to DEFAULT on the loan. (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ The applicant is NOT likely to default. (Confidence: {prob:.2%})")