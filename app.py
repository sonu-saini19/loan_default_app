import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="centered"
)

# ===============================
# Load model & encoders
# ===============================
model = joblib.load("loan_default_model.pkl")
encoders = joblib.load("encoder.pkl")

# ===============================
# App Title
# ===============================
st.title("Smart Loan Default Predictor")

st.title("💰 Smart Loan Default Predictor")
st.markdown(
    "Fill the applicant details in the sidebar and click **Predict** to check loan default risk."
)
st.sidebar.header("📋 Applicant Information")

# ===============================
# Numerical Inputs
# ===============================


person_age = st.sidebar.number_input("Age", 18, 100, 30)
person_income = st.sidebar.number_input("Annual Income", 0, value=50000)
person_emp_length = st.sidebar.number_input("Employment Length (years)", 0, 50, 5)
loan_amnt = st.sidebar.number_input("Loan Amount", 0, value=10000)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", 0.0, value=10.0)
loan_percent_income = st.sidebar.number_input("Loan % of Income", 0.0, value=0.2)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", 0, value=5)

# ===============================
# Categorical Inputs
# ===============================
person_home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

loan_grade = st.sidebar.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

cb_person_default_on_file = st.sidebar.selectbox(
    "Previous Default",
    ["Y", "N"]
)


# ===============================
# Encode categorical values
# ===============================
person_home_ownership_enc = encoders["person_home_ownership"].transform(
    [person_home_ownership]
)[0]

loan_intent_enc = encoders["loan_intent"].transform(
    [loan_intent]
)[0]

cb_person_default_on_file_enc = encoders["cb_person_default_on_file"].transform(
    [cb_person_default_on_file]
)[0]

loan_grade_enc = encoders["loan_grade"].transform(
    [loan_grade]
)[0]

# ===============================
# Prediction
# ===============================
st.markdown("---")  # horizontal line

predict_btn = st.button("🔍 Predict Loan Default")


if st.button("Predict Loan Default"):

    # Create input in correct order
    input_data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "person_home_ownership": person_home_ownership_enc,
        "loan_intent": loan_intent_enc,
        "loan_grade": loan_grade_enc,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "cb_person_default_on_file": cb_person_default_on_file_enc
    }
    
# 🔑 reorder columns EXACTLY as model expects
    input_df = pd.DataFrame(
        [input_data],
        columns=model.feature_names_in_
    )
    proba = model.predict_proba(input_df)[0][1]

    st.info(f"📊 Default Probability: {proba*100:.2f}%")


    prediction = model.predict(input_df)[0]
    st.markdown("## Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk: Customer is likely to DEFAULT on the loan.")
    else:
        st.success("✅ Low Risk: Customer is NOT likely to default.")

# ===============================
# Debug (optional – comment later)
# ===============================
# st.write("Expected features:", model.feature_names_in_)
# st.write("Input features:", input_df.columns.tolist())
