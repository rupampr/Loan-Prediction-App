import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))

# Title
st.title("🏦 Loan Approval Predictor")

st.write("Enter applicant details below:")

# Inputs
no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=12
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

total_asset = st.number_input(
    "Total Assets",
    min_value=0,
    value=1000000
)

# Encoding
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Predict Button
if st.button("Predict Loan Approval"):

    input_df = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'Total_asset': [total_asset]
    })

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(
        f"Approval Probability: {probability[0][1]*100:.2f}%"
    )