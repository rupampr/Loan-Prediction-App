import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("loan_model.pkl", "rb"))

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🏦 Loan Approval Predictor")

st.sidebar.info(
    """
    This application predicts whether a loan is likely to be approved
    based on applicant details using a Random Forest Machine Learning model.
    """
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🏦 Loan Approval Predictor")

st.markdown(
    """
    Enter the applicant details below and click **Predict Loan Approval**.
    """
)

# -----------------------------
# APPLICANT DETAILS
# -----------------------------
st.subheader("👤 Applicant Information")

st.caption(
    "Dependents are family members who rely on your income for financial support "
    "(e.g., children, spouse, parents)."
)

no_of_dependents = st.selectbox(
    "Number of Dependents",
    [0, 1, 2, 3, 4, 5]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

# -----------------------------
# FINANCIAL DETAILS
# -----------------------------
st.subheader("💰 Financial Information")

income_lakh = st.number_input(
    "Annual Income (₹ Lakhs)",
    min_value=0.0,
    value=10.0,
    step=0.5,
    help="Example: 10 means ₹10 Lakhs per year"
)

loan_amount_lakh = st.number_input(
    "Loan Amount Required (₹ Lakhs)",
    min_value=0.0,
    value=5.0,
    step=0.5
)

total_asset_lakh = st.number_input(
    "Total Assets (₹ Lakhs)",
    min_value=0.0,
    value=20.0,
    step=0.5,
    help="Include bank balance, property, investments, vehicles, etc."
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

# -----------------------------
# ENCODING
# -----------------------------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Convert Lakhs to Rupees
income_annum = income_lakh * 100000
loan_amount = loan_amount_lakh * 100000
total_asset = total_asset_lakh * 100000

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Loan Approval"):

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

    approval_probability = probability[0][1] * 100

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(
        f"**Approval Probability:** {approval_probability:.2f}%"
    )

    st.progress(int(approval_probability))

    # Applicant Summary
    st.subheader("📋 Applicant Summary")

    st.write(f"**Dependents:** {no_of_dependents}")
    st.write(
        f"**Education:** {'Graduate' if education == 1 else 'Not Graduate'}"
    )
    st.write(
        f"**Self Employed:** {'Yes' if self_employed == 1 else 'No'}"
    )
    st.write(f"**Annual Income:** ₹{income_lakh:.2f} Lakhs")
    st.write(f"**Loan Amount:** ₹{loan_amount_lakh:.2f} Lakhs")
    st.write(f"**Total Assets:** ₹{total_asset_lakh:.2f} Lakhs")
    st.write(f"**Loan Term:** {loan_term} Months")
    st.write(f"**CIBIL Score:** {cibil_score}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "Built using Streamlit, Scikit-Learn, Pandas and Random Forest Classifier."
)