import streamlit as st
import requests 
import os

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

st.set_page_config(
    page_title="Credit Risk Portal",
    page_icon=":money_with_wings:",
    layout="wide"
)

st.title('Credit Risk Analysis Portal')
st.write("Enter the customer's data")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Information")
    person_age = st.number_input("Age:", min_value=18, max_value=100, value=28)
    person_income = st.number_input("Income:", min_value=1000, value=50000, step=1000)
    person_home_ownership = st.selectbox("Home Ownership:", ["OWN", "RENT", "MORTGAGE", "OTHER"])
    person_emp_length = st.number_input("Employment Length (years):", min_value=0.0, max_value=50.0, value=10.0, step=0.5)

with col2:
    st.subheader("Loan Details")
    loan_intent = st.selectbox("Loan Intent:", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox('Loan Grade:', ['A', "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount:", min_value=500, value=10000, step=500)
    loan_int_rate = st.number_input("Loan Interest Rate (%):", min_value=1.0, max_value=40.0, value=10.5, step=0.1)

with col3:
    st.subheader("History and Factors")
    loan_percent_income = st.number_input("Loan to Income Ratio (%):", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    cb_person_default_on_file = st.selectbox("Default on File:", ["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years):", min_value=0, value=4)

st.divider()


if st.button("Asses the Risk", use_container_width=True):
    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("Risk Assessment Result")

            col1, col2 = st.columns(2)
            col1.metric("Risk Score", result["risk_score"])
            col2.metric("Risk Status", result["risk_status"])

            if result["risk_status"] == "HIGH RISK":
                st.error(f"Decision: {result['decision']}")
            else:
                st.success(f"Decision: {result['decision']}")
        else:
            st.error("Invalid response from API server")
    except Exception as err:
        st.error(f"Unable to connect to the backend server. Make sure FastAPI is running. Error: {err}")