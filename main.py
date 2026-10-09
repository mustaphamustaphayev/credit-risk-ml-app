from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load('credit_risk_model.pkl')
model_columns = joblib.load('model_columns.pkl')

app = FastAPI(
    title='Real-Time Credit Prediction API',
    description="ML service that assesses a customer's credit risk in real time"
)

@app.get("/")
def home():
    return {"status": "Active!", "message": "Credit Risk Prediction API is active!"}

class CreditApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post("/predict")
def predict_credit_risk(application: CreditApplication):
    input_data = application.model_dump() if hasattr(application, "model_dump") else application.dict()
    df = pd.DataFrame([input_data])

    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['loan_to_age_ratio'] = df['loan_amnt'] / df['person_age']
    df['estimated_yearly_interest'] = (df['loan_amnt'] * df['loan_int_rate']) / 100

    df_encoded = pd.get_dummies(df)

    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    risk_probability = float(model.predict_proba(df_encoded)[0][1])

    is_risky = risk_probability > 0.5

    return {
        'risk_score': round(risk_probability, 4),
        'risk_percentage': f"{round(risk_probability * 100, 2)}%",
        'risk_status': 'HIGH RISK' if is_risky else 'LOW RISK',
        'decision': 'REJECT!' if is_risky else "APPROVE!"
    }