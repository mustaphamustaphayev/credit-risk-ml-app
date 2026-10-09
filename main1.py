from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("credit_risk_model.pkl") # onceki faylda joblib.dump ile diske yazdiqlarimizi cagiririq
model_columns = joblib.load("model_columns.pkl")

app = FastAPI(
    title = "Real-Time Credit Risk Prediction API",
    description = "ML service that assesses a customer's credit risk in real time"
)

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

@app.get("/")
def home():
    return {"status": "Active", "message": "Credit Risk Prediction API is active"}

@app.post("/predict")
def predict_credit_risk(application: CreditApplication):
    input_data = application.model_dump() if hasattr(application, "model_dump") else application.dict()# model_dump daha yaxsi ve yeni versiyadir amma pydantic v2 
    # modelinde movcuddur, pydantic v1de ise dict var idi. bu yazdigimiz setir model_d olub olmadigini yoxlayir bizde pyv2dir deye model_d edecek
    df = pd.DataFrame([input_data])

    # feature engineering edeceyik yene, cunki hem pkl fayli bu sutunlarin dusturlarini saxlamir hemde musteriler ucun de bu datalari daxil etmek cetin olacaq ona gore burda yeniden hesablayacagiq
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['loan_to_age_ratio'] = df['loan_amnt'] / df['person_age']
    df['estimated_yearly_interest'] = (df['loan_amnt'] * df['loan_int_rate']) / 100

    # sadece bir setile one hot encoding ede bilirik
    df_encoded = pd.get_dummies(df)

    # sutunlari modelin oyrendiyi deqiq sira ile yeniden duzuruk
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)# datada olmayan kateqoriya sutunu varsa yerine 0 yazacaq fill value

    risk_probabilty = float(model.predict_proba(df_encoded)[0][1])

    is_risky = risk_probabilty > 0.5

    return {
        "risk score": round(risk_probabilty, 4),
        "risk_percentage": f"{round(risk_probabilty * 100, 2)}%",
        "risk_status": "HIGH RISK" if is_risky else "LOW RISK",
        "decision": "REJECT!" if is_risky else "APPROVE!"
    }






