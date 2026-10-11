# 🏦 Credit Risk Assessment Portal

An end-to-end Machine Learning web application designed to evaluate loan applicant credit risk in real time. The system predicts credit approval status and calculates a precise risk score using an **XGBoost** classification model, served via a high-performance **FastAPI** backend and an interactive **Streamlit** user interface.

The entire infrastructure is containerized and orchestrated using **Docker Compose**.

---

## 🚀 Key Features

- **Real-Time Risk Scoring:** Instantly processes applicant inputs to output risk percentages, status tags (`LOW RISK` / `HIGH RISK`), and clear approval decisions (`APPROVE!` / `REJECT`).
- **Decoupled Architecture:** Clean separation of concerns with a FastAPI REST API backend and a Streamlit frontend UI.
- **Dockerized Workflow:** Multi-container deployment managed seamlessly via Docker Compose with internal service networking.
- **Robust ML Pipeline:** Includes data preprocessing, feature engineering (ratio metrics and median imputation for missing values), and an optimized XGBoost model.

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) (Port `8501`)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Port `8000`) + Uvicorn
- **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy, Joblib
- **Containerization:** Docker, Docker Compose

---

## 📁 Project Structure

```text
credit_risk/
├── app.py                   # Streamlit user interface frontend
├── main.py                  # FastAPI inference backend
├── Dockerfile.backend       # Docker configuration for FastAPI
├── Dockerfile.frontend      # Docker configuration for Streamlit
├── docker-compose.yml       # Multi-container orchestration setup
├── requirements.txt         # Python package dependencies
├── credit_risk_model.pkl    # Trained XGBoost model artifact
└── model_columns.pkl        # Saved feature columns schema
```

Instructions

Clone the repository:
    
git clone [https://github.com/mustaphamustaphayev/credit-risk-ml-app.git](https://github.com/mustaphamustaphayev/credit-risk-ml-app.git)
cd credit-risk-ml-app


Build and start the containers:

docker compose up --build

Access the application in your browser:

Streamlit Web UI: http://localhost:8501

FastAPI Docs (Swagger): http://localhost:8000/docs

---

Local Development (Without Docker)

If you wish to run the services locally in separate development terminals:
Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

Run the FastAPI backend:

uvicorn main:app --reload

Run the Streamlit frontend (open a second terminal window):

streamlit run app.py
