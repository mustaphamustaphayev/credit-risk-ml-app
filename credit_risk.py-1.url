import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

file_path = r'C:\Users\ASUS\Desktop\credit_risk\credit_risk_dataset.csv'
df = pd.read_csv(file_path)

print(df.isnull().sum().sort_values(ascending=False))

df = df[df['person_age'] < 100]
df = df[df['person_emp_length'] < 60]

df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
df['loan_to_age_ratio'] = df['loan_amnt'] / df['person_age']
df['estimated_yearly_interest'] = (df['loan_amnt'] * df['loan_int_rate']) / 100

df['loan_int_rate'] = df['loan_int_rate'].fillna(
    df['loan_int_rate'].median()
)
df['person_emp_length'] = df['person_emp_length'].fillna(
    df['person_emp_length'].median()
)
df['estimated_yearly_interest'] = df['estimated_yearly_interest'].fillna(
    df['estimated_yearly_interest'].median()
)

print(df.head())

X = df.drop(['loan_status'], axis=1)
y = df['loan_status']

print(df['loan_intent'].unique())

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    depth=5,
    random_state=42,
    eval_metric='logloss',
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print('---Model Performance---')
print(classification_report(y_test, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_proba)}')

joblib.dump(model, 'credit_risk_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

print('done!')
print(df.columns)
