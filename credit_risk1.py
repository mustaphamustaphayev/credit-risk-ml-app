import pandas as pd
import numpy as np

file_path = r'C:\Users\ASUS\Desktop\credit_risk\credit_risk_dataset.csv'
df = pd.read_csv(file_path)

# print(df.shape)
# print(df.head(10))
# print(df.info())
# print(df.columns)
# print((df['loan_amnt'] == 0).sum()) there is no 0 value
print(df.isnull().sum().sort_values(ascending=False)) # columns with the missing values are loan int rate and person emp length 

# Firstly cleaning and organising the dataset

df = df[df['person_age'] < 100]
df = df[df['person_emp_length'] < 60]

# Feature Engineering: burda yaradacagimiz sutunlarin sadece datalari pkl faylinda olacaq, dusturlar orda olmayacaq ona gore de main.py faylinda fengineering edeceyik yeniden
df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt'] 
df['loan_to_age_ratio'] = df['loan_amnt'] / df['person_age']
df['estimated_yearly_interest'] = (df['loan_amnt'] * df['loan_int_rate']) / 100

# missing value imputation
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['estimated_yearly_interest'] = df['estimated_yearly_interest'].fillna(df['estimated_yearly_interest'].median())

print(df.head())
# print(df.shape)
# print(df.isnull().sum())
# print(df[['income_to_loan_ratio', 'loan_to_age_ratio', 'estimated_yearly_interest']].head())

######

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

X = df.drop(['loan_status'], axis=1) # loan status is the target variable, 1 are payer ones and 0 are defaulter ones 
y = df['loan_status']

print(df['loan_intent'].unique()) # i want to chech it just out of curiosity, 6 unique values

# One Hot Encoding 
X = pd.get_dummies(X, drop_first=True) # get dummies only focus on the categorical variables and implements the one hot encoding 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)# our train test distribution is 0.2 and 0.8, stratify
# is used to maintain the same distribution in target variable 0s and 1sin both train and test set 

model = XGBClassifier(n_estimators=100, learning_rate=0.1, depth=5, random_state=42, eval_metric='logloss') # this model is used for classification problems, n estimators is the 
# numbers of trees so 100 trees , first tree looks data and makes mistakes and then second tree looks mistakes and corrects it and so on
# learning rate is the step size at each iteration while moving toward a minimum of a loss function, and maximum depth is the 5

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # this is the probability of the positive class, which is 1 in our clas

print('---Model Performance---')
print(classification_report(y_test, y_pred)) # her bir sinif meselen kreditini odeyen ve odemeyenler ucun precision, recall, f1 score ve support deyerlerini gosterir
# precision deqiqlikdir, yeni model kreditini odemeyecek 1 dediyi musterilerin real olaraq nece faizi kreditini odemedyini gosterir
# recall ise modelin datadaki firildacilarin(1) ne qederini dogru tapdigini gosterir . bunu bir numune ile dusunek, meselen model deyir ki 10 neferden subhelenirem
# bunlar firildaqcidir amma yoxlanilib gorulub ki 10 neferden 8i firildaqcidir, bu zaman precision 0.8 olur, amma meselen umumi olaraq 10 nefer firildaqci var ve
# model 7sini tapib cixarib, bu zaman recall 0.7 olur. precision iddia etdiklerinin nece faiznin dogru cixmasi, recall ise umumi hedefin nece faizini tapmasidi
# support test ucun ayirdigimiz datada nece neferin 0 ve 1 oldugunu gosterir
# f1 score ise precision ve recall deyerlerinin balanslasdirilmis halidir, eger 1.00a yaxindirsa demeli model riskli adamlari elden qacirmir(ruksek recall)
# hemde yaxsi adamlari sehven riskli cixarmir(yuksek precision)
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

########################

import joblib # python obyektlerini komputerin diskinde fayl kimi saxlamaq ve  sonradan RAMa hazir sekilde berpa etmek ucun istifade edilir 
# serialization - obyekti fayla cevirmek.. deserialization - fayli obyekte cevirmek..
# vs codu baglayanda RAM temizlenir ve belece bu obyekt silinir bir nov amma joblib.dump ile pkl fayli ile diskde saxlanilir ki basqa bir faylda joblib.load ile qaytarirsan

joblib.dump(model, 'credit_risk_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

print('done!')
print(df.columns)