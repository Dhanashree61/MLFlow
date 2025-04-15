import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os

# Load dataset from a URL
url = "https://raw.githubusercontent.com/rashida048/Datasets/refs/heads/master/StudentsPerformance.csv"
df = pd.read_csv(url)
os.makedirs('data',exist_ok=True)
df.to_csv("data/studentperformace.csv")



# Display first few rows
# print(df.head())

# Handle missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Select features and target
X = df.drop(columns=['math score'])  # Features
y = df['math score']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing: Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
print('MLflow tracking using uri :',mlflow.get_tracking_uri())
# 
exp = mlflow.set_experiment(experiment_name='Experiment1')

n_estimators = range(1,5)
random_state = range(1,5)
#1,2,3,4
for estimators, rand_states in zip(n_estimators,random_state):
    with mlflow.start_run(experiment_id=exp.experiment_id):
        # n_estimators=80
        # random_state=22
        params = {'n_estimators':estimators,'random_state':rand_states}
        # Train model
        model = RandomForestRegressor(n_estimators= params["n_estimators"], random_state=params["random_state"])
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {'mae':mae,'r2':r2}
        # mlflow.log_param('n_estimators',n_estimators)
        # mlflow.log_param('random_state',random_state)
        # mlflow.log_metric('mean_absolute_error',mae)
        # mlflow.log_metric('r2_score',r2)
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

        mlflow.sklearn.log_model(model,'MY_model')
        mlflow.log_artifacts("data/")

        # mlflow.end_run()
        run = mlflow.last_active_run()
        print(f'Run ID: {run.info.run_id}')
        print(f'Run URI: {run.info.run_name}')
        print("---------------------------------")