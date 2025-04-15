import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Load dataset from a URL
url = "https://raw.githubusercontent.com/rashida048/Datasets/refs/heads/master/StudentsPerformance.csv"
df = pd.read_csv(url)

# Display first few rows
print(df.head())

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

exp = mlflow.set_experiment(experiment_name='Experiment1')

with mlflow.start_run(experiment_id=exp.experiment_id):
    n_estimators=80
    random_state=22
    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('random_state',random_state)
    mlflow.log_metric('mean_absolute_error',mae)
    mlflow.log_metric('r2_score',r2)
    mlflow.sklearn.log_model(model,'MY_model')