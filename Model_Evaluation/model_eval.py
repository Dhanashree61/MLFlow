import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
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

train, test = train_test_split(df, test_size=0.2,random_state=23)


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
exp = mlflow.set_experiment(experiment_name='evaluation_Experiment')

with mlflow.start_run(experiment_id=exp.experiment_id):
    n_estimator = 78
    randam_state = 23
    model = RandomForestRegressor(n_estimators = n_estimator,random_state=randam_state)
    model.fit(train.drop(columns = ['math score']), train['math score'])
    mlflow.sklearn.log_model(model,"Randon_forest_model")
    artifacts_uri = mlflow.get_artifact_uri('Randon_forest_model')
    mlflow.evaluate(
        artifacts_uri,
        test,
        targets = 'math score',
        model_type = 'regressor',
        evaluators= ['default']
    )