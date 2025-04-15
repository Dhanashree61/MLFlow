import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")


url = "https://raw.githubusercontent.com/rashida048/Datasets/refs/heads/master/StudentsPerformance.csv"
df = pd.read_csv(url)



## dropping missing values 
df.dropna(inplace=True)


print(df.head())

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


train,test=train_test_split(df,test_size=0.2,random_state=42)


def smooth_huber_loss(eval_df,_builtin_metrics):
    y_true = eval_df['target']
    y_pred = eval_df['prediction']
    delta = 1.0
    loss = np.where(np.abs(y_true - y_pred) < delta, 0.5 * ((y_true - y_pred) ** 2), delta * (np.abs(y_true - y_pred) - 0.5 * delta))
    return np.mean(loss)

smooth_huberloss_metric=mlflow.models.make_metric(
    eval_fn=smooth_huber_loss,
    greater_is_better=False
)


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
import matplotlib.pyplot as plt
def custom_target_scatter(eval_df,_builtin_metrics,artifact_dir):
    
    y_true = eval_df['target']
    y_pred = eval_df['prediction']
    plt.scatter(y_true, y_pred)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("True vs Predicted values")
    plot_path=artifact_dir+"/scatter_plot.png"
    plt.savefig(plot_path)
    return {"scatter_plot_artifact":plot_path}



exp=mlflow.set_experiment(experiment_name="experiment_evaluation_custom")
#train model
with mlflow.start_run(experiment_id=exp.experiment_id):
    n_estimators=56
    random_state=42
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(train.drop(columns=['math score']), train['math score'])
    mlflow.sklearn.log_model(model, "random_forest_model")
    artifacts_uri=mlflow.get_artifact_uri("random_forest_model")
    mlflow.evaluate(
        artifacts_uri,
        test,
        targets="math score",
        model_type="regressor",
        evaluators=["default"],
        extra_metrics=[smooth_huberloss_metric],
        custom_artifacts=[custom_target_scatter]
    )



