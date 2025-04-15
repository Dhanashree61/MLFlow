import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import mlflow

iris = datasets.load_iris()
iris_train = pd.DataFrame(iris.data,columns= iris.feature_names)
clf = RandomForestClassifier(max_depth=10,random_state=0)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Signature_exp")
with mlflow.start_run():
    clf.fit(iris_train, iris.target)

    input_example = iris_train.iloc[[0]]

    mlflow.sklearn.log_model(clf, 'iris_rf', input_example= input_example)