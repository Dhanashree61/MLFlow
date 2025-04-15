<!-- python -m venv venv -->

<!-- .\venv\Scripts\activate

pip install -r requirements.txt -->

python -m mlflow ui --backend-store-uri "mlflowtesting"

mlflow server --backend-store-uri sqlite:///demo.db
