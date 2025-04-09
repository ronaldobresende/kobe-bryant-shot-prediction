import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

#mlflow.set_tracking_uri("file:///c:/projects/python/kobe-bryant-shot-prediction/mlruns")

def clean_data(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    if mlflow.active_run():
        mlflow.end_run()
        
    with mlflow.start_run(run_name="PreparacaoDados", nested=True): 
        mlflow.log_metric("rows_before", data.shape[0])
        mlflow.log_metric("columns_before", data.shape[1])

        data = data[columns + ["shot_made_flag"]]
        data = data.dropna()

        mlflow.log_metric("rows_after", data.shape[0])
        mlflow.log_metric("columns_after", data.shape[1])

        return data
    
def prepare_train_test(data: pd.DataFrame, test_size: float, random_state: int):
    X = data.drop(columns=["shot_made_flag"])
    y = data["shot_made_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("train_size", train_data.shape[0])
    mlflow.log_metric("test_size", test_data.shape[0])

    return train_data, test_data