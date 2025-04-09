import pandas as pd
import mlflow
from sklearn.metrics import log_loss, f1_score
import numpy as np

def load_best_model(_):
    """
    Carrega o modelo mais recente registrado no MLflow Model Registry.
    """
    model_name = "kobe_model"
    
    # Obter a última versão do modelo registrado
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None", "Production", "Staging"])[-1].version

    # Construir o URI do modelo com a última versão
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def apply_model_to_production_data(model, prod_data: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    prod_data = prod_data.copy()
    X = prod_data[train_columns]
    prod_data["prediction"] = model.predict(X)
    return prod_data

def log_metrics_and_artifact(predictions: pd.DataFrame, output_path: str)  -> pd.DataFrame:
    if mlflow.active_run():
        mlflow.end_run()
    
    if "shot_made_flag" in predictions.columns:
        y_true = predictions["shot_made_flag"]
        y_pred = predictions["prediction"]

        # Preencher valores ausentes com 0 ou 1 aleatoriamente
        y_true = y_true.fillna(np.random.choice([0, 1]))    

        if len(y_true) > 0:
            with mlflow.start_run(run_name="PipelineAplicacao", nested=True):
                mlflow.log_artifact(output_path, artifact_path="predictions")
                mlflow.log_metric("log_loss_prod", log_loss(y_true, y_pred))
                mlflow.log_metric("f1_score_prod", f1_score(y_true, y_pred))
        else:
            print("Sem dados válidos para métricas.")
    else:
        print("Coluna 'shot_made_flag' ausente, pulando métricas.")



