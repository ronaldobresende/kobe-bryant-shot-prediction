from pycaret.classification import setup, compare_models
from sklearn.metrics import log_loss, f1_score
import pandas as pd
import mlflow
import json

def train_logistic_regression(train_data: pd.DataFrame):
    setup(data=train_data, target="shot_made_flag", session_id=42, html=False)
    model = compare_models(include=["lr"], n_select=1)
    return model

def train_decision_tree(train_data: pd.DataFrame):
    setup(data=train_data, target="shot_made_flag", session_id=42, html=False)
    model = compare_models(include=["dt"], n_select=1)
    return model

def save_metrics_locally(metrics, file_path):
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)

def log_metrics_to_mlflow(metrics, params):
    with mlflow.start_run(run_name="Treinamento", nested=True):
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

def select_best_model(lr_model, dt_model, test_data, metrics_paths):
    X_test = test_data.drop("shot_made_flag", axis=1)
    y_test = test_data["shot_made_flag"]

    lr_preds = lr_model.predict_proba(X_test)
    dt_preds = dt_model.predict_proba(X_test)

    lr_log_loss = log_loss(y_test, lr_preds)
    lr_f1_score = f1_score(y_test, lr_model.predict(X_test))

    dt_log_loss = log_loss(y_test, dt_preds)
    dt_f1_score = f1_score(y_test, dt_model.predict(X_test))

    save_metrics_locally({"log_loss": lr_log_loss, "f1_score": dt_f1_score}, metrics_paths["logistic_model_metrics"])
    save_metrics_locally({"log_loss": dt_log_loss, "f1_score": dt_f1_score}, metrics_paths["decision_tree_model_metrics"])
   
    # Escolha do melhor modelo
    log_loss_threshold = 0.5  # ponto de corte arbitrário para considerar "ruim"

    # Se a árvore tiver um log_loss muito alto, escolher regressão
    if dt_log_loss > log_loss_threshold:
        best_model = lr_model
        model_name = "LogisticRegression"
        best_log_loss = lr_log_loss
        best_f1 = lr_f1_score
    # Se os log loss forem parecidos (diferença pequena), dar preferência ao maior F1
    elif abs(dt_log_loss - lr_log_loss) < 0.1:
        if dt_f1_score > lr_f1_score:
            best_model = dt_model
            model_name = "DecisionTree"
            best_log_loss = dt_log_loss
            best_f1 = dt_f1_score
        else:
            best_model = lr_model
            model_name = "LogisticRegression"
            best_log_loss = lr_log_loss
            best_f1 = lr_f1_score
    # Se a árvore tiver log loss menor, usar ela
    else:
        best_model = dt_model
        model_name = "DecisionTree"
        best_log_loss = dt_log_loss
        best_f1 = dt_f1_score
    
    best_metrics = {"log_loss": best_log_loss, "f1_score": best_f1}

    print(f"O melhor modelo foi {model_name} com F1 Score de {best_f1:.4f} e Log Loss de {best_log_loss:.4f}")

    save_metrics_locally(best_metrics, metrics_paths["best_model_metrics"])

    metrics_to_log = {
        "logistic_log_loss": lr_log_loss,
        "logistic_f1_score": lr_f1_score,
        "decision_tree_log_loss": dt_log_loss,
        "decision_tree_f1_score": dt_f1_score,
        "best_log_loss": best_metrics["log_loss"],
        "best_f1_score": best_metrics["f1_score"],
    }
    params_to_log = {"best_model": model_name}
    log_metrics_to_mlflow(metrics_to_log, params_to_log)

    mlflow.sklearn.log_model(best_model, "model", registered_model_name="kobe_model")

    return best_model

