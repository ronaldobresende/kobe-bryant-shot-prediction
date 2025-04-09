from kedro.pipeline import node, Pipeline, pipeline
from .nodes import (
    load_best_model,
    apply_model_to_production_data,
    log_metrics_and_artifact,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_best_model,
            inputs="final_model",
            outputs="best_model",
            name="load_best_model_node"
        ),
        node(
            func=apply_model_to_production_data,
            inputs=["best_model", "raw_data_prod", "params:train_columns"],
            outputs="prod_predictions",
            name="apply_model_node"
        ),
        node(
            func=log_metrics_and_artifact,
            inputs=["prod_predictions", "params:prod_predictions_path"],
            outputs=None,
            name="log_metrics_node"
        )
    ])

