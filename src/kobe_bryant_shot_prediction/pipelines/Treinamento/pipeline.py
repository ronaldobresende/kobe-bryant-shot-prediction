from kedro.pipeline import node, Pipeline, pipeline 
from .nodes import (
    train_logistic_regression,
    train_decision_tree,
    select_best_model,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_regression,
            inputs="train_data",
            outputs="logistic_model",
            name="train_logistic_regression_node",
        ),
        node(
            func=train_decision_tree,
            inputs="train_data",
            outputs="decision_tree_model",
            name="train_decision_tree_node",
        ),
        node(
            func=select_best_model,
            inputs=[
                "logistic_model",
                "decision_tree_model",
                "test_data",
                "params:metrics_paths",
            ],
            outputs="final_model",
            name="select_best_model_node",
        ),
    ])