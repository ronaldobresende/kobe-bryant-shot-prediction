from kedro.pipeline import node, Pipeline, pipeline
from .nodes import clean_data, prepare_train_test

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=dict(
                data="raw_data_dev",
                columns="params:train_columns"
            ),
            outputs="filtered_data",
            name="clean_data_node",
        ),
        node(
            func=prepare_train_test,
            inputs=dict(
                data="filtered_data",
                test_size="params:test_size",
                random_state="params:random_state"
            ),
            outputs=["train_data", "test_data"],
            name="prepare_train_test_node",
        )
    ])


