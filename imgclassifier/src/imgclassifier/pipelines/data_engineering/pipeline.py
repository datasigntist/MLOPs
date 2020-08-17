from kedro.pipeline import node, Pipeline
from .nodes import (
    load_Data,
    flatten_and_Scale_Data
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_Data,
                inputs=None,
                outputs="loaded_Data",
                name="loading_Data",
            ),
            node(
                func=flatten_and_Scale_Data,
                inputs="loaded_Data",
                outputs="flattened_and_Scaled_Data",
                name="flattening_and_Scaling_Data",
            ),
        ]
    )