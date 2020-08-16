from kedro.pipeline import node, Pipeline
from .nodes import (
    prepocess_trainingdata,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=prepocess_trainingdata,
                inputs="housingdata_train",
                outputs="preprocessed_housingdata_train",
                name="preprocessing_housingdata_train",
            ),
        ]
    )