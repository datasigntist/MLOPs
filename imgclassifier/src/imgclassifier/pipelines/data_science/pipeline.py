from kedro.pipeline import node, Pipeline
from .nodes import (
    train_Neural_Network,
    predict_neural_network_train,
    predict_neural_network_test
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_Neural_Network,
                inputs=["parameters","flattened_and_Scaled_Data"],
                outputs="parameters_learned",
                name="training_Neural_Network",
            ),
            node(
                func=predict_neural_network_train,
                inputs=["flattened_and_Scaled_Data","parameters_learned"],
                outputs="probability_outcomes_train",
                name="predict_neural_network_train",
            ),
            node(
                func=predict_neural_network_test,
                inputs=["flattened_and_Scaled_Data","parameters_learned"],
                outputs="probability_outcomes_test",
                name="predict_neural_network_test",
            ),
        ]
    )