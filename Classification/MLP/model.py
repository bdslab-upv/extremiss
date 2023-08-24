"""
DESCRIPTION: script to define multi-layer perceptron model.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from torch import Tensor
from torch.nn import Module

from Classification.MLP.builder import Builder
from Classification.MLP.forwarder import DenseForwarder


# MULTI-LAYER PERCEPTRON MODEL DEFINITION
class MultiLayerPerceptron(Module):
    # CLASS ATTRIBUTES
    _activation_function_dense = 'relu'
    _activation_function_output = 'softmax'
    _normalize = True
    _normalizer = 'layer_normalization'
    _initialize_weights = True

    # INITIALIZATION
    def __init__(self, *, number_features: int, number_classes: int, hidden_sizes: list,
                 dropout_ratio: float = 0.1) -> None:
        super().__init__()

        # Inputs checking
        if type(number_features) is not int:
            raise TypeError
        if type(number_classes) is not int:
            raise TypeError
        if type(hidden_sizes) is not list:
            raise TypeError
        if type(dropout_ratio) is not float:
            raise TypeError

        # Metadata attributes definition
        self._number_features = number_features
        self._number_classes = number_classes

        # Modules building
        self._dense_module = Builder.build_dense_module_output(
            hidden_sizes=[number_features] + hidden_sizes + [number_classes],
            initialize_weights=self._initialize_weights, normalize=self._normalize, normalizer=self._normalizer,
            activation_function_dense=self._activation_function_dense,
            activation_function_output=self._activation_function_output, dropout_ratio=dropout_ratio)

    # FORWARD
    def forward(self, features: Tensor) -> Tensor:
        # Dense module propagation
        output = DenseForwarder.forward_dense_module(tensor=features, dense_module=self._dense_module)

        # Output
        return output
