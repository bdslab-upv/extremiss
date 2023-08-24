"""
DESCRIPTION: generator and discriminator classes.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch.nn.functional as F
from torch import tensor, cat, Tensor, matmul, sigmoid
from torch.nn import Module, Parameter, LayerNorm, Dropout


# WEIGHT INITIALIZATION
def xavier_initialization(size) -> np.array:
    # Weight initialization
    input_dimension = size[0]
    xavier_std_deviation = 1. / np.sqrt(input_dimension / 2.)
    initial_weights = np.random.normal(size=size, scale=xavier_std_deviation)

    # Output
    return initial_weights


# DISCRIMINATOR
# Abstraction of the Discriminator
class Discriminator(ABC, Module):
    # INITIALIZATION
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    # FORWARD PROPAGATION
    @abstractmethod
    def forward(self, *, features_combined: Tensor, hint: Tensor) -> Tensor:
        raise NotImplementedError


# Original (from GAIN paper)
class DiscriminatorOriginal(Discriminator):
    # INITIALIZATION
    def __init__(self, *, number_features: int, hidden_dimension_1: int, hidden_dimension_2: int) -> None:
        super().__init__()

        # Initialization
        # weights
        w1 = tensor(xavier_initialization([number_features * 2, hidden_dimension_1]))
        w2 = tensor(xavier_initialization([hidden_dimension_1, hidden_dimension_2]))
        w3 = tensor(xavier_initialization([hidden_dimension_2, number_features]))
        # biases
        b1 = tensor(np.zeros(shape=[hidden_dimension_1]))
        b2 = tensor(np.zeros(shape=[hidden_dimension_2]))
        b3 = tensor(np.zeros(shape=[number_features]))

        # Inclusion as parameters
        # weights
        self._w1 = Parameter(w1)
        self._w2 = Parameter(w2)
        self._w3 = Parameter(w3)
        # biases
        self._b1 = Parameter(b1)
        self._b2 = Parameter(b2)
        self._b3 = Parameter(b3)

    # FORWARD
    def forward(self, *, features_combined: Tensor, hint: Tensor) -> Tensor:
        # Concatenation
        features_imputed_hint = cat(dim=1, tensors=[features_combined, hint])

        # Hidden layers propagation
        hidden_1_discriminator = F.relu(matmul(features_imputed_hint, self._w1) + self._b1)
        hidden_2_discriminator = F.relu(matmul(hidden_1_discriminator, self._w2) + self._b2)

        # Feature generation
        real_imputed_probabilities = sigmoid(matmul(hidden_2_discriminator, self._w3) + self._b3)

        # Output
        return real_imputed_probabilities


# Customized
class DiscriminatorCustom(Discriminator):
    # SETTINGS
    _dropout_probability = 0.1

    # INITIALIZATION
    def __init__(self, *, number_features: int, hidden_dimension_1: int, hidden_dimension_2: int) -> None:
        super().__init__()

        # Initialization
        # weights
        w1 = tensor(xavier_initialization([number_features * 2, hidden_dimension_1]))
        w2 = tensor(xavier_initialization([hidden_dimension_1, hidden_dimension_2]))
        w3 = tensor(xavier_initialization([hidden_dimension_2, number_features]))
        # biases
        b1 = tensor(np.zeros(shape=[hidden_dimension_1]))
        b2 = tensor(np.zeros(shape=[hidden_dimension_2]))
        b3 = tensor(np.zeros(shape=[number_features]))

        # Inclusion as parameters
        # weights
        self._w1 = Parameter(w1)
        self._w2 = Parameter(w2)
        self._w3 = Parameter(w3)
        # biases
        self._b1 = Parameter(b1)
        self._b2 = Parameter(b2)
        self._b3 = Parameter(b3)

        # Layer normalization
        self._layer_norm_1 = LayerNorm(normalized_shape=hidden_dimension_1)
        self._layer_norm_2 = LayerNorm(normalized_shape=hidden_dimension_2)

        # Dropout
        self._dropout_1 = Dropout(self._dropout_probability)
        self._dropout_2 = Dropout(self._dropout_probability)

    # FORWARD
    def forward(self, *, features_combined: Tensor, hint: Tensor) -> Tensor:
        # Concatenation
        features_combined_hint = cat(dim=1, tensors=[features_combined, hint])

        # Hidden layers propagation
        hidden_1_discriminator = self._dropout_1(
            F.relu(self._layer_norm_1(matmul(features_combined_hint, self._w1) + self._b1)))
        hidden_2_discriminator = self._dropout_2(
            F.relu(self._layer_norm_2(matmul(hidden_1_discriminator, self._w2) + self._b2)))

        # Feature generation
        real_imputed_probabilities = sigmoid(matmul(hidden_2_discriminator, self._w3) + self._b3)

        # Output
        return real_imputed_probabilities


# GENERATOR
# Abstraction of the generator
class Generator(ABC, Module):
    # INITIALIZATION
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    # FORWARD PROPAGATION
    @abstractmethod
    def forward(self, *, features_combined: Tensor, missing_mask: Tensor) -> Tensor:
        raise NotImplementedError


# Original (from the GAIN paper)
class GeneratorOriginal(Generator):
    # INITIALIZATION
    def __init__(self, *, number_features: int, hidden_dimension_1: int, hidden_dimension_2: int) -> None:
        super().__init__()

        # Initialization
        # weights
        w1 = tensor(xavier_initialization([number_features * 2, hidden_dimension_1]))
        w2 = tensor(xavier_initialization([hidden_dimension_1, hidden_dimension_2]))
        w3 = tensor(xavier_initialization([hidden_dimension_2, number_features]))
        # biases
        b1 = tensor(np.zeros(shape=[hidden_dimension_1]))
        b2 = tensor(np.zeros(shape=[hidden_dimension_2]))
        b3 = tensor(np.zeros(shape=[number_features]))

        # Inclusion as parameters
        # weights
        self._w1 = Parameter(w1)
        self._w2 = Parameter(w2)
        self._w3 = Parameter(w3)
        # biases
        self._b1 = Parameter(b1)
        self._b2 = Parameter(b2)
        self._b3 = Parameter(b3)

    # FORWARD
    def forward(self, *, features_noise: Tensor, missing_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Concatenation
        features_noise_mask = cat(dim=1, tensors=[features_noise, missing_mask])

        # Hidden layers propagation
        hidden_1_generator = F.relu(matmul(features_noise_mask, self._w1) + self._b1)
        hidden_2_generator = F.relu(matmul(hidden_1_generator, self._w2) + self._b2)

        # Feature generation
        features_generated = sigmoid(matmul(hidden_2_generator, self._w4) + self._b4)

        # Feature combination
        features_combined = features_noise * missing_mask + features_generated * (1 - missing_mask)

        # Output
        return features_generated, features_combined


# Customized
class GeneratorCustom(Generator):
    # SETTINGS
    _dropout_probability = 0.1

    # INITIALIZATION
    def __init__(self, *, number_features: int, hidden_dimension_1: int, hidden_dimension_2: int,
                 hidden_dimension_3: int, hidden_dimension_4: int) -> None:
        super().__init__()

        # Initialization
        # weights
        w1 = tensor(xavier_initialization([number_features * 2, hidden_dimension_1]))
        w2 = tensor(xavier_initialization([hidden_dimension_1, hidden_dimension_2]))
        w3 = tensor(xavier_initialization([hidden_dimension_2, hidden_dimension_3]))
        w4 = tensor(xavier_initialization([hidden_dimension_3, hidden_dimension_4]))
        w5 = tensor(xavier_initialization([hidden_dimension_4, number_features]))
        # biases
        b1 = tensor(np.zeros(shape=[hidden_dimension_1]))
        b2 = tensor(np.zeros(shape=[hidden_dimension_2]))
        b3 = tensor(np.zeros(shape=[hidden_dimension_3]))
        b4 = tensor(np.zeros(shape=[hidden_dimension_4]))
        b5 = tensor(np.zeros(shape=[number_features]))

        # Inclusion as parameters
        # weights
        self._w1 = Parameter(w1)
        self._w2 = Parameter(w2)
        self._w3 = Parameter(w3)
        self._w4 = Parameter(w4)
        self._w5 = Parameter(w5)
        # biases
        self._b1 = Parameter(b1)
        self._b2 = Parameter(b2)
        self._b3 = Parameter(b3)
        self._b4 = Parameter(b4)
        self._b5 = Parameter(b5)

        # Layer normalization
        self._layer_norm_1 = LayerNorm(normalized_shape=hidden_dimension_1)
        self._layer_norm_2 = LayerNorm(normalized_shape=hidden_dimension_2)
        self._layer_norm_3 = LayerNorm(normalized_shape=hidden_dimension_3)
        self._layer_norm_4 = LayerNorm(normalized_shape=hidden_dimension_4)

        # Dropout
        self._dropout_1 = Dropout(self._dropout_probability)
        self._dropout_2 = Dropout(self._dropout_probability)
        self._dropout_3 = Dropout(self._dropout_probability)
        self._dropout_4 = Dropout(self._dropout_probability)

    # FORWARD
    def forward(self, *, features_noise: Tensor, missing_mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Concatenation
        features_noise_mask = cat(dim=1, tensors=[features_noise, missing_mask])

        # Hidden layers propagation
        hidden_1_generator = self._dropout_1(
            F.relu(self._layer_norm_1(matmul(features_noise_mask, self._w1) + self._b1)))
        hidden_2_generator = self._dropout_2(
            F.relu(self._layer_norm_2(matmul(hidden_1_generator, self._w2) + self._b2)))
        hidden_3_generator = self._dropout_3(
            F.relu(self._layer_norm_3(matmul(hidden_2_generator, self._w3) + self._b3)))
        hidden_4_generator = self._dropout_4(
            F.relu(self._layer_norm_4(matmul(hidden_3_generator, self._w4) + self._b4)))

        # Feature generation
        features_generated = F.relu(
            matmul(hidden_4_generator, self._w5) + self._b5)  # In the original implementation they use a sigmoid

        # Feature combination
        features_combined = features_noise * missing_mask + features_generated * (1 - missing_mask)

        # Output
        return features_generated, features_combined
