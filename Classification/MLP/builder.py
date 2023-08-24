"""
DESCRIPTION: classes and operations for building deep learning blocks.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULE IMPORT
from typing import Union

import torch as th
import torch.nn as nn
from torch.nn import Module, ModuleList

from Classification.MLP.activation import ActivationFactory
from Classification.MLP.normalization import NormalizationFactory


# PYTORCH MODULES AND BLOCKS BUILDER
class Builder:

    # DENSE MODULE
    # Inner dense module
    @classmethod
    def build_dense_module_inner(cls, *, hidden_sizes: Union[list, tuple], initialize_weights: bool, normalize: bool,
                                 normalizer: str, activation_function_dense: str, dropout_ratio: float) -> ModuleList:
        # Memory allocation
        dense_module = []

        # Building
        for i in range(len(hidden_sizes) - 1):
            dense_module.append(cls._build_dense_block_inner(
                input_dimension=hidden_sizes[i], output_dimension=hidden_sizes[i + 1],
                initialize_weights=initialize_weights, normalize=normalize, normalizer=normalizer,
                activation_function_dense=activation_function_dense, dropout_ratio=dropout_ratio))

        # Conversion to module list
        dense_module = nn.ModuleList(dense_module)

        # Output
        return dense_module

    # Output dense module
    @classmethod
    def build_dense_module_output(cls, *, hidden_sizes: Union[list, tuple], initialize_weights: bool, normalize: bool,
                                  normalizer: str, activation_function_dense: str,
                                  activation_function_output: str, dropout_ratio: float) -> ModuleList:
        # Memory allocation
        dense_module = []

        # Building
        for i in range(len(hidden_sizes) - 1):
            if not i == (len(hidden_sizes) - 2):  # not the last layer
                dense_module.append(
                    cls._build_dense_block_inner(
                        input_dimension=hidden_sizes[i], output_dimension=hidden_sizes[i + 1],
                        initialize_weights=initialize_weights, normalize=normalize, normalizer=normalizer,
                        activation_function_dense=activation_function_dense, dropout_ratio=dropout_ratio))

            else:  # last layer (output layer)
                dense_module.append(
                    cls._build_dense_block_output(
                        input_dimension=hidden_sizes[i], output_dimension=hidden_sizes[i + 1],
                        initialize_weights=initialize_weights, activation_function_output=activation_function_output))

        # Conversion to module list
        dense_module = nn.ModuleList(dense_module)

        # Output
        return dense_module

    # DENSE BLOCK
    # Inner dense block
    @staticmethod
    def _build_dense_block_inner(*, input_dimension: int, output_dimension: int, initialize_weights: bool,
                                 normalize: bool, normalizer: str, activation_function_dense: str,
                                 dropout_ratio: float):
        # Dense block initialization
        dense_block = nn.Sequential()

        # Linear transformation
        linear = nn.Linear(input_dimension, output_dimension, bias=True)  # bias=True
        dense_block.add_module('linear', linear)

        # Weight initialization
        if initialize_weights:
            th.nn.init.kaiming_uniform_(linear.weight)

        # Normalization
        if normalize:
            normalizer_module = NormalizationFactory.get_normalizer(identifier=normalizer, dimension=output_dimension)
            dense_block.add_module('normalization', normalizer_module)

        # Non-linear transformation (activation function)
        activation_function = ActivationFactory.get_activation_function(activation_function_dense)
        dense_block.add_module('activation_function', activation_function)

        # Dropout
        dense_block.add_module('dropout', nn.Dropout(dropout_ratio))

        # Output
        return dense_block

    # Output dense block
    @staticmethod
    def _build_dense_block_output(*, input_dimension: int, output_dimension: int, initialize_weights: bool,
                                  activation_function_output: str):
        # Dense block initialization
        dense_block = nn.Sequential()

        # Linear transformation
        linear = nn.Linear(input_dimension, output_dimension)
        dense_block.add_module('linear', linear)

        # Weight initialization
        if initialize_weights:
            th.nn.init.xavier_uniform_(linear.weight)

        # Non-linear transformation (activation function)
        activation_function = ActivationFactory.get_activation_function(activation_function_output)
        dense_block.add_module('activation_function', activation_function)

        # Output
        return dense_block

    # EMBEDDING LAYER
    @staticmethod
    def build_embedding_layer(*, number_features: int, embedding_dimension: int, padding_index: int) -> Module:

        """
        Used to map integers to a dense vector representation through a lookup table.
        """

        # Embedding layer definition
        embedding_layer = nn.Embedding(num_embeddings=number_features, embedding_dim=embedding_dimension,
                                       padding_idx=padding_index)

        # Output
        return embedding_layer
