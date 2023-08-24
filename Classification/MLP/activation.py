"""
DESCRIPTION: classes and operations for activation functions.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULE IMPORT
import torch.nn as nn
from torch.nn import Module


# ACTIVATION FUNCTIONS FACTORY
class ActivationFactory:

    # FACTORY METHOD
    @staticmethod
    def get_activation_function(identifier: str) -> Module:
        # ReLU
        if identifier == 'relu':
            return nn.ReLU()

        # Leaky ReLU
        elif identifier == 'leaky_relu':
            return nn.LeakyReLU()

        # Swish
        elif identifier == 'swish':
            return nn.SiLU()

        # GELU
        elif identifier == 'gelu':
            return nn.GELU()

        # Softmax
        elif identifier == 'softmax':
            return nn.Softmax(dim=1)

        # Unrecognized activation function identifier
        else:
            raise ValueError('Unrecognized activation function identifier.')
