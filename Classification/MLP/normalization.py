"""
DESCRIPTION: classes and operations for normalization of deep models.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULE IMPORT
import torch.nn as nn
from torch.nn import Module


# NORMALIZATION FACTORY
class NormalizationFactory:

    # FACTORY METHOD
    @staticmethod
    def get_normalizer(identifier: str, dimension: int) -> Module:
        # Batch normalization
        if identifier == 'batch_normalization':
            normalizer = nn.BatchNorm1d(num_features=dimension)

        # Layer normalization
        elif identifier == 'layer_normalization':
            normalizer = nn.LayerNorm(normalized_shape=dimension)

        # Unrecognized normalizer
        else:
            raise ValueError('Unrecognized normalizer identifier.')

        # Output
        return normalizer
