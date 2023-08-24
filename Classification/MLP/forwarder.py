"""
DESCRIPTION: classes and operations to do forward propagation in pytorch models.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""


# MODULE IMPORT


# DENSE FORWARDER
class DenseForwarder:
    @staticmethod
    def forward_dense_module(tensor, dense_module):
        for module in dense_module:
            tensor = module(tensor)

        return tensor

