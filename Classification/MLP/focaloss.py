"""
DESCRIPTION: focal loss for MLP classifier.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from torch import tensor, pow, sum, Tensor
from torch.cuda import is_available


# FOCAL LOSS
class FocalLoss:
    # CLASS ATTRIBUTES
    _cuda_flag = True if is_available() else False
    _class_weighting = True

    # INITIALIZATION
    def __init__(self, *, gamma: float, class_weights: list = None) -> None:
        # Inputs checking
        if type(gamma) is not float:
            raise TypeError
        if self._class_weighting:
            if type(class_weights) is not list:
                raise TypeError

        # Attributes assignation
        self._gamma = gamma
        if self._class_weighting:
            self._class_weights = tensor(class_weights)
            if self._cuda_flag:
                self._class_weights = self._class_weights.cuda()
            self.calculate_loss = self._calculate_focal_loss_weighted
        else:
            self.calculate_loss = self._calculate_focal_loss

    # LOSS MATRIX CALCULATION
    def _calculate_loss_matrix(self, *, labels: Tensor, scores: Tensor) -> Tensor:
        # Focal matrix calculation
        loss_matrix = -labels * pow(input=(1 - scores), exponent=self._gamma) * scores.log()

        # Output
        return loss_matrix

    # SCALARIZATION
    @staticmethod
    def _scalarize(loss_matrix: Tensor) -> Tensor:
        # Scalarization
        loss_vector = sum(loss_matrix, dim=1)  # sum across classes (columns)
        loss_scalar = loss_vector.mean()  # weighted sum across instances (rows)

        # Output
        return loss_scalar

    # FOCAL LOSS CALCULATION
    def _calculate_focal_loss(self, *, labels: Tensor, scores: Tensor) -> Tensor:
        # Focal matrix calculation
        loss_matrix = self._calculate_loss_matrix(labels=labels, scores=scores)

        # Scalarization
        loss_scalar = self._scalarize(loss_matrix)

        # Output
        return loss_scalar

    # CLASS-WEIGHTED FOCAL LOSS CALCULATION
    def _calculate_focal_loss_weighted(self, labels: Tensor, scores: Tensor) -> Tensor:
        # Focal matrix calculation
        loss_matrix = self._calculate_loss_matrix(labels=labels, scores=scores)

        # Weighting
        loss_matrix_weighted = loss_matrix * self._class_weights  # broadcasting

        # Scalarization
        loss_scalar = self._scalarize(loss_matrix_weighted)

        # Output
        return loss_scalar
