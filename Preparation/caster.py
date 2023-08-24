"""
DESCRIPTION: classes and operations for data casting.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULE IMPORT
import torch as th
from pandas import DataFrame
from numpy import ndarray
from torch import Tensor


# CASTER
class Caster:
    # DATA FRAME TO TENSOR
    @classmethod
    def frame2tensor(cls, frame: DataFrame) -> Tensor:
        array = cls._frame2array(frame)

        return cls.array2tensor(array)

    # DATA FRAME TO ARRAY
    @staticmethod
    def _frame2array(data_frame: DataFrame) -> ndarray:
        return data_frame.to_numpy()

    # NUMPY ARRAY TO TENSOR
    @staticmethod
    def array2tensor(array: ndarray) -> Tensor:
        return th.from_numpy(array).float()

    # TENSOR TO ARRAY
    @classmethod
    def tensor2array(cls, tensor: Tensor) -> ndarray:
        if tensor.is_cuda:
            try:
                return cls._tensor2array_gpu(tensor)
            except RuntimeError:
                return cls._tensor2array_detach_gpu(tensor)
        else:
            try:
                return cls._tensor2array_cpu(tensor)
            except RuntimeError:
                return cls._tensor2array_detach_cpu(tensor)

    @staticmethod
    def _tensor2array_cpu(tensor: Tensor) -> ndarray:
        return tensor.numpy()

    @staticmethod
    def _tensor2array_gpu(tensor: Tensor) -> ndarray:
        return tensor.cpu().numpy()

    @staticmethod
    def _tensor2array_detach_cpu(tensor: Tensor) -> ndarray:
        return tensor.detach().numpy()

    @staticmethod
    def _tensor2array_detach_gpu(tensor: Tensor) -> ndarray:
        return tensor.cpu().detach().numpy()
