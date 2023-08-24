"""
DESCRIPTION: datasets generation for TabTransformer model for COVID-19 mortality prediction.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from typing import Tuple

from numpy import ndarray
from torch import Tensor, unsqueeze, zeros
from torch.utils.data import Dataset as BaseDataset

from Preparation.caster import Caster


# MLP DATASET
class MLPDataset(BaseDataset):
    # INITIALIZATION
    def __init__(self, number_data: int, features: Tensor, label_tensor: Tensor) -> None:
        # Dimension expansion
        label_tensor = unsqueeze(input=label_tensor, dim=1)

        # One-hot encoding of the binary label
        label_onehot = zeros(size=(number_data, 2))
        for i in range(number_data):
            value = label_tensor[i, 0]
            label_onehot[i, 0] = 0 if value > 0 else 1
            label_onehot[i, 1] = value

        # Attributes assignation
        self._number_data = number_data
        self._features = features
        self._label_tensor = label_onehot

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._label_tensor

    # ITEM EXTRACTION
    def __getitem__(self, indexes):
        # Features and label extraction
        features_sliced = self._features[indexes, :]
        label_sliced = self._label_tensor[indexes, :]

        # Arrangement
        data_batch = {'features': features_sliced, 'label': label_sliced}

        # Output
        return data_batch

    # NUMBER OF DATA EXTRACTION
    def __len__(self):
        return self._number_data


# GENERATE DATASETS
# Training and test
def generate_datasets_traintest(*, X_train: ndarray, y_train: ndarray, X_test: ndarray, y_test: ndarray) -> Tuple[
    MLPDataset, MLPDataset]:
    # Number data extraction
    number_data_train = X_train.shape[0]
    number_data_test = X_test.shape[0]

    # Tensor casting
    # training
    X_train = Caster.array2tensor(X_train)
    y_train = Caster.array2tensor(y_train)
    # test
    X_test = Caster.array2tensor(X_test)
    y_test = Caster.array2tensor(y_test)

    # Dataset generation
    # training
    dataset_train = MLPDataset(number_data=number_data_train, features=X_train, label_tensor=y_train)
    # test
    dataset_test = MLPDataset(number_data=number_data_test, features=X_test, label_tensor=y_test)

    # Output
    return dataset_train, dataset_test
