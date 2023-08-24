"""
DESCRIPTION: classes and operations for data splitting.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULE IMPORT
from abc import ABC, abstractmethod
from types import GeneratorType
from typing import Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series


# SPLITTER
class Splitter(ABC):

    # INITIALIZATION
    def __init__(self, random_seed=8374) -> None:
        # Inputs checking
        if type(random_seed) is not int:
            raise TypeError

        # Splitting attributes
        self._random_seed = random_seed

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def random_seed(self) -> int:
        return self._random_seed

    # SPLITTING
    @abstractmethod
    def split(self, data_frame: DataFrame) -> Union[Tuple[DataFrame, DataFrame], GeneratorType]:
        raise NotImplementedError('This method needs to be implemented in those classes inheriting from Splitter.')

    # NUMBER OF DATA EXTRACTION
    @staticmethod
    def _get_number_data(data_frame: DataFrame) -> int:
        return data_frame.shape[0]


# HOLDOUT SPLITTER
class HoldoutSplitter(Splitter):
    # INITIALIZATION
    def __init__(self, eval_ratio: float = 0.2) -> None:
        # Parent method call
        super().__init__()

        # Attributes assignation
        self._eval_ratio = eval_ratio

    # SPLITTING
    def split(self, data_frame: DataFrame) -> Tuple[DataFrame, DataFrame]:
        # Masks getting
        mask_train, mask_eval = self._get_split_masks(data_frame)

        # Data splitting
        data_train, data_eval = data_frame[mask_train], data_frame[mask_eval]

        # Output
        return data_train, data_eval

    # TRAINING AND EVALUATION SPLITTING MASKS GENERATION
    def _get_split_masks(self, data_frame: DataFrame) -> tuple:
        # Number of data extraction
        number_data = self._get_number_data(data_frame)

        # Indexes generation
        indexes = np.linspace(start=0, stop=(number_data - 1), num=number_data, dtype=int)
        indexes_full = pd.DataFrame(indexes, columns=['SAMPLE_IDXS'])
        indexes_evaluation = indexes_full.sample(frac=self._eval_ratio, replace=False, random_state=self._random_seed)
        indexes_training = indexes_full.iloc[indexes_full.index.difference(indexes_evaluation.index)]

        # Masks generation
        mask_train = self._get_mask(indexes_training, number_data)
        mask_eval = self._get_mask(indexes_evaluation, number_data)

        # Output
        return mask_train, mask_eval

    # MASK GENERATION
    @staticmethod
    def _get_mask(indexes_series: Union[Series, DataFrame], number_data: int) -> ndarray:
        # Mask generation
        indexes_array = indexes_series['SAMPLE_IDXS'].values
        mask_list = [True if i in indexes_array else False for i in range(0, number_data)]

        # Output
        return pd.Series(mask_list).values


# CROSS-VALIDATION SPLITTER
class CrossvalSplitter(Splitter):
    # INITIALIZATION
    def __init__(self, number_folds: int = 4) -> None:
        # Parent method call
        super().__init__()

        # Inputs checking
        if type(number_folds) is not int:
            raise TypeError
        if not 2 < number_folds < 11:  # Not more than 10 to control computational cost
            raise ValueError

        # Attributes assignation
        self._number_folds = number_folds

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def number_folds(self) -> int:
        return self._number_folds

    # SPLITTING
    def split(self, data_frame: DataFrame) -> GeneratorType:
        # Shuffling
        data_frame_shuffled = self._shuffle(data_frame=data_frame)

        # Indexes generation
        indexes_split_list = self._get_split_indexes(data_frame=data_frame)

        # Splitting and yielding
        # initialization
        fold_array = np.arange(0, self.number_folds)
        # iteration
        for fold in range(0, self.number_folds):
            # Training folds extraction
            training_folds = np.setdiff1d(fold_array, np.array([fold]))

            # Training frame generation
            data_frame_train = self._get_training_frame(
                data_frame_shuffled=data_frame_shuffled, indexes_split_list=indexes_split_list,
                train_folds_array=training_folds)

            # Evaluation frame generation
            data_frame_eval = data_frame_shuffled.iloc[indexes_split_list[fold], :]

            # Output
            yield data_frame_train, data_frame_eval

    # SHUFFLING
    def _shuffle(self, data_frame: DataFrame) -> DataFrame:
        # Output
        return data_frame.sample(frac=1, random_state=self.random_seed)

    # INDEXES EXTRACTION
    def _get_split_indexes(self, data_frame: DataFrame) -> list:
        # Indexes extraction
        number_data = self._get_number_data(data_frame=data_frame)
        indexes_array = np.arange(0, number_data)
        indexes_split_list = np.array_split(indexes_array, indices_or_sections=self.number_folds)

        # Output
        return indexes_split_list

    # TRAINING FRAME EXTRACTION
    @staticmethod
    def _get_training_frame(*, data_frame_shuffled: DataFrame, indexes_split_list: list,
                            train_folds_array: ndarray) -> DataFrame:
        # Training frame
        indexes_train_list = []
        for fold in train_folds_array:
            indexes_train_list.append(indexes_split_list[fold])
        indexes_train_array = np.concatenate(indexes_train_list, axis=0)

        # Output
        return data_frame_shuffled.iloc[indexes_train_array, :]
