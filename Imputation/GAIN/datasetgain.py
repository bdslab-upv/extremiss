"""
DESCRIPTION: Generative Adversarial Imputation Networks (GAIN) dataset.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from math import isnan

from numpy.random import seed as set_random_seed
from pandas import DataFrame
from torch import rand, randn, Tensor
from torch.cuda import is_available
from torch.utils.data import Dataset as BaseDataset

from Preparation.caster import Caster

# SETTINGS
RANDOM_SEED = 112


# DATASET
class DatasetGAIN(BaseDataset):
    # CLASS ATTRIBUTES
    _encoding_value = -1  # Value to encode missings. It won't be actually used, it is just to allow Hadamard product.
    _lower_bound_noise = 0
    _upper_bound_noise = 1  # 0.01 is the value used a default in the GAIN paper
    _hint_threshold = 0.95  # Also known as hint ratio or hint probability. Original 0.9 in the paper.
    _use_gpu = is_available()
    _std_perturbation = 0.025  # Standard deviation of the perturbation Gaussian noise.

    # INITIALIZATION
    def __init__(self, data: DataFrame) -> None:
        # Number of data extraction
        number_data = data.shape[0]

        # Number of features extraction
        number_features = data.shape[1]

        # Missing mask generation
        missing_mask_data = self._generate_missing_mask(data)

        # Missing encoding
        data_encoded = self._encode_missings(data)

        # Casting to tensor
        features = Caster.frame2tensor(data_encoded)
        missing_mask = Caster.frame2tensor(missing_mask_data)

        # Random seed setting
        set_random_seed(RANDOM_SEED)

        # Attributes assignation
        self._number_data = number_data
        self._number_features = number_features
        self._features = features
        self._missing_mask = missing_mask

    # MISSING MASK GENERATION
    @staticmethod
    def _generate_missing_mask(data: DataFrame) -> DataFrame:
        # Generation
        missing_mask = data.applymap(lambda x: 0 if isnan(x) else 1)

        # Output
        return missing_mask

    # MISSING ENCODING
    @classmethod
    def _encode_missings(cls, data: DataFrame) -> DataFrame:
        # Generation
        data_encoded = data.applymap(lambda x: cls._encoding_value if isnan(x) else x)

        # Output
        return data_encoded

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    # Features
    @property
    def features(self) -> Tensor:
        return self._features

    # Missing mask
    @property
    def missing_mask(self) -> Tensor:
        return self._missing_mask

    # ITEM EXTRACTION
    def __getitem__(self, index) -> dict:
        # Features and missing mask slicing
        features_batch = self._features[index, :]
        missing_mask_batch = self._missing_mask[index, :]

        # Noise generation
        noise_batch = self._generate_noise()

        # Features and noise combination
        features_noise_batch = features_batch * missing_mask_batch + noise_batch * (1 - missing_mask_batch)

        # Features and noise perturbation
        features_noise_perturbed_batch = self._perturb_features(features_noise_batch)

        # Hint generation
        hint_batch = self._generate_hint(missing_mask_batch)

        # Arrangement
        data_batch = {'features_noise': features_noise_batch,
                      'features_noise_perturbed': features_noise_perturbed_batch,
                      'missing_mask': missing_mask_batch,
                      'hint': hint_batch}

        # Cuda allocation
        if self._use_gpu:
            data_batch_ = {identifier: value.cuda() for identifier, value in data_batch.items()}
            data_batch = data_batch_

        # Output
        return data_batch

    # NOISE GENERATION
    def _generate_noise(self) -> Tensor:
        # Random uniform matrix generation [0, 1)
        random_matrix_uniform_0_1 = rand(size=(self._number_features,))

        # Bounds adjusting
        low = self._lower_bound_noise
        up = self._upper_bound_noise
        random_matrix_uniform_low_up = (low - up) * random_matrix_uniform_0_1 + up

        # Output
        return random_matrix_uniform_low_up

    # PERTURBATION GENERATION
    def _perturb_features(self, features_noise: Tensor) -> Tensor:
        # Standardized normal data generation
        random_matrix = randn(size=(self._number_features,))

        # Features perturbation
        features_noise_perturbed = features_noise + self._std_perturbation * random_matrix

        # Output
        return features_noise_perturbed

    # HINT GENERATION
    def _generate_hint(self, missing_mask: Tensor) -> Tensor:
        # Mask matrix generation
        random_matrix = rand(size=(self._number_features,))
        boolean_matrix = random_matrix > self._hint_threshold
        mask_matrix = 1. * boolean_matrix

        # Hint matrix
        hint_matrix = mask_matrix * missing_mask

        # Output
        return hint_matrix

    # NUMBER OF DATA EXTRACTION
    def __len__(self):
        return self._number_data
