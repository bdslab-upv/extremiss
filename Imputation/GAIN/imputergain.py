"""
DESCRIPTION: GAIN imputer.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from numpy import ndarray
from torch import cat
from torch.utils.data import DataLoader
from Preparation.caster import Caster
from Imputation.GAIN.discgen import GeneratorCustom


# ADVERSARIAL IMPUTER
class AdversarialImputer:

    # INITIALIZATION
    def __init__(self, generator: GeneratorCustom) -> None:
        # Attributes assignation
        self._generator = generator

    # IMPUTATION
    def impute(self, loader: DataLoader) -> ndarray:
        # Iterative imputation
        for batch_index, batch_data in enumerate(loader):
            # Data extraction
            features_noise_batch = batch_data['features_noise']
            features_noise_perturbed_batch = batch_data['features_noise_perturbed']
            missing_mask_batch = batch_data['missing_mask']

            # Discriminator updating
            # feature generation
            _, features_combined_batch = self._generator.forward(features_noise=features_noise_perturbed_batch,
                                                                 missing_mask=missing_mask_batch)

            # Feature combination without perturbation
            features_combined_clean_batch = features_noise_batch * missing_mask_batch + features_combined_batch * (
                        1 - missing_mask_batch)

            # Tensor expansion
            if batch_index > 0:
                features_combined_clean = cat(tensors=[features_combined_clean, features_combined_clean_batch], dim=0)
            else:
                features_combined_clean = features_combined_clean_batch

        # Casting
        features_combined_clean = Caster.tensor2array(features_combined_clean)

        # Output
        return features_combined_clean
