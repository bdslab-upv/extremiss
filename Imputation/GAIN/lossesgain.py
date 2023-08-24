"""
DESCRIPTION: Generative Adversarial Imputation Networks (GAIN) losses.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from torch import Tensor, mean, log

# SETTINGS
alpha = 10
offset = 1e-8


# DISCRIMINATOR LOSS
def loss_discriminator(*, real_imputed_probabilities: Tensor, missing_mask: Tensor):
    # Loss calculation
    loss_discriminator_ = -mean(missing_mask * log(real_imputed_probabilities + offset) + (1 - missing_mask) * log(
        1. - real_imputed_probabilities + offset))

    # Output
    return loss_discriminator_


# GENERATOR LOSS
def loss_generator(*, features_noise_perturbed: Tensor, missing_mask, features_generated: Tensor,
                   real_imputed_probabilities: Tensor):
    # Loss calculation
    # classification loss
    loss_generator_classification = -mean((1 - missing_mask) * log(real_imputed_probabilities + offset))
    # reconstruction loss
    loss_generator_reconstruction = mean(
        (features_noise_perturbed * missing_mask - features_generated * missing_mask) ** 2) / mean(missing_mask)
    # combination
    loss_generator_ = loss_generator_classification + alpha * loss_generator_reconstruction

    # Output
    return loss_generator_, loss_generator_classification, loss_generator_reconstruction
