"""
DESCRIPTION: script to define imputer hyperparameters.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT


# TRANSLATION AND ENCODING
HYPERPARAMETERS_TRANSENC = {'translation_scalar': 0.2, 'encoding_value': 0}

# K-NEAREST NEIGHBORS
HYPERPARAMETERS_KNNIMP = {'number_neighbors': 49}

# GENERATIVE ADVERSARIAL IMPUTATION NETWORKS
HYPERPARAMETERS_GAIN = {'batch_size_train': 64, 'batch_size_prediction': 512, 'number_epochs': 30,
                        'learning_rate_generator': 0.0001, 'learning_rate_discriminator': 0.00001,
                        'hidden_dimension_1': 512, 'hidden_dimension_2': 256, 'hidden_dimension_3': 128,
                        'hidden_dimension_4': 64}
