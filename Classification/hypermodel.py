"""
DESCRIPTION: script to define classifier hyperparameters.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT


# K-NEAREST NEIGHBORS
HYPERPARAMETERS_KNN = {('robust', 'missing_mask'): {'number_neighbors': 200},
                       ('robust', 'mean'): {'number_neighbors': 300},
                       ('robust', 'bayesian_regression'): {'number_neighbors': 300},
                       ('robust', 'knn'): {'number_neighbors': 300},
                       ('minmax', 'translation_encoding'): {'number_neighbors': 300},
                       ('minmax', 'gain'): {'number_neighbors': 300}}

# RANDOM FOREST
HYPERPARAMETERS_RANDOM_FOREST = {('robust', 'missing_mask'): {'number_trees': 500, 'maximum_depth': 9},
                                 ('robust', 'mean'): {'number_trees': 400, 'maximum_depth': 11},
                                 ('robust', 'bayesian_regression'): {'number_trees': 500, 'maximum_depth': 11},
                                 ('robust', 'knn'): {'number_trees': 500, 'maximum_depth': 11},
                                 ('minmax', 'translation_encoding'): {'number_trees': 400, 'maximum_depth': 11},
                                 ('minmax', 'gain'): {'number_trees': 400, 'maximum_depth': 11}}

# GRADIENT BOOSTING
HYPERPARAMETERS_GRADIENT_BOOSTING = {('robust', 'missing_mask'): {'number_trees': 100, 'maximum_depth': 3},
                                     ('robust', 'mean'): {'number_trees': 200, 'maximum_depth': 3},
                                     ('robust', 'bayesian_regression'): {'number_trees': 100, 'maximum_depth': 3},
                                     ('robust', 'knn'): {'number_trees': 200, 'maximum_depth': 3},
                                     ('minmax', 'translation_encoding'): {'number_trees': 200, 'maximum_depth': 3},
                                     ('minmax', 'gain'): {'number_trees': 100, 'maximum_depth': 3}}

# MULTI-LAYER PERCEPTRON
HYPERPARAMETERS_MLP = {
    ('robust', 'missing_mask'):
        {'dropout_ratio': 0.25, 'hidden_sizes': [256, 256, 128, 128, 64], 'gamma': 1.5,
         'class_weights': [0.25, 0.75], 'learning_rate': 0.0001, 'weight_decay': 0.01, 'batch_size': 64,
         'maximum_epochs': 35},
    ('robust', 'mean'):
        {'dropout_ratio': 0.25, 'hidden_sizes': [128, 64, 32], 'gamma': 1.5,
         'class_weights': [0.25, 0.75], 'learning_rate': 0.0001, 'weight_decay': 0.01, 'batch_size': 32,
         'maximum_epochs': 50},
    ('robust', 'bayesian_regression'):
        {'dropout_ratio': 0.25, 'hidden_sizes': [128, 64], 'gamma': 1.5, 'class_weights': [0.3, 0.7],
         'learning_rate': 0.0001, 'weight_decay': 0.01, 'batch_size': 32, 'maximum_epochs': 50},
    ('robust', 'knn'):
        {'dropout_ratio': 0.25, 'hidden_sizes': [256, 256, 128, 128, 64], 'gamma': 1.5,
         'class_weights': [0.3, 0.7], 'learning_rate': 0.0001, 'weight_decay': 0.00001, 'batch_size': 64,
         'maximum_epochs': 50},
    ('minmax', 'translation_encoding'):
        {'dropout_ratio': 0.15, 'hidden_sizes': [128, 64], 'gamma': 1.5, 'class_weights': [0.3, 0.7],
         'learning_rate': 0.0001, 'weight_decay': 0.001, 'batch_size': 32, 'maximum_epochs': 50},
    ('minmax', 'gain'):
        {'dropout_ratio': 0.15, 'hidden_sizes': [128, 128, 64, 64], 'gamma': 2.5,
         'class_weights': [0.3, 0.7], 'learning_rate': 0.0001, 'weight_decay': 0.00001, 'batch_size': 32,
         'maximum_epochs': 50}}
