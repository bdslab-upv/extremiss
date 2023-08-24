"""
DESCRIPTION: axuliar functions for classification.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from typing import Union

from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from torch import no_grad, cat
from torch.cuda import is_available
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader

from Classification.MLP.datasets import generate_datasets_traintest
from Classification.MLP.focaloss import FocalLoss
from Classification.MLP.model import MultiLayerPerceptron
from Classification.MLP.trainmodel import train_model as train_deep_model
from Classification.hypermodel import HYPERPARAMETERS_KNN, HYPERPARAMETERS_RANDOM_FOREST, \
    HYPERPARAMETERS_GRADIENT_BOOSTING, HYPERPARAMETERS_MLP
from Preparation.caster import Caster

# SETTINGS
RANDOM_SEED = 112
BATCH_SIZE_EVAL = 256


# CLASSIFIER INITIALIZATION
def initialize_model(*, model_identifier: str, scaling_method: str, imputation_method: str,
                     feature_identifiers: list = None, number_classes: int = None) -> Union[
    BaseEstimator, Module]:
    # Classifier initialization
    # k-nearest neighbors
    if model_identifier == 'k_nearest_neighbors':
        # Hyperparameters extraction
        hyperpars = HYPERPARAMETERS_KNN[(scaling_method, imputation_method)]
        number_neighbors = hyperpars['number_neighbors']

        # Classifier initialization
        model = KNeighborsClassifier(n_neighbors=number_neighbors)

    # logistic regression
    elif model_identifier == 'logistic_regression':
        model = LogisticRegression(random_state=RANDOM_SEED)

    # random forest
    elif model_identifier == 'random_forest':
        # Hyperparameters extraction
        hyperpars = HYPERPARAMETERS_RANDOM_FOREST[(scaling_method, imputation_method)]
        number_trees = hyperpars['number_trees']
        maximum_depth = hyperpars['maximum_depth']

        # Classifier initialization
        model = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=number_trees,
                                       max_depth=maximum_depth)

    # gradient boosting
    elif model_identifier == 'gradient_boosting':
        # Hyperparameters extraction
        hyperpars = HYPERPARAMETERS_GRADIENT_BOOSTING[(scaling_method, imputation_method)]
        number_trees = hyperpars['number_trees']
        maximum_depth = hyperpars['maximum_depth']

        # Classifier initialization
        model = GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=number_trees,
                                           max_depth=maximum_depth)

    # multi-layer perceptron
    elif model_identifier == 'multilayer_perceptron':
        # Inputs checking
        if feature_identifiers is None or number_classes is None:
            raise ValueError('Feature identifiers or/and number classes must be specified.')

        # Hyperparameters extraction
        hyperpars = HYPERPARAMETERS_MLP[(scaling_method, imputation_method)]
        hidden_sizes = hyperpars['hidden_sizes']
        dropout_ratio = hyperpars['dropout_ratio']
        number_features = len(feature_identifiers)

        # Classifier initialization
        model = MultiLayerPerceptron(number_features=number_features, number_classes=number_classes,
                                     hidden_sizes=hidden_sizes, dropout_ratio=dropout_ratio)

    # unrecognized classifier identifier
    else:
        raise ValueError('Unrecognized classifier identifier.')

    # Output
    return model


# CLASSIFIER TRAINING AND PREDICTIONS CALCULATION
def train_model_calculate_predictions(*, model: Union[BaseEstimator, Module], data_train: DataFrame,
                                      data_eval: DataFrame, feature_identifiers: list, label_identifier: str,
                                      scaling_method: str, imputation_method: str) -> dict:
    # Memory allocation
    labels_predictions = dict()

    # Data extraction
    # train
    #   features
    X_train = data_train[feature_identifiers].values
    #   label
    y_train = data_train[[label_identifier]].values
    # test
    #   features
    X_test = data_eval[feature_identifiers].values
    #   label
    y_test = data_eval[[label_identifier]].values

    # Classifier training and predictions calculation
    # sklearn models
    if isinstance(model, BaseEstimator):
        # Training
        model.fit(X=X_train, y=y_train)

        # Prediction
        # test
        yhat_test = model.predict_proba(X=X_test)

    # PyTorch model
    elif isinstance(model, Module):
        # Hyperparameters extraction
        hyperpars = HYPERPARAMETERS_MLP[(scaling_method, imputation_method)]
        batch_size = hyperpars['batch_size']
        gamma = hyperpars['gamma']
        class_weights = hyperpars['class_weights']
        learning_rate = hyperpars['learning_rate']
        weight_decay = hyperpars['weight_decay']
        maximum_epochs = hyperpars['maximum_epochs']

        # Data preparation
        # datasets generation
        dataset_train, dataset_test = generate_datasets_traintest(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        # data loaders generation
        loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
        loader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE_EVAL, shuffle=False, drop_last=False)

        # Optimizer initialization
        optimizer = AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Loss function definition
        lossfun = FocalLoss(gamma=gamma, class_weights=class_weights)

        # Model training
        model, _, _ = train_deep_model(
            model=model, optimizer=optimizer, loss_function=lossfun, loader_train=loader_train, loader_val=loader_test,
            maximum_epochs=maximum_epochs)

        # Prediction
        output_map = _predict_mlp(model=model, loader_train=loader_train, loader_test=loader_test)

        # Casting
        # labels
        y_test = Caster.tensor2array(output_map['labels_test'])
        # predictions
        yhat_test = Caster.tensor2array(output_map['probs_test'])

    # unrecognized model
    else:
        raise ValueError('Unrecognized model.')

    # Labels and predictions arrangement
    labels_predictions[('test', 'label')] = y_test
    labels_predictions[('test', 'prediction')] = yhat_test

    # Output
    return labels_predictions


# Prediction
def _predict_mlp(*, model: Module, loader_train: DataLoader, loader_test: DataLoader) -> dict:
    # Predictions extraction
    # model preparation
    model.eval()
    if is_available():
        model.cuda()
    # forward propagation
    #   train
    with no_grad():
        for batch_index_train, data_batch_train in enumerate(loader_train):
            features_batch_train = data_batch_train['features']
            labels_batch_train = data_batch_train['label']

            if is_available():
                features_batch_train = features_batch_train.cuda()

            probs_batch_train = model.forward(features=features_batch_train)

            if batch_index_train > 0:
                labs_train = cat((labs_train, labels_batch_train), dim=0)
                probs_train = cat((probs_train, probs_batch_train), dim=0)
            else:
                labs_train = labels_batch_train
                probs_train = probs_batch_train
        #   test
        for batch_index_test, data_batch_test in enumerate(loader_test):
            features_batch_test = data_batch_test['features']
            labels_batch_test = data_batch_test['label']

            if is_available():
                features_batch_test = features_batch_test.cuda()

            probs_batch_test = model.forward(features=features_batch_test)

            if batch_index_test > 0:
                labs_test = cat((labs_test, labels_batch_test), dim=0)
                probs_test = cat((probs_test, probs_batch_test), dim=0)
            else:
                labs_test = labels_batch_test
                probs_test = probs_batch_test

    # Arrangement
    output_map = {'labels_train': labs_train, 'probs_train': probs_train, 'labels_test': labs_test,
                  'probs_test': probs_test}

    # Output
    return output_map
