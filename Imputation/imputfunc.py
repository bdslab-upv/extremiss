"""
DESCRIPTION: functions to impute data.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from math import isnan
from typing import Tuple

from numpy import expand_dims
from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from torch.utils.data import DataLoader

from Imputation.hyperimput import HYPERPARAMETERS_TRANSENC, HYPERPARAMETERS_KNNIMP, HYPERPARAMETERS_GAIN
from Imputation.GAIN.datasetgain import DatasetGAIN
from Imputation.GAIN.discgen import DiscriminatorCustom, GeneratorCustom
from Imputation.GAIN.imputergain import AdversarialImputer
from Imputation.GAIN.traineradvimp import AdversarialImputerTrainer

# SETTINGS
RANDOM_SEED = 112


# MISSING IMPUTATION
def impute_missings(*, data_split: dict, imputation_method: str, scaling_method: str, columns2impute: list,
                    feature_identifiers: list = None) -> dict:
    # Inputs consistency checking
    if scaling_method == 'robust':
        assert imputation_method in ('missing_mask', 'mean', 'bayesian_regression', 'knn')
    elif scaling_method == 'minmax':
        assert imputation_method in ('translation_encoding', 'gain')
    else:
        raise ValueError('Unrecognized scaling approach.')

    # Training and test imputation
    # data extraction
    data_train = data_split['train_test']['train']
    data_test = data_split['train_test']['test']
    # imputation
    data_train, data_test = _impute_features(data_train=data_train, data_eval=data_test,
                                             columns_to_impute=columns2impute, imputation_method=imputation_method,
                                             feature_identifiers=feature_identifiers)
    # arrangement
    data_split['train_test']['train'] = data_train
    data_split['train_test']['test'] = data_test

    # Cross-validation folds imputation
    kfolds_map = data_split['kfolds'].copy()

    for fold_idf in kfolds_map.keys():
        data_puretrain = kfolds_map[fold_idf]['puretrain']
        data_validation = kfolds_map[fold_idf]['validation']

        data_puretrain, data_validation = _impute_features(data_train=data_puretrain, data_eval=data_validation,
                                                           columns_to_impute=columns2impute,
                                                           imputation_method=imputation_method,
                                                           feature_identifiers=feature_identifiers)

        data_split['kfolds'][fold_idf]['puretrain'] = data_puretrain
        data_split['kfolds'][fold_idf]['validation'] = data_validation

    # Output
    return data_split


# Data frame level
def _impute_features(*, data_train: DataFrame, data_eval: DataFrame, columns_to_impute: list,
                     imputation_method: str, feature_identifiers: list = None) -> Tuple[DataFrame, DataFrame]:
    # Univariate methods
    if imputation_method in ('missing_mask', 'mean', 'translation_encoding'):

        # Imputation over features
        for column in columns_to_impute:
            # Values extraction
            series_train = data_train[column]
            series_eval = data_eval[column]

            # Imputation
            # missing mask
            if imputation_method == 'missing_mask':
                series_train_imputed = series_train.apply(lambda x: 0 if isnan(x) else 1)
                series_eval_imputed = series_eval.apply(lambda x: 0 if isnan(x) else 1)

            # mean
            elif imputation_method == 'mean':
                series_train_expanded = expand_dims(series_train, axis=1)
                series_eval_expanded = expand_dims(series_eval, axis=1)

                imputer = SimpleImputer(strategy='mean')

                imputer.fit(series_train_expanded)
                series_train_imputed = imputer.transform(series_train_expanded)
                series_eval_imputed = imputer.transform(series_eval_expanded)

            # translation and encoding
            elif imputation_method == 'translation_encoding':
                # Hyperparameters definition
                translation_scalar = HYPERPARAMETERS_TRANSENC['translation_scalar']
                encoding_value = HYPERPARAMETERS_TRANSENC['encoding_value']

                # Imputation
                # training
                series_train_imputed = series_train.apply(
                    _translate_encode, translation_scalar=translation_scalar, encoding_value=encoding_value)
                # evaluation
                series_eval_imputed = series_eval.apply(
                    _translate_encode, translation_scalar=translation_scalar, encoding_value=encoding_value)

            # unrecognized imputation method
            else:
                raise ValueError('Unrecognized univariate imputation method.')

            # Arrangement
            data_train[column] = series_train_imputed
            data_eval[column] = series_eval_imputed

    # Multivariate methods
    elif imputation_method in ('bayesian_regression', 'knn', 'gain'):
        # Inputs checking
        if feature_identifiers is None:
            raise ValueError('Feature identifiers must be specified.')

        # Features extraction
        features_train = data_train[feature_identifiers]
        features_eval = data_eval[feature_identifiers]

        # Imputation for sklearn imputers
        if imputation_method in ('bayesian_regression', 'knn'):
            # Imputer definition
            # bayesian ridge regression
            if imputation_method == 'bayesian_regression':
                imputer = IterativeImputer(random_state=RANDOM_SEED)

            # k-nearest neighbors
            elif imputation_method == 'knn':
                # Hyperparameters definition
                number_neighbors = HYPERPARAMETERS_KNNIMP['number_neighbors']

                # Imputer definition
                imputer = KNNImputer(n_neighbors=number_neighbors)

            # unrecognized imputation approach
            else:
                raise ValueError('Unrecognized imputation approach.')

            # Imputation
            imputer.fit(features_train)
            features_train_imputed = imputer.transform(features_train)
            features_eval_imputed = imputer.transform(features_eval)

        # Imputation with adversarial neural networks
        elif imputation_method == 'gain':
            # Hyperparameters definition
            # learning hyperparameters
            batch_size_train = HYPERPARAMETERS_GAIN['batch_size_train']
            learning_rate_discriminator = HYPERPARAMETERS_GAIN['learning_rate_discriminator']
            learning_rate_generator = HYPERPARAMETERS_GAIN['learning_rate_generator']
            number_epochs = HYPERPARAMETERS_GAIN['number_epochs']
            # architecture hyperparameters
            hidden_dimension_1 = HYPERPARAMETERS_GAIN['hidden_dimension_1']
            hidden_dimension_2 = HYPERPARAMETERS_GAIN['hidden_dimension_2']
            hidden_dimension_3 = HYPERPARAMETERS_GAIN['hidden_dimension_3']
            hidden_dimension_4 = HYPERPARAMETERS_GAIN['hidden_dimension_4']
            # additional settings
            batch_size_prediction = HYPERPARAMETERS_GAIN['batch_size_prediction']

            # Metadata extraction
            number_features_ = len(feature_identifiers)

            # Datasets generation
            # training
            dataset_train = DatasetGAIN(features_train)
            # evaluation
            dataset_eval = DatasetGAIN(features_eval)

            # Data loaders generation
            # training
            loader_train2train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True,
                                            drop_last=False)
            loader_train2impute = DataLoader(dataset=dataset_train, batch_size=batch_size_prediction, shuffle=False,
                                             drop_last=False)
            # evaluation
            loader_eval = DataLoader(dataset=dataset_eval, batch_size=batch_size_prediction, shuffle=False,
                                     drop_last=False)

            # Models initialization
            # discriminator
            discriminator = DiscriminatorCustom(number_features=number_features_, hidden_dimension_1=hidden_dimension_3,
                                                hidden_dimension_2=hidden_dimension_4)
            # generator
            generator = GeneratorCustom(number_features=number_features_, hidden_dimension_1=hidden_dimension_1,
                                        hidden_dimension_2=hidden_dimension_2, hidden_dimension_3=hidden_dimension_3,
                                        hidden_dimension_4=hidden_dimension_4)
            # model parameters casting
            discriminator.float()
            generator.float()

            # Training
            # initialization
            trainer_gain = AdversarialImputerTrainer(
                discriminator=discriminator, generator=generator, loader_train=loader_train2train,
                loader_eval=loader_eval,
                learning_rate_discriminator=learning_rate_discriminator,
                learning_rate_generator=learning_rate_generator,
                number_epochs=number_epochs)
            # training
            trainer_gain.train()

            # Imputation
            # imputer initialization
            imputer = AdversarialImputer(trainer_gain.generator)
            # imputation
            features_train_imputed = imputer.impute(loader_train2impute)
            features_eval_imputed = imputer.impute(loader_eval)

        # Unrecognized imputation method
        else:
            raise ValueError('Unrecognized multivariate imputation method.')

        # Arrangement
        data_train[feature_identifiers] = features_train_imputed
        data_eval[feature_identifiers] = features_eval_imputed

    # Unrecognized method
    else:
        raise ValueError('Unrecognized imputation method.')

    # Output
    return data_train, data_eval


# Instance level
def _translate_encode(value: float, translation_scalar: float, encoding_value: float) -> float:
    # Translation and encoding
    if isnan(value):
        value_transformed = encoding_value
    else:
        value_transformed = value + translation_scalar

    # Output
    return value_transformed
