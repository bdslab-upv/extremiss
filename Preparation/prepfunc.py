"""
DESCRIPTION: functions to prepare data.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from math import isnan
from typing import Union, Tuple

from pandas import DataFrame

from Preparation.splitter import HoldoutSplitter, CrossvalSplitter


# DATA SPLITTING
def split_data(data: DataFrame) -> dict:
    # Memory allocation
    data_split = {'train_test': dict(), 'kfolds': dict()}

    # Splitters instantiation
    holdout_splitter = HoldoutSplitter(eval_ratio=0.2)
    crossval_splitter = CrossvalSplitter(number_folds=4)

    # Splitting
    # train and test partition
    data_train, data_test = holdout_splitter.split(data)
    # cross-validation partition
    data_trainval_generator = crossval_splitter.split(data_train)

    # Arrangement
    # training and test
    data_split['train_test']['train'] = data_train
    data_split['train_test']['test'] = data_test
    # cross-validation
    counter = 0
    for puretrain_frame, validation_frame in data_trainval_generator:
        fold_identifier = f'fold_{counter}'

        data_split['kfolds'][fold_identifier] = dict()
        data_split['kfolds'][fold_identifier]['puretrain'] = puretrain_frame
        data_split['kfolds'][fold_identifier]['validation'] = validation_frame

        counter += 1

    # Output
    return data_split


# DATA SCALING
# Data split level
def scale_data(*, data_split: dict, scaling_method: str, columns2scale: list) -> dict:
    # Training and test scaling
    # data extraction
    data_train = data_split['train_test']['train']
    data_test = data_split['train_test']['test']
    # scaling
    data_train, data_test = _scale_features(data_train=data_train, data_eval=data_test, columns_to_scale=columns2scale,
                                            scaling_method=scaling_method)
    # arrangement
    data_split['train_test']['train'] = data_train
    data_split['train_test']['test'] = data_test

    # Cross-validation folds scaling
    kfolds_map = data_split['kfolds'].copy()

    for fold_idf in kfolds_map.keys():
        data_puretrain = kfolds_map[fold_idf]['puretrain']
        data_validation = kfolds_map[fold_idf]['validation']

        data_puretrain, data_validation = _scale_features(data_train=data_puretrain, data_eval=data_validation,
                                                          columns_to_scale=columns2scale, scaling_method=scaling_method)

        data_split['kfolds'][fold_idf]['puretrain'] = data_puretrain
        data_split['kfolds'][fold_idf]['validation'] = data_validation

    # Output
    return data_split


# Data frame level
def _scale_features(*, data_train: DataFrame, data_eval: DataFrame, columns_to_scale: list,
                    scaling_method: str) -> Tuple[DataFrame, DataFrame]:
    # Iterative scaling
    for column in columns_to_scale:
        series_train = data_train[column]
        series_eval = data_eval[column]

        if scaling_method == 'robust':
            # Percentiles extraction
            first_quartile_train = series_train.quantile(q=0.25)
            median_train = series_train.quantile(q=0.5)
            third_quartile_train = series_train.quantile(q=0.75)

            # Scaling
            series_train_scaled = series_train.apply(_scale_value_robust, median=median_train,
                                                     first_quartile=first_quartile_train,
                                                     third_quartile=third_quartile_train)
            series_eval_scaled = series_eval.apply(_scale_value_robust, median=median_train,
                                                   first_quartile=first_quartile_train,
                                                   third_quartile=third_quartile_train)

        elif scaling_method == 'minmax':
            # Percentiles extraction
            lower_quantile_train = series_train.quantile(q=0.025)
            upper_quantile_train = series_train.quantile(q=0.975)

            # Scaling
            series_train_scaled = series_train.apply(_scale_value_minmax, lower_quartile=lower_quantile_train,
                                                     upper_quartile=upper_quantile_train)
            series_eval_scaled = series_eval.apply(_scale_value_minmax, lower_quartile=lower_quantile_train,
                                                   upper_quartile=upper_quantile_train)

        else:
            raise ValueError('Unrecognized scaling approach.')

        data_train[column] = series_train_scaled
        data_eval[column] = series_eval_scaled

    # Output
    return data_train, data_eval


# Value level
# robust scaling
def _scale_value_robust(value: float, median: float, first_quartile: float, third_quartile: float) -> float:
    # Initialization
    scaled_value = value

    # Scaling if the value is not missing
    if not isnan(value):
        scaled_value = (value - median) / (third_quartile - first_quartile)

    # Output
    return scaled_value


# minmax scaling
def _scale_value_minmax(value: float, lower_quartile: float, upper_quartile: float) -> float:
    # Initialization
    scaled_value = value

    # Scaling if the value is not missing
    if not isnan(value):
        if lower_quartile <= value <= upper_quartile:
            return (value - lower_quartile) / (upper_quartile - lower_quartile)
        elif value < lower_quartile:
            return 0
        elif value > upper_quartile:
            return 1
        else:
            raise ValueError('Unconsidered case.')

    # Output
    return scaled_value
