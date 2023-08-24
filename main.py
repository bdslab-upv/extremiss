"""
DESCRIPTION: main script.
AUTHOR: Pablo Ferri
DATE: 20/08/2023
"""

# MODULES IMPORT
from os.path import join

from pandas import read_csv
from Preparation.prepfunc import split_data, scale_data
from Imputation.imputfunc import impute_missings
from Classification.classifunc import initialize_model, train_model_calculate_predictions

# SETTINGS
# Data directory
# replace with the directory where your data is located
data_directory = 'C:/Users/Pablo/Desktop/Universitat/BDSLab/extremiss/'

# Data filename
# replace with the filename of your csv data file
data_filename = 'data_original.csv'

# Delimiter used in the csv file
# replace with your csv data file delimiter
delimiter = ';'

# Numerical features to be imputed
# replace with your actual feature names of those numerical features to be imputed
features2impute = ['FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_5']

# Feature identifiers
# replace with your actual feature names of those numerical features to be imputed plus those which do not require
# to be scaled or imputed (they are already prepared), but they need to be considered for the classification task
feature_identifiers = ['FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_5', 'FEATURE_6', 'FEATURE_7',
                       'FEATURE_8']

# Scaling method specification
# to choose between 'robust', 'minmax'
scaling_method = 'minmax'

# Imputation method specification
# to choose among 'missing_mask', 'mean', 'translation_encoding', 'bayesian_regression', 'knn', 'gain'
# 'missing_mask', 'mean', 'translation_encoding', 'bayesian_regression' and 'knn' are intended to be used in
# combination with 'robust' scaling
# 'translation_encoding' and 'gain' require 'minmax' scaling
imputation_method = 'translation_encoding'

# Classification model specification
# to choose among 'k_nearest_neighbors', 'logistic_regression', 'random_forest', 'gradient_boosting' and
# 'multilayer_perceptron'
model_identifier = 'multilayer_perceptron'

# Classification label specification
# update with the actual label identifier
label_identifier = 'OUTCOME'
# update with the actual number of classes
number_classes = 2

# EXECUTION
if __name__ == '__main__':
    # DATA LOADING
    # Filepath definition
    absolute_filepath = join(data_directory, data_filename)

    # Loading
    data = read_csv(filepath_or_buffer=absolute_filepath, delimiter=delimiter, encoding='latin-1', engine='python')

    # DATA PREPARATION
    # Data splitting
    data_split = split_data(data)

    # Workspace cleaning
    del data

    # Data scaling
    data_split_scaled = scale_data(data_split=data_split, scaling_method=scaling_method, columns2scale=features2impute)

    # Workspace cleaning
    del data_split

    # DATA IMPUTATION
    data_split_imputed = impute_missings(data_split=data_split_scaled, imputation_method=imputation_method,
                                         scaling_method=scaling_method, columns2impute=features2impute,
                                         feature_identifiers=feature_identifiers)

    # CLASSIFICATION
    # Data extraction
    # To simplify readability we present the classification pipeline with the 'train' and 'test' partition
    data_train = data_split_imputed['train_test']['train']
    data_test = data_split_imputed['train_test']['test']

    # Classifier initialization
    model = initialize_model(model_identifier=model_identifier, scaling_method=scaling_method,
                             imputation_method=imputation_method, feature_identifiers=feature_identifiers,
                             number_classes=number_classes)

    # Classifier training and predictions calculation
    labels_predictions = train_model_calculate_predictions(
        model=model, data_train=data_train, data_eval=data_test, feature_identifiers=feature_identifiers,
        label_identifier=label_identifier, scaling_method=scaling_method, imputation_method=imputation_method)
