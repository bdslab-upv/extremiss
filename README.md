# extremiss
Numerical data imputation methods for extremely missing data contexts.

## Description
This repository provides pipelines for managing extremely missing numerical data. It includes feature scaling procedures necessary for proper imputation, various imputation methods, and multiple classification pipelines that complement the imputation processes. 

The available scaling methods encompass:
- Robust scaling.
- Min-max scaling.

The imputation methods available are:
- Missing mask imputation: it consists of the replacement of the missing values by a value of zero and the filled values by a value of one, in those features presenting missing values.  
- Mean imputation: for each feature p, it consists of replacing the missing values with the mean value along each feature. 
- Translation and encoding: we propose here an imputing approach named translation and encoding. In this method, non-missing numerical data features undergo translation and then missing values in these numerical features are encoded. 
- K-nearest neighbors imputation: it consists of finding the K-nearest neighbors for each missing feature of a specific observation, based on the other observation features  which are not missing, and then averaging the values of those neighboring points presenting a non-missing value in the feature we aim to fill. 
- Bayesian ridge regression imputation: multivariate imputation strategy that addresses missing values in each feature by modeling them using the available remaining features in a round-robin fashion, employing a distributional approach—a Bayesian paradigm—to handle imputations.
- Generative adversarial imputation networks: method based on generative adversarial neural networks. In this approach, the generator examines certain features of the input data, imputes the missing components based on the observed information, and then produces a complete observation. Conversely, the discriminator analyzes this observation and tries to distinguish between the features that were genuinely observed and those that were imputed.

The classifiers included are:
- K-nearest neighbors.
- Logistic regression.
- Random forest.
- Gradient boosting.
- Multi-layer perceptron.

This code has been evaluated on a protected COVID-19 data repository in the context of the project 'Severity Subgroup Discovery and Classification on COVID-19 Real World Data through Machine Learning and Data Quality assessment' (SUBCOVERWD-19) funded by Fondo Supera COVID-19 by CRUE (Conferencia de Rectores de las Universidades Españolas) - Santander Universidades (Santander Bank).

## Citation
The results are published in the following article, please cite it if you use this code:

Pablo Ferri, Nekane Romero-Garcia, Rafael Badenes, David Lora-Pablos, Teresa García Morales, Agustín Gómez de la Cámara, Juan M García-Gómez and Carlos Sáez. "Extremely missing numerical data in Electronic Health Records for machine learning can be managed through simple imputation methods considering informative missingness: a comparative of solutions in a COVID-19 mortality case study." (Under Review)

## How to use it
1. Download the entire repository.
2. Install the necessary packages as indicated in the requirements.txt file.
3. Update the main.py file following the indications described in the comments: directory where your data are located, which numerical features require imputation, the scaling method to use, the imputation method to consider, etc.
4. Run the main.py script.

## Credits
- **Main developer**: Pablo Ferri
- **Authors**: Pablo Ferri (UPV), Nekane Romero-Garcia (HCUV), Rafael Badenes (HCUV), David Lora-Pablos (H12O), Teresa García Morales (H12O), Agustín Gómez de la Cámara (H12O), Juan M García-Gómez (UPV) and Carlos Sáez (UPV)

Copyright: 2023 - Biomedical Data Science Lab, Universitat Politècnica de Valècia, Spain (UPV)

