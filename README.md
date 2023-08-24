# extremiss
Numerical data imputation methods for extremely missing data contexts.

## Description
This repository provides pipelines for managing extremely missing numerical data. It includes feature scaling procedures necessary for proper imputation, various imputation methods, and multiple classification pipelines that complement the imputation processes. 

The scaling methods available are:
- Robust scaling.
- Min-max scaling.

The imputation methods available are:
- Missing mask imputation.
- Mean imputation.
- Translation and encoding.
- K-nearest neighbors imputation.
- Bayesian regression imputation.
- Generative adversarial imputation networks.

The classifiers available are:
- K-nearest neighbors.
- Logistic regression.
- Random forest.
- Gradient boosting.
- Multi-layer perceptron.

This code has been evaluated on a protected COVID-19 data repository by the SUBCOVERWD-19 in the context of the project Severity Subgroup Discovery and Classification on COVID-19 Real World Data through Machine Learning and Data Quality assessment (SUBCOVERWD-19) funded by Fondo Supera COVID-19 by CRUE (Conferncia de Rectores de las Universidades Españolas) - Santander Universidades (Santander Bank). The results are published in the following article, please cite it if you use this code:
Pablo Ferri, Nekane Romero-Garcia, Rafael Badenes, David Lora-Pablos, Teresa García Morales, Agustín Gómez de la Cámara, Juan M García-Gómez and Carlos Sáez  "Extremely missing numerical data in Electronic Health Records for machine learning can be managed through simple imputation methods considering informative missingness: a comparative of solutions in a COVID-19 mortality case study." Computer Methods and Programs in Biomedicine (2023) (Under Review)

## How to use it
1. Download the entire repository.
2. Install the necessary packages as indicated in the requirements.txt file.
3. Update the main.py file following the indications highlighted by the TODO statements (directory where data is located, scaling method to use, imputation method to consider, etc.).
4. Run the main_prepdat.py script, indicating in the SETTINGS section the numerical features to be imputed and the chosen scaling method.

## Credits
- **Main developer**: Pablo Ferri
- **Authors**: Pablo Ferri (UPV), Nekane Romero-Garcia (UV), Rafael Badenes (UV), David Lora-Pablos (H12O), Teresa García Morales (H12O), Agustín Gómez de la Cámara (H12O), Juan M García-Gómez (UPV) and Carlos Sáez (UPV)

Copyright: 2023 - Biomedical Data Science Lab, Universitat Politècnica de Valècia, Spain (UPV)
