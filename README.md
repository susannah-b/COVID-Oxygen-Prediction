# Prediction of the need for oxygen therapy in COVID-19 patients using patient metadata and mass spectrometry of serum proteins

## Introduction
The data originates from a Surrey hospital where COVID-19 patients were sampled, and will later be expanded to include other sites. The project goal is the predict the 'O2 req.' column, which states the patients current need for oxygen (Y/N) at the time of sampling. 
The majority of patients were sampled only upon admission, but data does exist at other timepoints for some paricipants, which may lead to estimation of high-risk patients for requiring oxygen therapy. 

In this project, we will consolidate the patient data, pre-process the data (including imputation of missing values with MICE), select the most relevant features, select model candidates, and build a hyperparameter-tuned/cross-validated model for final prediction.

## Scripts
Scripts should be run in the following order:
- **data_investigation.py:** Initial cleaning, consolidation, and train-test split of the patient data for Surrey.
- **data_preprocessing.py:** Imputing, encoding, and separating into features and target variable.
- **model_building.py:** Scaling, feature selection, identification of candidate models, hyperparameter tuning, training of the final model, prediction, and evaluation of the model.

---
Please note that the current code is for a preliminary model - many stages exist to be refined later in the project.