# Prediction of the need for oxygen therapy in COVID-19 patients using patient metadata and mass spectrometry of serum proteins

## Introduction
The data originates from a Surrey hospital where COVID-19 patients were sampled, and will later be expanded to include other sites. The project goal is the predict the 'O2 req.' column, which states the patients current need for oxygen (Y/N) at the time of sampling. (TODO: more detail on data, eg surrey timepoints, isaric pmeta crf1a timepoint vs isaric.csv 'day 1' vs unspecified)
TODO Describe ISARIC dataset.
The majority of patients were sampled only upon admission, but data does exist at other timepoints for some paricipants, which may lead to estimation of high-risk patients for requiring oxygen therapy. 

In this project, we will consolidate the patient data, pre-process the data (including imputation of missing values with MICE), select the most relevant features, select model candidates, and build a hyperparameter-tuned/cross-validated model for final prediction.

WIP: Also building a neural network as an alternative form of evaluation.

## Files
### Traditional Machine Learning model:
- **default_config.yaml:** Make a copy of this file (named config.yaml) or run data_preprocessing.py to make a default 'config.yaml' for which values can be changed. Define settings for the model run here. Note: some setting remain embedded within the script to allow for ease of testing. Each setting is explained in the file comments, for example the 'validate' setting which modifies the training dataset to be compatible with the validation dataset. 
- **typos.yaml:** Dictionaries of changed to be made to correct typos/make generalisations (e.g. consolidating vitamin B treatment types) within the four comma-separated text columns in the Surrey metadata.
- **pipeline.py:** Wrapper script to run the entire model development pipeline in order. The input files will be stored in an 'inputs/\[model_name]' directory before copying the final outputs to the model_outputs\[model_name] directory to prevent overwriting. 

If wanting to run individual segments of the pipeline, then scripts should be run in the following order: TODO update this for NN
- **feature_engineering.py:** Initial cleaning, consolidation, and train-test split of the patient data for Surrey and ISARIC. 
- **data_preprocessing.py:** Imputing, encoding, and separating into features and target variable.
- **model_building.py:** Scaling, feature selection, identification of candidate models, hyperparameter tuning, training of the final model, prediction, and evaluation of the model.
- **external_validation.py:** Fitting the model to the ISARIC validation dataset.
- **functions.py:** Required functions for the project.

### Traditional Machine Learning model:
- **neural_network.py:** in-progress script for the neural network.