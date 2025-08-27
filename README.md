# Prediction of the need for oxygen therapy in COVID-19 patients using patient metadata and mass spectrometry of serum proteins

## Introduction
The goal of this project is the predict the 'O2 req.' column within the provided clinical trials datasets, which states the patients current need for oxygen (Y/N) at the time of sampling. The data consists of two datasets that were produced as part of studies on hospitalised COVID-19 patients: the training dataset was sourced from a Surrey hospital, and the validation dataset was sourced from the ISARIC foundation. Each dataset contains a number of general patient health features (Age, Gender, IgG levels, etc.) and hundreds of serum protein intensities derived from mass spectrometry.

The datasets are much smaller than advised for the training of a machine learning model - especially one with such a large number of features - so the results presented here are intended as exploratory, to illustrate the potential of a machine learning model when applied to clinical omics data. There is also a large mismatch between the features of the training and validation dataset, and subsequently within the model there is an option to enable or disable validation-compatibility, which removes or retains the features that are not part of the valdiation set.

In this project, we will consolidate the patient data, pre-process the data (including imputation of missing values with MICE), select the most relevant features, select model candidates, and build a hyperparameter-tuned/cross-validated model for final prediction. For each run of the model development pipeline, a neural network and a traditional machine learning model will be built, with parameters tuned and evaluated as part of the model development pipeline. Subsequent validation is optional.


## Files
Note: the pipeline is intended to be run with the current working directory as the Git repository.

### Automated pipeline scripts
- **default_config.yaml:** Make a copy of this file (named config.yaml) or run data_preprocessing.py to make a default 'config.yaml' for which values can be changed. Define settings for the model run here before running the pipeline.
- **typos.yaml:** Dictionaries of changes to be made to correct typos/make generalisations (e.g. consolidating vitamin B treatment types) within the four comma-separated text columns in the Surrey metadata. Only relevant if the various 'text_features' options are enabled (see default config.yaml file).
- **pipeline.py:** Script to run the entire model development pipeline in order. The input files will be stored in an 'inputs/\[model_name]' directory before copying the final outputs to the model_outputs\[model_name] directory to prevent overwriting.
  - **functions.py:** Required functions for the project, including setting the default graphs styles/palettes.

## Individual scripts
If wanting to run individual segments of the pipeline, then scripts should be run in the following order. The inputs and outputs used are stored in the current working directory, and will be overwritten if run multiple times (with the exception of the model_output file if the option is enabled, in which case certain output files will be copied there instead). 
- **feature_engineering.py:** Initial cleaning, consolidation, and train-test split of the patient data for Surrey and ISARIC. 
- **data_preprocessing.py:** Imputing, encoding, and separating into features and target variable.
- Model building scripts:
  - **model_building.py:** Scaling, feature selection, identification of candidate models, hyperparameter tuning, training of the final model, prediction, and evaluation of the model.
    - **external_validation.py:** Fitting the model to the ISARIC validation dataset for the traditional machine learning model.
  - **neural_network.py:** Scaling, feature selection, early stopping determination, hyperparameter tuning, training of the final model, prediction, and evaluation of the model.
    - **external_validation_NN.py:** Fitting the model to the ISARIC validation dataset for the neural network.

Currently, the **extra_outputs.py** and **extra_graphs.py** generate additional graphs (e.g. overlapping ROC curves), statistical tests (e.g. DeLong's), or re-generate information that has been overwritten by the main scripts. Ideally, these would be incorporated more elegantly into the main scripts where possible (and overwriting issues fixed), such as in cases where certain data is summarised manually before using as an input. However, due to the time constraint imposed upon the project, currently these remain as separate scripts.

## Model outputs
The exact outputs vary based on data configurations and randomness, however an example of the pipeline run with seven different configuratiions is shown below. 'V+/-' represents whether the pipeline was made to be validation compatible or not, 'M+/-' represents whether the general patient health data ('metadata') was included or not, and 'D+/-' represents whether timepoints beyond Day 0 were included or not. Confidence intervals are shown in brackets below each AUROC metric.

AUROC scores on the testing data of the training set:
![Test Heatmap](images/test_heatmap.png)

AUROC scores on the validation set:
![Validation Heatmap](images/validation_heatmap.png)


## Additional changes/fixes
Many improvements and corrections to the script are retained as comments within the code. Further improvements of note are recorded here:
- For some outputs (such as the calibration curve), the current file system will overwrite them as they are stored within e.g. the 'training_data' folder and not the 'training_data/ML' or '/NN' folder which would correct this. Additional testing needs to be carried out to check which graphs are affected and to resolve this without disrupting the other outputs/inputs which are automatically detected.
- Incorporate handling of cwd - detect cwd and handle accordingly, or move to the git repo.
- Ideally I would have done a lot more EDA before starting the development process, which could be added in.
- Currently the Merriweather .ttf file is required in the project repository to generate graphs in the same style, but this cannot be distributed within the repo. Source from Google fonts as  atemporary fix, but should have a failsafe. 