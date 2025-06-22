import numpy as np
import pandas as pd
from pathlib import Path
from scipy.datasets import ascent
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, f_classif, SelectKBest, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from mlflow.models.signature import infer_signature

# todo clean up at end

# TODO Any other useful metrics to generate? Or adjust what I assess by

# todo do i need to do anything with balancing classes?

# Bool to show additional detail
show_detail = False

# Set pandas to display all columns and longer rows # IMPROVE remove in final version
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

### Read in data
# Train
X_path = Path(__file__).parent / "Surrey_X_train.csv"
y_path = Path(__file__).parent / "Surrey_y_train.csv"
X_train = pd.read_csv(X_path, index_col=0)
y_train = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Test
X_path = Path(__file__).parent / "Surrey_X_test.csv"
y_path = Path(__file__).parent / "Surrey_y_test.csv"
X_test = pd.read_csv(X_path, index_col=0)
y_test = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# TODO Saw a mention of VIF analysis, do I use that in conjunction with thresholding? Currently only do 0 variance thresholding though

### OPTIONAL: DROP METADATA ############################################################################################ #TODO once the model is finished and can run on HPC, run with and without dropping to comapre results
drop_metadata = True # To experiment with excluding metadata from the model, enable this option

# Read in the original metadata file to read the column info
s_meta_file = Path(__file__).parent / "Surrey_Files" / "Surrey_Metadata_master_spreadsheet_130622_edit2.csv"
s_meta = pd.read_csv(s_meta_file)
# List starting and filtered columns
meta_columns = s_meta.columns.tolist()

meta_cols = 0 # Initialise
# Detect metadata columns in the dataset
def count_meta(dataset, name, metadata_features, drop):
    matched = False # Initialise
    existing_columns = dataset.columns.tolist()
    col_number = 0 # Initialise
    for col in reversed(metadata_features):
        if col in existing_columns:
            matched = True
            if show_detail:
                print(f"\nMetadata columns in {name}:")
                print(dataset.columns.get_loc(col) + 1) # +1 for 1-based indexing conversion / allows for splicing where the first number is inclusive and the second exclusive
            col_number = dataset.columns.get_loc(col) + 1
            break
    if not matched:
        if show_detail:
            print("No metadata columns found.")
    if drop: # Drop the metadata if bool is true
        dataset = dataset.iloc[:, col_number:]
        col_number = 0 # Now removed all metadata so count is 0
        print(f"Metadata was dropped from {name}; if unintended, disable drop_metadata in the script.")
    return col_number,dataset

# Call function
meta_cols,X_train = count_meta(X_train, "X_train", meta_columns, drop_metadata)
meta_cols,X_test = count_meta(X_test, "X_test", meta_columns, drop_metadata) # Colummn # should be identical for test and train, so can just reassign

### VARIANCE THRESHOLDING ##############################################################################################
if show_detail:
    print("Feature count before thresholding:", len(X_train.columns))

 # IMPROVE currently set to 0 variance (which removes none of my data) and got a better result - can experiment with other values later. Since sample # is low it may be best with minimal filtering
# Calculate median variance of all features
variances = X_train.var(axis=0)
#threshold = np.median(variances) # Example of thresholds - either delete or experiment with
#threshold = np.quantile(variances, 0.75)
threshold = 0.0

# Creating a preprocessing pipeline to scale and feature select through variance thresholding
preprocessor = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=threshold)),
    ('scaler', StandardScaler())
])
#todo i'm transforming data with a preprocessor outside of the main modeling pipeline, which could lead to data leakage. implement as one pipeline

# Fit on training data and transform test data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Track retained features
scaler = preprocessor.named_steps['scaler']
var_thresh = preprocessor.named_steps['var_thresh']

# Get feature mask after both scaling and thresholding
retained_mask = var_thresh.get_support()
retained_features = X_train.columns[retained_mask]

if show_detail:
    print("Features after thresholding:", len(retained_features))
    print("Retained features:", list(retained_features))

# Step 5: Convert back to DataFrames
X_train = pd.DataFrame(X_train_transformed,
                      columns=retained_features,
                      index=X_train.index)

X_test = pd.DataFrame(X_test_transformed,
                     columns=retained_features,
                     index=X_test.index)

if show_detail:
    print("Feature count after thresholding:", len(X_train.columns))
    print("Retained features:", list(X_train.columns))

### DEFINE FEATURE SELECTION PER MODEL #################################################################################
# Whether to do feature selection or not - applies to all steps (basic training, hyperparameter exploration, and final model training
feature_selection = True

# Define a dictionary mapping classifier types to their optimal feature selectors

# Dictionary of feature selectors to use in basic model training #TODO maybe you could have a params variable if you wanted to specify more. and another dict for param search space
#    Feature selection methods taken from scikit-learn documentation # IMPROVE - methods were chosen to cover a wide range of approaches/models but could be tweaked further
feature_selectors = {
                     # 'RFECV_LR': RFECV(estimator=LogisticRegression(),
                     #                step=1,
                     #                cv=StratifiedKFold(5),
                     #                scoring="accuracy",
                     #                min_features_to_select=1,
                     #                n_jobs=2),
                     # 'RFECV_SVC': RFECV(estimator=SVC(),
                     #                step=1,
                     #                cv=StratifiedKFold(5),
                     #                scoring="accuracy",
                     #                min_features_to_select=1,
                     #                n_jobs=2),
                     # 'RFECV_RF': RFECV(estimator=RandomForestClassifier(),
                     #                step=1,
                     #                cv=StratifiedKFold(5),
                     #                scoring="accuracy",
                     #                min_features_to_select=1,
                     #                n_jobs=2),
                     # 'RFECV_XGB': RFECV(estimator=XGBClassifier(),
                     #                step=1,
                     #                cv=StratifiedKFold(5),
                     #                scoring="accuracy",
                     #                min_features_to_select=1,
                     #                n_jobs=2),
                     # 'SFM_LR': SelectFromModel(estimator=LogisticRegression(solver='saga', tol=1e-3, max_iter=200)),
                     # 'SFM_SVC': SelectFromModel(estimator=SVC()),
                     'SFM_RF': SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, max_depth=5,random_state=42), threshold="median"), # todo added params to replicate prior results - should experiment with all options
                     # 'SFM_XGB': SelectFromModel(estimator=XGBClassifier()),
                     # 'SFM_LAS': SelectFromModel(estimator=Lasso()),
                     # 'SFS_LR': SequentialFeatureSelector(estimator=LogisticRegression(), n_features_to_select=270, direction="forward"),
                     # 'SFS_LSVC': SequentialFeatureSelector(estimator=LinearSVC(), n_features_to_select=270, direction="forward"),
                     # 'SFS_XGB': SequentialFeatureSelector(estimator=XGBClassifier(), n_features_to_select=270, direction="forward"),
                     # 'BORUTA': BorutaPy(estimator=RandomForestClassifier(), n_estimators='auto', max_iter=10),
                     # 'NONE': 'passthrough'
                     }

# feature_selectors_dict = { # WARNING delete
#     'svm': SelectFromModel(LinearSVC(C=0.1, random_state=42, max_iter=2000)),
#     'rf': SelectFromModel(RandomForestClassifier(n_estimators=100,
#                                                 max_depth=5,
#                                                 random_state=42),
#                           threshold="median"),
#     'logreg': SelectFromModel(LogisticRegression(penalty="l1",
#                                                 solver="saga",
#                                                 C=0.1,
#                                                 max_iter=2000,
#                                                 random_state=42)),
#     'xgb': SelectFromModel(XGBClassifier(n_estimators=100,
#                                          max_depth=3,
#                                          random_state=42)),
#     'gb': SelectFromModel(GradientBoostingClassifier(random_state=42)),
#     'knn': SelectKBest(f_classif, k=20),  # Using statistical test instead of RFECV
#     'ada': SelectFromModel(AdaBoostClassifier(n_estimators=50, random_state=42))
# }

### ESTIMATE BEST MODELS WITH BASIC SETTINGS ###########################################################################
# Bool to decide whether to go through the training of basic models to determine the most promising candidates/feature selectors
basic_training = True

# Initialise dict to store best results per model
top_model_scores = {}

# Basic model training function to get some initial scores and decide which model to proceed with
def basic_train(model, X_train, y_train, identifier, scores_dict):
    # Iterate over feature selectors
    model_results = {}
    for fs_name, selector in feature_selectors.items():
        # Create a pipeline with feature selection and classifier - ensures same CV folds/feature selection
        pipe = Pipeline([
            ('feature_selector', selector if feature_selection else 'passthrough'), # If FS is turned off, use passthrough instead of selector
            ('classifier', model)
        ])
        try:
            # 10-fold cross validation for F1 score and accuracy
            f1_val = cross_val_score(pipe, X_train, y_train, scoring='f1', cv=StratifiedKFold(10, shuffle=True, random_state=42))
            accuracy_val = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=StratifiedKFold(10, shuffle=True, random_state=42))

            # Fit the pipeline on the training data
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_train)
            f1_train = f1_score(y_train, y_pred)
            accuracy_train = accuracy_score(y_train, y_pred)

            model_results[fs_name] = [identifier, fs_name, f1_train, f1_val.mean(), accuracy_train, accuracy_val.mean()]
            print(f"Training of {identifier} using {fs_name} complete.")
        except Exception as e:
            print(f"Error training {identifier} with {fs_name}: {str(e)}")
            model_results[identifier] = [identifier, None, None, None, None, None]
    # Print results from best feature selection methods
    model_results_df = pd.DataFrame.from_dict(model_results,
                           orient='index',
                           columns=['Model', 'Selector', 'Train F1', 'Test F1', 'Train Accuracy',
                                    'Test Accuracy']).sort_values(by=['Test Accuracy'], ascending=False)
    print(f"Metrics from {identifier} experimentation:")
    print(model_results_df, "\n")
    # Take the top result unless empty
    if model_results_df.empty:
        scores_dict[identifier] = [identifier, None, None, None, None, None]
        print(f"All feature selection methods failed for {identifier}.")
    else:
        scores_dict[identifier] = model_results_df.iloc[0].to_list()
    print(f"Finished training {identifier}")

# Determine which models to test
Logistic_regression = True
SVM = False
Random_forest = False
AdaBoost = False
Gradient_boosting = False
XGBoost = False
KNN = False

#  WARNING version with all switched on - delete after testing:
# logistic_regression = True
# SVM = True
# Random_forest = True
# AdaBoost = True
# Gradient_boosting = True
# XGBoost = True
# KNN = True

# Dictionary to store the highest performing models and their feature selection methods
best_models_fs = {}

# Outline each model and perform the basic training function to evaluate performance of each
if basic_training:
    # Logistic Regression
    if Logistic_regression:
        log_reg = LogisticRegression(solver='saga', tol=1e-4, max_iter=1500)
        basic_train(log_reg, X_train, y_train, 'Logistic Regression', top_model_scores)

    # SVM
    if SVM:
        svc_clf = SVC()
        basic_train(svc_clf, X_train, y_train, 'Support Vector Classifier', top_model_scores)

    # Random Forest
    if Random_forest:
        rnd_clf = RandomForestClassifier(random_state=42)
        basic_train(rnd_clf, X_train, y_train, 'RandomForestClassifier', top_model_scores)

    # AdaBoost
    if AdaBoost:
        dt_clf_ada = DecisionTreeClassifier()
        ada_clf = AdaBoostClassifier(estimator=dt_clf_ada, random_state=42)
        basic_train(ada_clf, X_train, y_train, "AdaBoost Classifier", top_model_scores)

    # GradientBoosting
    if Gradient_boosting:
        gdb_clf = GradientBoostingClassifier(random_state=42, subsample=0.8)
        basic_train(gdb_clf, X_train, y_train, "GradientBoosting Classifier", top_model_scores)

    # XGBoost
    if XGBoost:
        xgb_clf = XGBClassifier(verbosity=0)
        basic_train(xgb_clf, X_train, y_train, "XGBoost Classifier", top_model_scores)

    # KNN
    if KNN:
        knn_clf = KNeighborsClassifier()
        basic_train(knn_clf, X_train, y_train, 'K-Nearest Neighbors Classifier', top_model_scores)

    # Make dataframe of model scores and print results
    scores = pd.DataFrame.from_dict(top_model_scores,
                                    orient='index',
                                    columns=['Model', 'Selector', 'Train F1', 'Test F1', 'Train Accuracy',
                                             'Test Accuracy']).reset_index(drop=True).sort_values(by='Test Accuracy', ascending=False)
    # Print the top results for each model
    print("\nBest results per model:")
    print(scores.head(len(scores)))

    ### Determine the best performing models to take to the tuning phase
    # Initialise objects
    n_models_to_tune = 1  # The top N models - change this as needed # TODO  disable basic train to just test with RF as previous
    # Extract best model and feature selection from top_model_scores
    for i in range(0, n_models_to_tune):
        model = scores.iloc[i, 0]
        fs = scores.iloc[i, 1]
        best_models_fs[model] = fs
    # Print results
    hypertune = pd.DataFrame(best_models_fs.items(), columns=['Model', 'Selector'])
    print("\nThe models taken to the hyperparameter tuning stage are:\n", hypertune)
else:
    # Set model and feature selector that perform the best - do this manually by adding entries below based on the results of prior basic_training runs
    best_models_fs["RandomForestClassifier"] = "SFM_RF"  # Note: example values included; not actual results
    # ... continue as needed
    hypertune = pd.DataFrame(best_models_fs.items(), columns=['Model', 'Selector'])
    print("\nThe models taken to the hyperparameter tuning stage are:\n", hypertune)


# Example printed result from basic training:
# TODO paste in example output once feature selection is more pinned down/state which perform the best

### HYPEROPT PARAMETER TUNING ##########################################################################################
# For selected models, define a parameter params['type'] for the model name. Then evaluate parameters and calculate the cross-validated accuracy.

# Dictionary to store the best model accuracies
best_accuracies = {
    'svm': 0.0,
    'rf': 0.0,
    'logreg': 0.0,
    'xgb': 0.0,
    'gb': 0.0,
    'ada': 0.0,
    'knn': 0.0
}

type_translation = { # IMPROVE could just use the full name for simplicity and remove this
    'svm': 'Support Vector Classifier',
    'rf': 'RandomForestClassifier',
    'logreg': 'Logistic Regression',
    'xgb': 'XGBoost Classifier',
    'gb': 'GradientBoosting Classifier',
    'ada': 'AdaBoost Classifier',
    'knn': 'K-Nearest Neighbors Classifier'
    }

# Objective function; which parameter configuation is used
def objective(params):
    classifier_type = params['type']
    del params['type']
    # Set feature selector based on classifier type
    selector = feature_selectors[best_models_fs[type_translation[classifier_type]]]

    # Build the classifier based on provided type and convert parameters that must be integers (hyperopt returns floats) if necessary
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    elif classifier_type == 'xgb':
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])
        clf = XGBClassifier(**params)
    elif classifier_type == 'gb':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        clf = GradientBoostingClassifier(**params)
    elif classifier_type == 'ada':
        params['n_estimators'] = int(params['n_estimators'])
        clf = AdaBoostClassifier(**params)
    elif  classifier_type == 'knn':
        params['n_neighbors'] = int(params['n_neighbors'])
        params['leaf_size'] = int(params['leaf_size'])
        clf = KNeighborsClassifier(**params)
    else:
        return {'loss': 1, 'status': STATUS_OK}

    # Incorporate feature selection into the pipeline
    pipe = Pipeline([
        ('feature_selector', selector if feature_selection else 'passthrough'), # If FS is turned off, use passthrough instead of selector
        ('classifier', clf)
    ])

    # Use 10-fold cross validation to compute the mean accuracy
    accuracy = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1').mean()  # Reduced to 5-fold for speed

    # Log the best accuracy for each model type if improved
    if accuracy > best_accuracies[classifier_type]:
        best_accuracies[classifier_type] = accuracy
        mlflow.log_metric(f"best_{classifier_type}_accuracy", accuracy)

    # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

### DEFINE SEARCH SPACES PER MODEL #####################################################################################
  # Define each search space per model type. If in the top performing models (determined in basic train/manually), add to the overall search space #todo split whole code into sections more with headers - done here but need for the rest

best_spaces = [] # Initialise list
### Define space for each model #TODO find more practical examples where these are defined; what is worth definining and what ranges? - google tuning [model]
# SVM
if type_translation['svm'] in best_models_fs:
    best_spaces.append({
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1.0),
        'kernel': hp.choice('svm_kernel', ['linear', 'rbf']),
        'gamma': hp.choice('svm_gamma', ['scale', 'auto']),
        'class_weight': hp.choice('svm_class_weight', [None, 'balanced']),
        'random_state': 42,
    })
# Random forest
if type_translation['rf'] in best_models_fs:
    best_spaces.append({
        'type': 'rf',
        'criterion': hp.choice('rf_criterion', ['gini', 'entropy', 'log_loss']),
        'n_estimators': hp.quniform('rf_n_estimators', 50, 500, 50),
        'max_depth': hp.quniform('rf_max_depth', 2, 10, 1),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', 0.8]),
        'class_weight': hp.choice('rf_class_weight', [None, 'balanced']),
        'random_state': 42,
    })
# Logistic regression
if type_translation['logreg'] in best_models_fs:
   best_spaces.append({
       'type': 'logreg',
       'C': hp.lognormal('lr_C', 0, 1.0),
       'solver': hp.choice('lr_solver', ['liblinear', 'saga']),
       'penalty': hp.choice('lr_penalty', ['l1', 'l2']), # Chosen to be compatible with liblinear and saga
       'class_weight': hp.choice('lr_class_weight', [None, 'balanced']),
       'random_state': 42,
       'max_iter': 3000
   })
# XGBoost
if type_translation['xgb'] in best_models_fs:
    best_spaces.append({
        'type': 'xgb',
        'max_depth': hp.quniform("xgb_max_depth", 3, 15, 1),
        'gamma': hp.uniform('xgb_gamma', 1, 9),
        'reg_alpha': hp.quniform('xgb_reg_alpha', 10, 180, 1),
        'reg_lambda': hp.uniform('xgb_reg_lambda', 0, 5),
        'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.5, 1),
        'min_child_weight': hp.quniform('xgb_min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 50),
        'seed': 0,
        'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
        'scale_pos_weight': hp.uniform('xgb_scale_pos_weight', 1, 10),  # Adjust if classes are imbalanced
        'max_delta_step': hp.uniform('xgb_max_delta_step', 0, 10),
        'random_state': 42,
    })
# Gradient Boosting
if type_translation['gb'] in best_models_fs:
    best_spaces.append({
        'type': 'gb',
        'n_estimators': hp.quniform('gb_n_estimators', 50, 250, 50),
        'max_depth': hp.quniform('gb_max_depth', 3, 15, 1),
        'min_samples_split': hp.quniform('gb_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('gb_min_samples_leaf', 1, 10, 1),
        'learning_rate': hp.loguniform('gb_learning_rate', 0.05, 0.5),
        'subsample': hp.uniform('gb_subsample', 0.6, 1.0),
        'max_features': hp.choice('gb_max_features', ['sqrt', 'log2', 0.8]),
        'loss': hp.choice('gb_loss', ['log_loss', 'exponential']),
        'criterion': hp.choice('gb_criterion', ['friedman_mse', 'squared_error']),
        'random_state': 42,
        # IMPROVE - more tunable features; see sklearn documentation
    })
# AdaBoost # WARNING Ada and KNN are untested - search spaces of these were omitted the first time - not sure if due to a reason or just missed out
if type_translation['ada'] in best_models_fs:
    best_spaces.append({
        'type': 'ada',
        'n_estimators': hp.uniform('ada_n_estimators', 30, 1000),
        'learning_rate': hp.uniform('ada_learning_rate', 0.1, 1.0),
        'estimator': hp.choice('ada_base_estimator', [
            DecisionTreeClassifier(random_state=42),
            LinearSVC(random_state=42), # WARNING: LinearSVC is untested
            LogisticRegression(random_state=42),
        ]),
        'random_state': 42,
    })
# K-Nearest Neighbors
if type_translation['knn'] in best_models_fs:
    best_spaces.append({
        'type': 'knn',
        'n_neighbors': hp.quniform('knn_n_neighbors', 1, 50, 1),
        'weights': hp.choice('knn_weights', ['uniform', 'distance']),
        'leaf_size': hp.uniform('knn_leaf_size', 10, 60),
        'p': hp.choice('knn_p', [1, 2]),
        'metric': hp.choice('knn_metric', ['minkowski', 'euclidean', 'cityblock']),
        'random_state': 42,
    })


# Define the search space over hyperparameters (for classifier only; feature selection is determined elsehwere)
search_space = hp.choice('classifier_type', best_spaces)

print("\nNow tuning hyperparameters\n")

with mlflow.start_run():
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50, #todo: think this needs to be increased on a better machine
        trials=Trials()
    )

# Print the best accuracies for each model type
print("\nHighest model accuracies on train data:")
best_accuracy_df = pd.DataFrame(list(best_accuracies.items()), columns=['Models', 'Highest accuracy'])
print(best_accuracy_df)

# Extract and print the best hyperparameter configuration
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

### TRAIN FINAL MODEL ###########################################################################################
# Train final model using the full training data
mlflow.sklearn.autolog()
with mlflow.start_run():  # TODO need to find examples of this being done - unsure on the final training/testing after hyperopt tuning
    classifier_type = best_config['type'] # Extract best classifier type
    best_params = {k: v for k, v in best_config.items() if k != 'type'} #Extract best hyperparameters
    # Set feature selector based on classifier type
    selector = feature_selectors[best_models_fs[type_translation[classifier_type]]]

    # Log the best hyperparameters
    mlflow.log_params(best_config)

    # Construct the classifier with the best parameters - converting to integers if needed
    if classifier_type == 'svm':
        classifier = SVC(**best_params)
    elif classifier_type == 'rf':
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        classifier = RandomForestClassifier(**best_params)
    elif classifier_type == 'logreg':
        classifier = LogisticRegression(**best_params)
    elif classifier_type == 'xgb':
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        classifier = XGBClassifier(**best_params)
    elif classifier_type == 'gb':
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        classifier = GradientBoostingClassifier(**best_params)
    elif classifier_type == 'ada': #TODO ada and knn were recently added so not tested - e.g. which params need integer conversion as abov
        best_params['n_estimators'] = int(best_params['n_estimators'])
        classifier = AdaBoostClassifier(**best_params)
    elif  classifier_type == 'knn':
        best_params['n_neighbors'] = int(best_params['n_neighbors'])
        best_params['leaf_size'] = int(best_params['leaf_size'])
        classifier = KNeighborsClassifier(**best_params)

    # Create the final pipeline with feature selection and classifier # TODO not sure if i need pipeline here since I dont feed into cross_val_score?
    final_pipeline = Pipeline([
        ('feature_selector', selector if feature_selection else 'passthrough'), # If FS is turned off, use passthrough instead of selector
        ('classifier', classifier)
    ])

    # Train on full training data
    final_pipeline.fit(X_train, y_train)
# WARNING: From here needs testing/improvement
    # Print the selected features
    try: # TODO this was made when RFECV was used for all, so likely no longer works
        selected = final_pipeline.named_steps['feature_selector']
        selected_features = X_train.columns[selected.support_]
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features.tolist())
    except:
        print("Unable to print features - see note in code.")

    ### Log the final pipeline model
    # Create input example
    input_example = X_train.iloc[:1]
    # Infer model signature
    signature = infer_signature(X_train, final_pipeline.predict(X_train))
    mlflow.sklearn.log_model(final_pipeline, "best_model", signature=signature, input_example=input_example)

    # Save final model #todo not sure what to do with this yet but worth saving - or does mlflow save?
    joblib.dump(final_pipeline, "Oxygen_Prediction_Model.joblib")
    mlflow.log_artifact("Oxygen_Prediction_Model.joblib")

    # Evaluate the final model on the test set
    y_pred = final_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    print(f"\nTest accuracy with best model ({classifier_type}): {test_accuracy:.4f}")
    print(f"Test F1 score with best model ({classifier_type}): {test_f1:.4f}")

    ### Plot learning curve #todo not sure if this (and auc) is in the right place - check over code too
    # Compute scores at varying training sizes
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=final_pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )

    # Plot the learning curve
    fig, ax = plt.subplots()
    LearningCurveDisplay(
        train_sizes=train_sizes,
        train_scores=train_scores,
        test_scores=val_scores
    ).plot(ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")

    # Log the figure as an MLflow artifact
    mlflow.log_figure(fig, "learning_curve.png")
    plt.close(fig)

    ### Plot ROC/AUC curves
    # Get prediction probabilities
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot and log
    fig, ax = plt.subplots()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

    # Log the AUC metric explicitly
    mlflow.log_metric("test_auc", roc_auc)


    # Get feature list
    # Grab the fitted selector step
    sel = final_pipeline.named_steps['feature_selector']

    # If it’s a scikit‐learn selector (RFECV, SelectKBest, etc.), you can use .get_support() # todo make compatible with all
    mask = sel.get_support()  # boolean mask of length n_features

    # Apply that mask to the original feature names
    feature_names = X_train.columns[mask]
    print(f"{len(feature_names)} features selected:")
    print(feature_names.tolist())

    ### Plot feature importance #todo test for all
    # Extract feature importances based on the classifier type
    if classifier_type in ['rf', 'xgb', 'gb']:
        # These classifiers have feature_importances_ attribute
        importances = final_pipeline.named_steps['classifier'].feature_importances_

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

        # Create DataFrame for easier plotting with seaborn
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
        ax.set_title(f'Top 20 Feature Importances - {classifier_type.upper()}', fontsize=16)
        ax.set_xlabel('Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        print(f"\nTop 10 most important features:")
        print(importance_df.head(10))

    elif classifier_type == 'svm' and best_params.get('kernel') == 'linear': #todo ??
        # For linear SVM, we can extract coefficients
        coefficients = np.abs(final_pipeline.named_steps['classifier'].coef_[0])

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Coefficient', y='Feature', data=importance_df.head(20), palette='viridis')
        ax.set_title('Top 20 Feature Coefficients - Linear SVM', fontsize=16)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "feature_coefficients.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("feature_coefficients.csv", index=False)
        mlflow.log_artifact("feature_coefficients.csv")

        print(f"\nTop 10 most important features (by coefficient magnitude):")
        print(importance_df.head(10))

    elif classifier_type == 'logreg':
        # For logistic regression, extract coefficients
        coefficients = np.abs(final_pipeline.named_steps['classifier'].coef_[0])

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Coefficient', y='Feature', data=importance_df.head(20), palette='viridis')
        ax.set_title('Top 20 Feature Coefficients - Logistic Regression', fontsize=16)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "feature_coefficients.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("feature_coefficients.csv", index=False)
        mlflow.log_artifact("feature_coefficients.csv")

        print(f"\nTop 10 most important features (by coefficient magnitude):")
        print(importance_df.head(10))

    else:
        # For other models where direct feature importance is not available
        # Use permutation importance as an alternative
        print("\nCalculating permutation importance for features...")

        # Calculate permutation importance
        perm_importance = permutation_importance(
            final_pipeline, X_test, y_test,
            n_repeats=10,
            random_state=42
        )

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
        ax.set_title(f'Top 20 Permutation Feature Importances - {classifier_type.upper()}', fontsize=16)
        ax.set_xlabel('Mean Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig("permutation_importance.png", dpi=300, bbox_inches='tight') # todo added - but logs below so not sure if redundant

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "permutation_importance.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("permutation_importances.csv", index=False)
        mlflow.log_artifact("permutation_importances.csv")

        print(f"\nTop 10 most important features (by permutation importance):")
        print(importance_df.head(10))


    ### END OF GRAPHS todo check over

# Example output:
# Test accuracy with best model (rf): 0.6000
# Test F1 score with best model (rf): 0.6957


# TODO: Question: If I get different models (LR and GB currently) on different runs, what should I do? Pick one? Use the
#  most common of multiple attempts? Or set seed so it's always consistent


#TODO deleted notes in the code but possibly implement a FS search space

# IMPROVE: Early stopping isn't implemented at all because it would work for some and not others so is more complicated to implement - but could add.
#  Could also do an ensemble model approach for the final training, and stacking/voting
#  More elegant way to handle hyperopt returning floats?

