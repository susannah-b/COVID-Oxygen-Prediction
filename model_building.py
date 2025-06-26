import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, f_classif, SelectKBest, RFECV, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
# todo clean up at end

# WARNING - suppress MLflow warning and precision warning - fix later
from mlflow.exceptions import MlflowException
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Bool to show additional detail
show_detail = False

### READ IN DATA #######################################################################################################
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

### OPTIONAL: DROP METADATA ############################################################################################ #TODO once the model is finished and can run on HPC, run with and without dropping to compare results
drop_metadata = False # To experiment with excluding metadata from the model, enable this option

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
  # Applied in the scikit-learn pipeline
 # IMPROVE currently set to 0 variance (which removes none of my data) and got a better result - can experiment with other values later. Since sample # is low it may be best with minimal filtering
# Calculate median variance of all features
variances = X_train.var(axis=0)
#threshold = np.median(variances) # Example of thresholds - either delete or experiment with
#threshold = np.quantile(variances, 0.75)
threshold = 1e-10 # Effectively zero but avoids floating-point issues

### VARIANCE INFLATION FACTOR ANALYSIS #################################################################################
# Note: currently nothing further is done with these results. The values are very high and across the entire dataset which is a problem for linear regression models: could be bypassed by omitting linear regression models (logistic regression)
#   WARNING for final results switch off logistic regression
# Make dataframe for results
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns

if show_detail:
    # Do VIF analysis
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]

    # Filter to any VIF above 5 to examine multicolinear features
    high_vif_data = vif_data[vif_data['VIF'] >= 5]

    # Sort and display results
    print("\nVIF Results (features with VIF over 5):\n", high_vif_data.sort_values(by="VIF", ascending=True))
    print("\nFull dataset length including low VIF scores:", len(vif_data))

    #  Check matrix rank
    matrix_rank = np.linalg.matrix_rank(X_train)
    print(f"\nMatrix rank: {matrix_rank}/{X_train.shape[1]} features are linearly independent")

### CONVERT INTEGER COLUMNS TO FLOAT ###################################################################################
  # Safely handles missing values
class IntToFloatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Only convert if DataFrame (preserves column names)
        if isinstance(X, pd.DataFrame):
            int_cols = X.select_dtypes(include=['int', 'int32', 'int64']).columns
            X[int_cols] = X[int_cols].astype(float)
        return X

### DEFINE FEATURE SELECTION PER MODEL #################################################################################
# Whether to do feature selection or not - applies to all steps (basic training, hyperparameter exploration, and final model training
feature_selection = True

### Feature selection methods taken from scikit-learn documentation # IMPROVE - methods were chosen to cover a wide range of approaches/models but could be tweaked further
# Dictionary of feature selector options. base_params are fixed parameters that also apply to basic_train, with other parameters tunable later in the search space #IMPROVE Go over docu and check parameter options
feature_selectors = { #TODO need to test and uncomment all methods - and look up sklearn docu to refine base/search space params and options
    # # RFECV with Logistic Regression
    # 'RFECV_LR': {
    #     'class': RFECV,
    #     'base_params': {
    #         'estimator': LogisticRegression(),
    #         'step': 1,
    #         'cv': StratifiedKFold(5),
    #         'scoring': "f1",
    #         'min_features_to_select': 1,
    #         'n_jobs': 2
    #     }
    # },
    # # RFECV with Support Vector Classifier
    # 'RFECV_SVC': {
    #     'class': RFECV,
    #     'base_params': {
    #         'estimator': SVC(),
    #         'step': 1,
    #         'cv': StratifiedKFold(5),
    #         'scoring': "f1",
    #         'min_features_to_select': 1,
    #         'n_jobs': 2
    #     }
    # },
    # # RFECV with Random Forest
    # 'RFECV_RF': {
    #     'class': RFECV,
    #     'base_params': {
    #         'estimator': RandomForestClassifier(),
    #         'step': 1,
    #         'cv': StratifiedKFold(5),
    #         'scoring': "f1",
    #         'min_features_to_select': 1,
    #         'n_jobs': 2
    #     }
    # },
    # # RFECV with XGBoost
    # 'RFECV_XGB': {
    #     'class': RFECV,
    #     'base_params': {
    #         'estimator': XGBClassifier(),
    #         'step': 1,
    #         'cv': StratifiedKFold(5),
    #         'scoring': "f1",
    #         'min_features_to_select': 1,
    #         'n_jobs': 2
    #     }
    # },
    # # SelectFromModel with Logistic Regression
    # 'SFM_LR': {
    #     'class': SelectFromModel,
    #     'base_params': {
    #         'estimator': LogisticRegression(solver='saga', tol=1e-3, max_iter=200)
    #     }
    # },
    # # SelectFromModel with Support Vector Classifier
    # 'SFM_SVC': {
    #     'class': SelectFromModel,
    #     'base_params': {
    #         'estimator': SVC()
    #     }
    # },
    # SelectFromModel with Random Forest
    'SFM_RF': {
        'class': SelectFromModel,
        'base_params': {
            'estimator': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            'threshold': "median"
        }
    },
    # # SelectFromModel with XGBoost
    # 'SFM_XGB': {
    #     'class': SelectFromModel,
    #     'base_params': {
    #         'estimator': XGBClassifier()
    #     }
    # },
    # # SelectFromModel with Lasso
    # 'SFM_LAS': {
    #     'class': SelectFromModel,
    #     'base_params': {
    #         'estimator': Lasso()
    #     }
    # },
    # # Sequential Feature Selection with Logistic Regression
    # 'SFS_LR': {
    #     'class': SequentialFeatureSelector,
    #     'base_params': {
    #         'estimator': LogisticRegression(),
    #         'n_features_to_select': 'auto',
    # }
    # },
    # # Sequential Feature Selection with Linear SVC
    # 'SFS_LSVC': {
    #     'class': SequentialFeatureSelector,
    #     'base_params': {
    #         'estimator': LinearSVC(),
    #         'n_features_to_select': 'auto',
    # }
    # },
    # # Sequential Feature Selection with XGBoost
    # 'SFS_XGB': {
    #     'class': SequentialFeatureSelector,
    #     'base_params': {
    #         'estimator': XGBClassifier(),
    #         'n_features_to_select': 'auto',
    # }
    # },
    # # No feature selection
    # 'NONE': {
    #     'class': None,
    #     'base_params': {}
    # }
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
    for fs_name, selector_config in feature_selectors.items():
        # Build selector from configuration
        if fs_name == 'NONE':
            selector = 'passthrough'
        else:
            # Instantiate selector with base parameters
            selector = selector_config['class'](**selector_config['base_params'])

        # Create a pipeline with a) the required preprocessing steps and b) the FS and model. This is then applied to each CV fold to avoid data leakage vs applying to all of X_train
        pipe = Pipeline([
            ('preprocessor', Pipeline([
                ('int_to_float', IntToFloatTransformer()),
                ('var_thresh', VarianceThreshold(threshold=threshold)),
                ('scaler', StandardScaler())
            ])),
            ('feature_selector', selector if feature_selection else 'passthrough'),
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

            #model_results[fs_name] = [identifier, fs_name, f1_train, f1_val.mean(), accuracy_train, accuracy_val.mean()]
            model_results[fs_name] = [identifier, fs_name, accuracy_train, accuracy_val.mean(), f1_train, f1_val.mean(), ]
            print(f"Training of {identifier} using {fs_name} complete.")
        except Exception as e:
            print(f"Error training {identifier} with {fs_name}: {str(e)}")
            model_results[identifier] = [identifier, None, None, None, None, None]
    # Print results from best feature selection methods
    model_results_df = pd.DataFrame.from_dict(model_results,
                           orient='index',
                           columns=['Model', 'Selector', 'Train Accuracy', 'CV Accuracy', 'Train F1',
                                    'Test F1']).sort_values(by=['Test F1'], ascending=False)
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
Logistic_regression = False
SVM = False
Random_forest = True
AdaBoost = False
Gradient_boosting = False
XGBoost = False
KNN = False

#  WARNING version with all switched on - delete after testing:
# Logistic_regression = True
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
                                    columns=['Model', 'Selector', 'Train Accuracy', 'CV Accuracy', 'Train F1',
                                             'Test F1']).reset_index(drop=True).sort_values(by='Test F1', ascending=False)
    # Print the top results for each model
    print("\nBest results per model:")
    print(scores.head(len(scores)))

    ### Determine the best performing models to take to the tuning phase
    # Initialise objects
    n_models_to_tune = 1  # The top N models - change this as needed #IMPROVE change back after tests - 2 or 3
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

### OBJECTIVE FUNCTION FOR HYPEROPT PARAMETER TUNING ###################################################################
# For selected models, define a parameter params['type'] for the model name. Then evaluate parameters and calculate the cross-validated accuracy.

# Dictionary to store the best model accuracies
best_f1 = {
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
    # Get classifier type for each search space
    classifier_type = params['type']
    del params['type']

    # Set feature selector and parameters based on classifier type (as defined by basic_train)
    fs_params = params.pop('fs_params', {}) # Remove FS params from classifier search space

    # Get selector configuration
    selector_type = best_models_fs[type_translation[classifier_type]]
    selector_config = feature_selectors[selector_type]

    ### Build the feature selector
    if selector_type == 'NONE':
        selector = 'passthrough'
    else:
        # Merge base parameters with tuned parameters
        all_params = {**selector_config['base_params'], **fs_params}
        selector = selector_config['class'](**all_params)
        print(f"Now tuning: {classifier_type} with {selector_type}") #IMPROVE maybe delete this after testing - otherwise it's one print per eval

    ### Build the classifier based on provided type and convert parameters that must be integers (hyperopt returns floats) if necessary
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
        ('int_to_float', IntToFloatTransformer()),
        ('feature_selector', selector if feature_selection else 'passthrough'), # If FS is turned off, use passthrough instead of selector
        ('classifier', clf)
    ])
    # Incorporate required preprocessing/FS steps and the model
    pipe = Pipeline([
        ('preprocessor', Pipeline([
            ('int_to_float', IntToFloatTransformer()),
            ('var_thresh', VarianceThreshold(threshold=threshold)),
            ('scaler', StandardScaler())
        ])),
        ('feature_selector', selector if feature_selection else 'passthrough'), # If FS is turned off, use passthrough instead of selector
        ('classifier', clf)
    ])

    # Use 10-fold cross validation to compute the mean accuracy
    f1_score_mean = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1').mean()  # Reduced to 5-fold for speed

    # Log the best accuracy for each model type if improved
    if f1_score_mean > best_f1[classifier_type]:
        best_f1[classifier_type] = f1_score_mean
        mlflow.log_metric(f"best_{classifier_type}_F1", f1_score_mean)

    # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
    return {'loss': -f1_score_mean, 'status': STATUS_OK}

### DEFINE SEARCH SPACES PER FEATURE SELECTOR ########################################################################## # TODO - go over documentation and check which options to include for each parameter, and decide whether to go in base params or the search space - AI-gened for now
selector_param_spaces = { # Note: for new data, values may need to be tweaked as in feature selection parameter tuning, some fits can fail and crash the script
    'SFM_RF': {
        'threshold': hp.choice('sfm_rf_threshold', [None, 'median', 'mean'])
    },
    'RFECV_SVC': {
        'step': hp.uniform('rfecv_step', 0.01, 0.3),
        'min_features_to_select': hp.quniform('rfecv_min_feat', 5, 30, 1)
    },
    'SFM_XGB': {
        'max_features': hp.uniform('xgb_max_feat', 0.1, 1.0),
        'threshold': hp.choice('xgb_threshold', ['median', 0.5, 1.0])
    },
    'RFECV_LR': {
        'step': hp.uniform('rfecv_lr_step', 0.01, 0.3),
        'min_features_to_select': hp.quniform('rfecv_lr_min_feat', 5, 30, 1)
    },
    'RFECV_RF': {
        'step': hp.uniform('rfecv_rf_step', 0.01, 0.3),
        'min_features_to_select': hp.quniform('rfecv_rf_min_feat', 5, 30, 1)
    },
    'RFECV_XGB': {
        'step': hp.uniform('rfecv_xgb_step', 0.01, 0.3),
        'min_features_to_select': hp.quniform('rfecv_xgb_min_feat', 5, 30, 1)
    },
    'SFM_LR': {
        'threshold': hp.choice('sfm_lr_threshold', [None, 'median', 'mean', 0.1, 0.5, 1.0])
    },
    'SFM_SVC': {
        'threshold': hp.choice('sfm_svc_threshold', [None, 'median', 'mean', 0.1, 0.5, 1.0])
    },
    'SFM_LAS': {
        'threshold': hp.choice('sfm_las_threshold', [None, 'median', 'mean', 0.1, 0.5, 1.0])
    },
    'SFS_LR': {
        'n_features_to_select': hp.quniform('sfs_lr_n_features', 5, min(50, X_train.shape[1]), 1)
    },
    'SFS_LSVC': {
        'n_features_to_select': hp.quniform('sfs_lsvc_n_features', 5, min(50, X_train.shape[1]), 1)
    },
    'SFS_XGB': {
        'n_features_to_select': hp.quniform('sfs_xgb_n_features', 5, min(50, X_train.shape[1]), 1)
    },
}

### DEFINE SEARCH SPACES PER MODEL #####################################################################################
  # Define each search space per model type. If in the top performing models (determined in basic train/manually), add to the overall search space

best_spaces = [] # Initialise list
### Define space for each model
# SVM
if type_translation['svm'] in best_models_fs:
    best_spaces.append({
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1.0),
        'kernel': hp.choice('svm_kernel', ['linear', 'rbf']),
        'gamma': hp.choice('svm_gamma', ['scale', 'auto']),
        'class_weight': hp.choice('svm_class_weight', [None, 'balanced']),
        'random_state': 42,
        'probability' : True
    })
# Random forest
if type_translation['rf'] in best_models_fs:
    best_spaces.append({
        'type': 'rf',
        'criterion': hp.choice('rf_criterion', ['gini', 'entropy', 'log_loss']),
        'n_estimators': hp.quniform('rf_n_estimators', 50, 500, 25),
        'max_depth': hp.quniform('rf_max_depth', 2, 10, 1),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', 0.8]),
        'class_weight': hp.choice('rf_class_weight', [None, 'balanced']),
        'random_state': 42,
        'fs_params': selector_param_spaces[best_models_fs[type_translation['rf']]] # The FS shorthand name, e.g. SFM_RF
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
        'n_estimators': hp.quniform('gb_n_estimators', 50, 250, 25),
        'max_depth': hp.quniform('gb_max_depth', 3, 15, 1),
        'min_samples_split': hp.quniform('gb_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('gb_min_samples_leaf', 1, 10, 1),
        'learning_rate': hp.loguniform('gb_learning_rate', 0.05, 0.5),
        'subsample': hp.uniform('gb_subsample', 0.6, 1.0),
        'max_features': hp.choice('gb_max_features', ['sqrt', 'log2', 0.8]),
        'loss': hp.choice('gb_loss', ['log_loss', 'exponential']),
        'criterion': hp.choice('gb_criterion', ['friedman_mse', 'squared_error']),
        'random_state': 42,
    })
# AdaBoost
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
        'n_neighbors': hp.quniform('knn_n_neighbors', 1, 11, 1), # For the Surrey data, this had to be limited to a smaller max value
        'weights': hp.choice('knn_weights', ['uniform', 'distance']),
        'leaf_size': hp.uniform('knn_leaf_size', 10, 60),
        'p': hp.choice('knn_p', [1, 2]),
        'metric': hp.choice('knn_metric', ['minkowski', 'euclidean', 'cityblock']),
    })


# Define the search space over hyperparameters (for classifier only; feature selection is determined elsehwere)
search_space = hp.choice('classifier_type', best_spaces)

### HYPEROPT TUNING WITH MLFLOW ########################################################################################
print("\nNow tuning hyperparameters\n")

with mlflow.start_run():
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=3, #todo: increase on a better machine
        trials=Trials()
    )

# Print the best accuracies for each model type
print("\nHighest model accuracies on train data:")
best_f1_df = pd.DataFrame(list(best_f1.items()), columns=['Models', 'Highest F1'])
print(best_f1_df)

# Extract and print the best hyperparameter configuration
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

### TRAIN FINAL MODEL ###########################################################################################
# Train final model using the full training data
mlflow.sklearn.autolog()
with mlflow.start_run():
    # Extract best classifier type
    classifier_type = best_config['type']

    # Get parameters for the classifier and feature selector
    fs_params = best_config.get('fs_params', {})  # Feature selector params
    best_params = {k: v for k, v in best_config.items() if k not in ['type', 'fs_params']}

    # Get selector configuration
    selector_type = best_models_fs[type_translation[classifier_type]]
    selector_config = feature_selectors[selector_type]

    # Build selector
    if selector_type == 'NONE':
        selector = 'passthrough'
    else:
        all_params = {**selector_config['base_params'], **fs_params}
        selector = selector_config['class'](**all_params)


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
    elif classifier_type == 'ada':
        best_params['n_estimators'] = int(best_params['n_estimators'])
        classifier = AdaBoostClassifier(**best_params)
    elif  classifier_type == 'knn':
        best_params['n_neighbors'] = int(best_params['n_neighbors'])
        best_params['leaf_size'] = int(best_params['leaf_size'])
        classifier = KNeighborsClassifier(**best_params)

    # Create the final pipeline with feature selection and classifier
    final_pipeline = Pipeline([
        ('preprocessor', Pipeline([
            ('int_to_float', IntToFloatTransformer()),
            ('var_thresh', VarianceThreshold(threshold=0.0)),
            ('scaler', StandardScaler())
        ])),
        ('feature_selector', selector if feature_selection else 'passthrough'),
        ('classifier', classifier)
    ])

    # Train on full training data
    final_pipeline.fit(X_train, y_train)

    if show_detail:
        # Track retained features post-preprocessing
        preprocessor = final_pipeline.named_steps['preprocessor']
        var_thresh = preprocessor.named_steps['var_thresh']
        retained_mask = var_thresh.get_support()
        retained_features = X_train.columns[retained_mask]
        print("Features after thresholding:", len(retained_features.tolist()))
        show_features = False # Enable or disable as required
        if show_features:
            print(retained_features.tolist())
        else:
            print("show_features is disabled in model_building.py. To view features as a list, enable this bool.")

    # Print the selected features post-feature selection method # WARNING Not tested with all methods
    try:
        selector = final_pipeline.named_steps['feature_selector']
        if hasattr(selector, 'get_support'): # Standard scikit-learn selector
            support_mask = selector.get_support()
            selected_features = X_train.columns[support_mask].tolist()
        elif hasattr(selector, 'support_'):# Other selector types
            selected_features = X_train.columns[selector.support_].tolist()
        else: # For other selector types, get features via transformation
            print("Feature selection method is incompatible with current handling to extract features - results are not printed.")
        # Print features
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features)
    except Exception as e:
        print(f"Unable to print features for this feature selection method: {str(e)}")

    ### Log the final pipeline model
    # Create input example
    input_example = X_train.iloc[:1]
    # Infer model signature
    signature = infer_signature(X_train, final_pipeline.predict(X_train))
    mlflow.sklearn.log_model(final_pipeline, "best_model", signature=signature, input_example=input_example)

    # Evaluate the final model on the test set
    y_pred = final_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    print(f"\nTest accuracy with best model ({classifier_type}): {test_accuracy:.4f}")
    print(f"Test F1 score with best model ({classifier_type}): {test_f1:.4f}")

### GRAPHS #############################################################################################################
    ### Plot learning curve
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

    # Save the plot
    plt.savefig("learning_curve.png", dpi=150, bbox_inches='tight')

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

    # Save the plot
    plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log the AUC metric explicitly
    mlflow.log_metric("test_auc", roc_auc)

    ### Plot feature importance
    # Extract feature importances based on the classifier type
    if classifier_type in ['rf', 'xgb', 'gb']:
        # These classifiers have feature_importances_ attribute
        importances = final_pipeline.named_steps['classifier'].feature_importances_

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

    elif classifier_type == 'svm' and best_params.get('kernel') == 'linear':
        # For linear SVM, we can extract coefficients
        coefficients = np.abs(final_pipeline.named_steps['classifier'].coef_[0])

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
        # For other models where direct feature importance is not available use permutation importance as an alternative
        print("\nCalculating permutation importance for features as an alternative to feature importance.")
        try: # WARNING - currently fails - if SVC does perform highly then rectify
            # Calculate permutation importance
            perm_importance = permutation_importance(
                final_pipeline,
                X_test,
                y_test,
                n_repeats=30,
                random_state=42,
                scoring='f1',
            )

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
            plt.savefig("permutation_importance.png", dpi=300, bbox_inches='tight')

            # Log the figure to MLflow
            mlflow.log_figure(plt.gcf(), "permutation_importance.png")
            plt.close()

            # Also save the full feature importance DataFrame as CSV
            importance_df.to_csv("permutation_importances.csv", index=False)
            mlflow.log_artifact("permutation_importances.csv")

            print(f"\nTop 10 most important features (by permutation importance):")
            print(importance_df.head(10))
        except Exception as e:
            print(f"Unable to calculate feature importance.\n{e}")

# Example output:
# Test accuracy with best model (rf): 0.6000
# Test F1 score with best model (rf): 0.6957


# IMPROVE: Early stopping isn't implemented at all because it would work for some and not others so is more complicated to implement - but could add.
#  Could also do an ensemble model approach for the final training, and stacking/voting

