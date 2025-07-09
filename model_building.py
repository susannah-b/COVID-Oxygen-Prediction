### SCRIPT USAGE #######################################################################################################
# Run this script to train the ML model using the Surrey dataset.
# To use the run_name_config.txt, modify the file to start with the desired ID number and suffix in quotes, or delete to
# generate a fresh file. The suffix string is optional.
#   The run_name for MLflow will be produced in the format N_MMDD-HH:MM_[suffix]. Change the suffix to the desired string
#   to represent the run, e.g. for runs without metadata "no_metadata" may be added.
#   The suffix will be used as a descriptor for the run in the final output subdirectory, so should be descriptive of the
#   settings for the model.

# If track_final is enabled, the logged model and graphs will be copied to the model_output subdirectory for perusal,
# including a file stating the corresponding hyperopt MLflow run name. In the MLflow UI, this hyperopt run will also be added as a
# tag, and vice vera for the hyperopt trials and the final model run name.

######### SETUP ########################################################################################################
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, f_classif, SelectKBest, RFECV, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, to_graphviz
from xgboost import plot_tree as xgb_plot_tree
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from functions import count_meta, basic_train, IntToFloatTransformer, port_in_use, pca_original, plot_learning_curve, \
    plot_roc_auc, plot_feature_importance, plot_calibration_curve, plot_decision_tree, plot_precision_recall, \
    plot_pca_predicted, plot_confusion_matrix
import re
import os
from datetime import datetime
import subprocess
import time
import shutil
import json
import joblib
# todo clean up at end

#IMPROVE Break script into smaller parts

# WARNING - suppress MLflow warning and precision warning - fix later
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Bool to show additional detail
show_detail = False

### READ IN DATA #######################################################################################################
# Set pandas to display all columns and longer rows # IMPROVE remove in final version
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

# Create output directory for the data
data_dir = 'training_data'
os.makedirs(data_dir, exist_ok=True)
# Create graphs directory for the data
graphs_dir = 'training_graphs'
os.makedirs(graphs_dir, exist_ok=True)

### Read in data
# Train
X_path = Path(__file__).parent / data_dir / "Surrey_X_train.csv"
y_path = Path(__file__).parent / data_dir / "Surrey_y_train.csv"
X_train = pd.read_csv(X_path, index_col=0)
y_train = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Test
X_path = Path(__file__).parent / data_dir / "Surrey_X_test.csv"
y_path = Path(__file__).parent / data_dir / "Surrey_y_test.csv"
X_test = pd.read_csv(X_path, index_col=0)
y_test = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# TODO: Note that some isaric columns were selected that might be innaccurate (eg day 1 x ray infiltrates as analogous to Bilateral CXR changes
#  in Surrey data. It would be worth experimenting with dropping some of these here to see if the model improves (although if not feature-selected
#  then it most likely has very minimal impact. Do here to avoid regenerating data, although technically it could affect imputation/scaling/maybe FS.

### OPTIONAL: DROP METADATA ############################################################################################ #TODO once the model is finished and can run on HPC, run with and without dropping to compare results
drop_metadata = False # To experiment with excluding metadata from the model, enable this option

# Read in the original metadata file to read the column info
s_meta_file = Path(__file__).parent / "Surrey_Files" / "Surrey_Metadata_master_spreadsheet_130622_edit2.csv"
s_meta = pd.read_csv(s_meta_file)
# List starting and filtered columns
meta_columns = s_meta.columns.tolist()

meta_cols = 0 # Initialise

# Call function to count metadata columns
meta_cols,X_train = count_meta(X_train, "X_train", meta_columns, drop_metadata, show_detail)
meta_cols,X_test = count_meta(X_test, "X_test", meta_columns, drop_metadata, show_detail) # Colummn # should be identical for test and train, so can just reassign


### PCA ON ORIGINAL DATA ###############################################################################################
try:
    # Combine train and test
    X_full = pd.concat([X_train, X_test]).values
    y_full = np.concatenate([y_train, y_test])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(14, 10))

    # Plot directly from arrays
    plt.scatter(principal_components[y_full == 0, 0],
                principal_components[y_full == 0, 1],
                c='#088BDD', alpha=0.7, label='Does not require O₂')
    plt.scatter(principal_components[y_full == 1, 0],
                principal_components[y_full == 1, 1],
                c='red', alpha=0.7, label='Requires O₂')

    # Add explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title('PCA of Full Dataset - Surrey')
    plt.grid(alpha=0.3)
    plt.legend()

    # Save and show
    plt.savefig(f"{graphs_dir}/pca_all_data.png", dpi=200, bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Error creating PCA biplot on full dataset: {str(e)}")
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

# Determine which models to test
Logistic_regression = False
SVM = False
Random_forest = True
AdaBoost = False
Gradient_boosting = False
XGBoost = True
KNN = False

#  WARNING version with all switched on - delete other after testing:
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
        basic_train(log_reg, X_train, y_train, 'Logistic Regression', top_model_scores, feature_selectors, feature_selection, threshold)

    # SVM
    if SVM:
        svc_clf = SVC()
        basic_train(svc_clf, X_train, y_train, 'Support Vector Classifier', top_model_scores, feature_selectors, feature_selection, threshold)

    # Random Forest
    if Random_forest:
        rnd_clf = RandomForestClassifier(random_state=42)
        basic_train(rnd_clf, X_train, y_train, 'RandomForestClassifier', top_model_scores, feature_selectors, feature_selection, threshold)

    # AdaBoost
    if AdaBoost:
        dt_clf_ada = DecisionTreeClassifier()
        ada_clf = AdaBoostClassifier(estimator=dt_clf_ada, random_state=42)
        basic_train(ada_clf, X_train, y_train, "AdaBoost Classifier", top_model_scores, feature_selectors, feature_selection, threshold)

    # GradientBoosting
    if Gradient_boosting:
        gdb_clf = GradientBoostingClassifier(random_state=42, subsample=0.8)
        basic_train(gdb_clf, X_train, y_train, "GradientBoosting Classifier", top_model_scores, feature_selectors, feature_selection, threshold)

    # XGBoost
    if XGBoost:
        xgb_clf = XGBClassifier(verbosity=0)
        basic_train(xgb_clf, X_train, y_train, "XGBoost Classifier", top_model_scores, feature_selectors, feature_selection, threshold)

    # KNN
    if KNN:
        knn_clf = KNeighborsClassifier()
        basic_train(knn_clf, X_train, y_train, 'K-Nearest Neighbors Classifier', top_model_scores, feature_selectors, feature_selection, threshold)

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
    n_models_to_tune = 2  # The top N models - change this as needed #IMPROVE change back after tests - 2 or 3
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

    # Start MLFlow run for each trial
    with mlflow.start_run(nested=True):
        mlflow.set_tag("Corresponding final model", f"{run_name}")  # Can find the name of the trained model using this tag
        # Log trial hyperparameters
        mlflow.log_params({**params, "type": classifier_type, **{"fs_" + k: v for k, v in fs_params.items()}})

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
        'gamma': hp.uniform('xgb_gamma', 1, 7),
        'reg_alpha': hp.uniform('xgb_reg_alpha', 0, 5),
        'reg_lambda': hp.uniform('xgb_reg_lambda', 0, 5),
        'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.5, 1),
        'min_child_weight': hp.quniform('xgb_min_child_weight', 1, 10, 1),
        'n_estimators': hp.quniform('xgb_n_estimators', 50, 500, 50),
        'seed': 0,
        'learning_rate': hp.uniform('xgb_learning_rate', 0.05, 0.3),
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

### MLFLOW TRACKING ####################################################################################################
# Make folder for tracking runs
os.makedirs('./mlruns', exist_ok=True)

# Start local tracking server
host = "127.0.0.1"
port = 8080

if not port_in_use(host, port):
    print(f"Running tracking server on {host}:{port}")
    subprocess.Popen(["mlflow", "server", "--backend-store-uri", "./mlruns", "--host", host, "--port", f"{port}"])
else:
    print(f"MLflow tracking server already listening on {host}:{port}")

# Pause to allow the server to boot up
    time.sleep(5)

# Set MLFLow tracking URI
mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

### CREATE RUN ID ######################################################################################################
config_path = Path("run_name_config.txt") # Path to file
run_name = None # Initialise
hyperopt_name = None

if not os.path.exists(config_path):
    # If config file doesn't exist, create a 'blank' one
    with open(config_path, 'w') as file:
        file.write('1 ""')
else:
    ### Read in the run ID settings
    with open(config_path, "r") as f:
        content = f.read().strip()
        # Extract numeric prefix and optional quoted suffix
        match = re.match(r"(\d+)\s*(['\"]?)(.*?)\2?$", content) # Digit and optional quoted suffix
        if not match:
            raise ValueError("run_name_config.txt should be formatted like: 1 \"[string]\" or 1. Delete file to generate a fresh config, or modify accordingly.")
        # Set ID info
        run_number = int(match.group(1))
        suffix = match.group(3)
        timestamp = datetime.now().strftime("%m%d-%H%M")
        # Set run name for final model
        run_name = f"{run_number}_{timestamp}"
        if not suffix:
            suffix = "Unspecified" # If no description of the model run has been set in config, then just tag as unspecified
        run_name = f"{run_name}_{suffix}"
        # Set run name for hyperopt
        hyperopt_name = f"{run_number}_hyperopt_{timestamp}_{suffix}"
    # Increase the run_name by one for the next run
    with open(config_path, "w") as f:
        f.write(f"{run_number + 1} \"{suffix}\"")

### HYPEROPT TUNING WITH MLFLOW ########################################################################################
print("\nNow tuning hyperparameters\n")
store_hyp_id = None # Call this later to print the ID at the end
hyper_run_id = None

mlflow.set_experiment("Oxygen Prediction - Hyperparams") # Note: Could use same experiment ID as the final model in order to compare; for now I find it easier to keep them separate.
with mlflow.start_run(run_name=hyperopt_name) as run:
    mlflow.set_tag("Phase", "Hyperopt parameter tuning")
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10, #todo: increase on a better machine # IMPROVE this and other settings/bools could be set using a config file
        trials=Trials()
    )
    # Print run id
    hyper_run_id = run.info.run_id
    store_hyp_id = f"Run {run_name} for hyperparameter training completed. Run ID is {hyper_run_id}. See nested runs for individual trials"

# Print the best accuracies for each model type
print("\nHighest model accuracies on train data:")
best_f1_df = pd.DataFrame(list(best_f1.items()), columns=['Models', 'Highest F1'])
print(best_f1_df)

# Extract and print the best hyperparameter configuration
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

### TRAIN FINAL MODEL ##################################################################################################
# Create a new MLflow Experiment
mlflow.set_experiment("Oxygen Prediction - Surrey")

# Train final model using the full training data
mlflow.sklearn.autolog()
store_final_id = None # Initialise value to store run ID to print at end
final_run_id = None
final_exp_id = None
with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("Run name", run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("Phase", "Final model training")
    mlflow.set_tag("Hyperopt MLflow run", hyperopt_name)
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    final_exp_id = run.info.experiment_id # Get experiment id for folder management
    # Extract the best classifier type
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

    # Save inuput features for validation - JSON (human-readable) and joblib
    with open(f"{data_dir}/input_features.json", "w") as f:
        json.dump(X_train.columns.tolist(), f)
    joblib.dump(X_train.columns.tolist(), f"{data_dir}/input_features.joblib")

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

    # Save selected features for validation - JSON (human-readable) and joblib
    with open(f"{data_dir}/selected_features.json", "w") as f:
        json.dump(selected_features, f)
    joblib.dump(selected_features, f"{data_dir}/selected_features.joblib")

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

    # Save confusion matrix
    plot_confusion_matrix(cm, graphs_dir)

    print(f"\nTest accuracy with best model ({classifier_type}): {test_accuracy:.4f}")
    print(f"Test F1 score with best model ({classifier_type}): {test_f1:.4f}")

### GRAPHS #############################################################################################################
    # Plot PCA on the combined dataset - i.e. original data after feature selection
    with mlflow.start_run(nested=True): #Start another run to avoid auologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run

        # Combine the datasets
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

        # Call function to plot PCA on the dataset prior to feature selection
        pca_original(X_full, selected_features, y_full, graphs_dir)

    # Plot learning curve
    plot_learning_curve(final_pipeline, X_train, y_train, graphs_dir)

    # Plot ROC/AUC curves
    plot_roc_auc(final_pipeline, X_test, y_test, graphs_dir)

    # Plot feature importance
    plot_feature_importance(classifier_type, final_pipeline, selected_features, graphs_dir, data_dir, best_params,
                            X_test, y_test)

    # Plot calibration curve - TODO untested
    plot_calibration_curve(final_pipeline, X_test, y_test, classifier_type, graphs_dir)

    # Plot decision tree
    class_names = np.array(['No_Oxygen_Need', 'Oxygen_Need'])
    plot_decision_tree(classifier_type, final_pipeline, X_train, class_names, data_dir, graphs_dir)

    # Plot a precision-recall curve
    plot_precision_recall(final_pipeline, X_test, y_test, graphs_dir)

    ### Plot PCA on final predictions - Test data
    with mlflow.start_run(nested=True):  # Start another run to avoid autologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run
        # Plot PCA
        plot_pca_predicted(X_test, selected_features, y_test, graphs_dir, y_pred)

    # Print run id
    final_run_id = run.info.run_id
    store_final_id = f"Run {run_name} for final model completed. Run ID is {final_run_id}"

    # Log artifacts
    mlflow.log_artifacts(graphs_dir, artifact_path="graphs")
    mlflow.log_artifacts(data_dir, artifact_path="tables")

### STORE RESULTS IN NEW FOLDER ########################################################################################
# Move and rename runs to a new directory for easier examination - results are copied from the MLflow tracking folder
# (which is also available in the server) but renamed here for easier access based on the suffix defined in the config
# file.
# Bool to set whether to copy the runs to the final output subdirectory - for testing only this can be disabled
track_final = True #IMPROVE: take out useful individual subfolders vs whole folder contents - need to determine which bits are useful
if track_final:
    print("\'track_final\' has been enabled, so the model information will be copied to ./model_output for easier viewing.")

    # Determine file locations
    final_folder = Path("mlruns") / final_exp_id / final_run_id
    ml_artifacts = Path("mlartifacts") / final_exp_id / final_run_id
    output_folder = Path("model_output") / run_name
    output_artifacts = output_folder
    data_folder = Path(data_dir)
    graph_folder = Path(graphs_dir)

    # Copy final model folder contents
    shutil.copytree(final_folder, output_folder, dirs_exist_ok=True)
    # Copy final model artifacts from mlartifacts to the model_output model file
    #  Note: since setting an experiment name changes the artifacts location to mlartifacts instead of in the mlruns (run) folder, we will copy it over for our final output
    shutil.copytree(ml_artifacts, output_artifacts, dirs_exist_ok=True)
    # Copy training data and graphs folder
    shutil.copytree(data_folder, output_folder / data_dir, dirs_exist_ok=True)
    shutil.copytree(graph_folder, output_folder / graphs_dir, dirs_exist_ok=True)

    # Make note of the corresponding hyperopt MLflow run
    hyper_run_file = final_folder / "hyperopt_run_name.txt"
    hyper_run_file.write_text(f"{hyperopt_name}")

# Print run ids
print(store_hyp_id)
print(store_final_id)

# WARNING: If getting the 'too many 500 error responses' warning due to deleting files, run 'kill $(lsof -t -i tcp:8080)' in the terminal


# Example output:
# Test accuracy with best model (rf): 0.6000
# Test F1 score with best model (rf): 0.6957


# IMPROVE: Early stopping isn't implemented at all because it would work for some and not others so is more complicated to implement - but could add.
#  Could also do an ensemble model approach for the final training, and stacking/voting

# IMPROVE once final model is obtained, I'll likely want to plot more model-specific graphs

# TODO early stopping?

#todo extrapolate graphs into functions or class and apply to model building and validation