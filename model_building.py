### SCRIPT USAGE #######################################################################################################
# Run this script to train the ML model using the Surrey dataset.

# If track_final is enabled, the logged model and graphs will be copied to the model_output subdirectory for perusal,
# including a file stating the corresponding hyperopt MLflow run name. In the MLflow UI, this hyperopt run will also be added as a
# tag, and vice vera for the hyperopt trials and the final model run name.

######### SETUP ########################################################################################################
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, \
    classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFECV, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow
import yaml
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from functions import basic_train, IntToFloatTransformer, port_in_use, pca_pre_post_fs, plot_learning_curve, \
    plot_roc_auc, plot_feature_importance, plot_calibration_curve, plot_decision_tree, plot_precision_recall, \
    plot_pca_predicted, plot_confusion_matrix, plot_fs_performance, plot_decision_distribution, plot_pca_original, \
    plot_pca_test_unprocessed, remaining_meta, set_graph_style
import os
from datetime import datetime
import subprocess
import time
import shutil
import json
import joblib
import argparse
from mlflow.tracking import MlflowClient

# todo clean up at end

#IMPROVE Break script into smaller parts

# WARNING - suppress MLflow warning and precision warning - fix later
# warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")
# from sklearn.exceptions import UndefinedMetricWarning
# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Bool to show additional detail
show_detail = False

# Set graph style
set_graph_style()

### ARGPARSE TO SET RUN NAME ###########################################################################################
  # If running as part of pipeline.py, get the run_name from the stored config file not the config in cwd (avoids issues with multiple script runs)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Run the script with a predetermined run name; only necessary for use in the pipeline.py script. If "
         "running the script standalone the --run_name parameter is not used."
)
parser.add_argument(
    "--from_pipeline",
    action="store_true",
    help="Indicates the script is being called from pipeline.py"
)
args = parser.parse_args()
run_name = args.run_name

print(f"Starting run {run_name}\n")
#### READ CONFIG FILE ##################################################################################################
# Set config path based on whether the script is run standlone or part of pipeline.py (config moved to 'inputs')
if not args.from_pipeline:
    config_path = Path("config.yaml")
else:
    config_path = Path(f"inputs/ML/{run_name}/config.yaml") #TODO - might restructure wrapper to remove this now mb outputs are separated (and for NN)

# Read config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set parameters for this file:
validate = config['general']['validate'] # Whether to make the Surrey dataset validation compatible
var_threshold = config['model_building']['var_threshold'] # Threshold for variance filtering
feature_selection = config['model_building']['feature_selection'] # Whether to do feature selection at all stages
basic_training = config['model_building']['basic_training'] # Whether to run basic_training (vs load in specific model type/feature selector)
n_models_to_tune = config['model_building']['n_models_to_tune'] # How many model types to take to the hyperparamter tuning stage
host = config['general']['host'] # Host for tracking server
port = config['general']['port'] # Port for local tracking server
model_choice = config['model_building']['specify_model']['model_type'] # Model type if not basic training # TODO update with best once determined
fs_choice = config['model_building']['specify_model']['fs'] #Feature selector if not basic training # TODO update with best once determined
max_evals = config['model_building']['max_evals'] # How many evaluations to do in hyperopt tuning
track_final = config['model_building']['track_final'] # Whether to copy the model_output to the designated folder for easier browsing
meta_cols = config["general"]["training_meta_cols"]

# Determine which models to test (set in config file)
Logistic_regression = config['model_building']['models_to_test']['Logistic_regression']
SVM = config['model_building']['models_to_test']['SVM']
Random_forest = config['model_building']['models_to_test']['Random_forest']
AdaBoost = config['model_building']['models_to_test']['AdaBoost']
Gradient_boosting = config['model_building']['models_to_test']['Gradient_boosting']
XGBoost = config['model_building']['models_to_test']['XGBoost']
KNN = config['model_building']['models_to_test']['KNN']
enable_tracking = config['general']['enable_tracking']

### CREATE RUN NAME ####################################################################################################
if not args.from_pipeline:
     # Set run name - when run as part of pipeline.py instead, this is defined as an argument. Here it needs to be set so the model_output folder can be made, etc.
     timestamp = datetime.now().strftime("%m%d-%H%M%S")
     run_number = config["general"]["run_number"]
     run_suffix = config["general"]["run_suffix"] or "Unspecified"  # Set to unspecified if empty
     run_name = f"{run_number}_{timestamp}_{run_suffix}"  # Unique ID for each model built
     # Set ID for hyperopt
     hyperopt_name = f"{run_number}_hyperopt_{timestamp}_{run_suffix}"
     # Increase the run_name by one for the next run
     with open(config_path, "w") as f:
         config["general"]["run_number"] = run_number + 1
         yaml.dump(config, f, sort_keys=False)
     # WARNING: Run name is not automatically imported to external_validation.py if running standalone to allow specific runs to be used. Set manually.
else:
    rn_components = run_name.split("_")
    rn1 = rn_components[0]
    rn2 = rn_components[1]
    rn3 = '_'.join(rn_components[2:])
    hyperopt_name = f"{rn1}_hyperopt_{rn2}_{rn3}"

### READ IN DATA #######################################################################################################
# Set pandas to display all columns and longer rows # IMPROVE remove in final version
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

# Create input directories for the data
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    data_dir = 'training_data' # Combine with other training graphs if using training data
else: # Put into input storage folder to prevent overwriting
    data_dir = f'inputs/ML/{run_name}/training_data'
    # IMPROVE - if scripts are run standalone, old files in training_data are copied. As mentioned elsewhere, it would be better to specify which exact files are produced to prevent this

# Create output directories for the data
output_data_dir = f'{data_dir}/ML'
os.makedirs(output_data_dir, exist_ok=True)

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    graphs_dir = 'training_graphs/ML' # Combine with other training graphs if using training data
else: # Put into input storage folder to prevent overwriting
    graphs_dir = f'inputs/ML/{run_name}/training_graphs/ML'
os.makedirs(graphs_dir, exist_ok=True) # Make the ML graph that's specific to the ML outputs

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

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Feature dimensions: {X_train.shape[1]} | Classes: {y_train.nunique()}\n")

# TODO: Note that some isaric columns were selected that might be innaccurate (eg day 1 x ray infiltrates as analogous to Bilateral CXR changes
#  in Surrey data. It would be worth experimenting with dropping some of these here to see if the model improves (although if not feature-selected
#  then it most likely has very minimal impact. Do here to avoid regenerating data, although technically it could affect imputation/scaling/maybe FS.

### PCA ON ORIGINAL DATA ###############################################################################################
# Full dataset
plot_pca_original(X_train, X_test, y_train, y_test, graphs_dir)
# Test only (to compare to the post-processed before and after predictions graphs)
plot_pca_test_unprocessed(X_test, y_test, graphs_dir)

### VARIANCE THRESHOLDING ##############################################################################################
  # Applied in the scikit-learn pipeline
 # IMPROVE currently set to 0 variance (which removes none of my data) and got a better result - can experiment with other values later. Since sample # is low it may be best with minimal filtering
# Calculate median variance of all features
variances = X_train.var(axis=0)
#threshold = np.median(variances) # Example of thresholds - either delete or experiment with
#threshold = np.quantile(variances, 0.75)
threshold = float(var_threshold) # Effectively zero but avoids floating-point issues

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
### Feature selection methods taken from scikit-learn documentation # IMPROVE - methods were chosen to cover a wide range of approaches/models but could be tweaked further
# Dictionary of feature selector options. base_params are fixed parameters that also apply to basic_train, with other parameters tunable later in the search space
feature_selectors_all = {
    # RFECV with Logistic Regression
    'RFECV_LR': {
        'class': RFECV,
        'base_params': {
            'estimator': LogisticRegression(),
            'step': 1,
            'cv': StratifiedKFold(5),
            'scoring': "roc_auc",
            'min_features_to_select': 50,
        }
    },
    # RFECV with Support Vector Classifier
    'RFECV_SVC': {
        'class': RFECV,
        'base_params': {
            'estimator': SVC(kernel='linear'),
            'step': 1,
            'cv': StratifiedKFold(5),
            'scoring': "roc_auc",
            'min_features_to_select': 50,
        }
    },
    # RFECV with Random Forest #TODO untested - takes very long so either run on HPC or with more time
    'RFECV_RF': {
        'class': RFECV,
        'base_params': {
            'estimator': RandomForestClassifier(random_state=42),
            'step': 1,
            'cv': StratifiedKFold(5),
            'scoring': "roc_auc",
            'min_features_to_select': 50,
        }
    },
    # RFECV with XGBoost #TODO untested - takes very long so either run on HPC or with more time
    'RFECV_XGB': {
        'class': RFECV,
        'base_params': {
            'estimator': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='logloss', random_state=42),
            'step': 1,
            'cv': StratifiedKFold(5),
            'scoring': "roc_auc",
            'min_features_to_select': 50,
        }
    },
    # SelectFromModel with Logistic Regression
    'SFM_LR': {
        'class': SelectFromModel,
        'base_params': {
            'estimator': LogisticRegression()
        }
    },
    # SelectFromModel with Support Vector Classifier
    'SFM_SVC': {
        'class': SelectFromModel,
        'base_params': {
            'estimator': SVC(kernel='linear')
        }
    },
    # SelectFromModel with Random Forest
    'SFM_RF': {
        'class': SelectFromModel,
        'base_params': {
            'estimator': RandomForestClassifier(random_state=42),
            'threshold': "median"
        }
    },
    # SelectFromModel with XGBoost
    'SFM_XGB': {
        'class': SelectFromModel,
        'base_params': {
            'estimator': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='logloss', random_state=42)
        }
    },
    # SelectFromModel with Lasso
    'SFM_LAS': {
        'class': SelectFromModel,
        'base_params': {
            'estimator': Lasso(alpha=0.05, max_iter=10000, random_state=42)
        }
    },
    # Sequential Feature Selection with Logistic Regression
    'SFS_LR': {
        'class': SequentialFeatureSelector,
        'base_params': {
            'estimator': LogisticRegression(),
            'n_features_to_select': 'auto',
            'tol': 0.01,
        }
    },
    # Sequential Feature Selection with Linear SVC
    'SFS_LSVC': {
        'class': SequentialFeatureSelector,
        'base_params': {
            'estimator': LinearSVC(dual=True, max_iter=3000),
            'n_features_to_select': 'auto',
            'tol': 0.01,
        }
    },
    # Sequential Feature Selection with XGBoost #TODO untested - takes very long so either run on HPC or with more time
    'SFS_XGB': {
        'class': SequentialFeatureSelector,
        'base_params': {
            'estimator': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='logloss', random_state=42),
            'n_features_to_select': 'auto',
            'tol': 0.01,
        }
    },
    # No feature selection #TODO untested
    'NONE': {
        'class': None,
        'base_params': {}
        }
    }

candidate_fs = ['RFECV_LR', 'RFECV_SVC', 'RFECV_RF', 'RFECV_XGB', 'SFM_LR', 'SFM_SVC', 'SFM_RF', 'SFM_XGB', 'SFM_LAS',
                'SFS_LR', 'SFS_LSVC', 'SFS_XGB', 'NONE']

# If enabled in config, add to the feature_selectors dictionary for use in basic_train
feature_selectors = {}
for fs_name in candidate_fs:
    if config['model_building']['feature_selectors'].get(fs_name):  # Check if enabled
        feature_selectors[fs_name] = feature_selectors_all[fs_name]

### ESTIMATE BEST MODELS WITH BASIC SETTINGS ###########################################################################
# Initialise dict to store best results per model
top_model_scores = {}

# Dictionary to store the highest performing models and their feature selection methods
best_models_fs = {}

# List to store results from each model basic training
basic_results = []
# Outline each model and perform the basic training function to evaluate performance of each
if basic_training:
    "Beginning basic_training function to determine the best model type and feature selector."
    # Logistic Regression
    if Logistic_regression:
        log_reg = LogisticRegression(solver='saga', tol=1e-4, max_iter=1500)
        lr_results = basic_train(log_reg, X_train, y_train, 'Logistic Regression', top_model_scores, feature_selectors, feature_selection, threshold)
        basic_results.append(lr_results)

    # SVM
    if SVM:
        svc_clf = SVC(probability=True)
        svm_results = basic_train(svc_clf, X_train, y_train, 'Support Vector Classifier', top_model_scores, feature_selectors, feature_selection, threshold)
        basic_results.append(svm_results)

    # Random Forest
    if Random_forest:
        rnd_clf = RandomForestClassifier(random_state=42)
        rf_results = basic_train(rnd_clf, X_train, y_train, 'RandomForestClassifier', top_model_scores, feature_selectors, feature_selection, threshold)
        basic_results.append(rf_results)

    # AdaBoost
    if AdaBoost:
        dt_clf_ada = DecisionTreeClassifier()
        ada_clf = AdaBoostClassifier(estimator=dt_clf_ada, random_state=42, algorithm='SAMME')
        ada_results = basic_train(ada_clf, X_train, y_train, "AdaBoost Classifier", top_model_scores, feature_selectors, feature_selection, threshold)
        basic_results.append(ada_results)

    # GradientBoosting
    if Gradient_boosting:
        gdb_clf = GradientBoostingClassifier(random_state=42, subsample=0.8)
        gb_results = basic_train(gdb_clf, X_train, y_train, "GradientBoosting Classifier", top_model_scores, feature_selectors, feature_selection, threshold)
        basic_results.append(gb_results)

    # XGBoost
    if XGBoost:
        xgb_clf = XGBClassifier(verbosity=0)
        xgb_results = basic_train(xgb_clf, X_train, y_train, "XGBoost Classifier", top_model_scores, feature_selectors, feature_selection, threshold)
        basic_results.append(xgb_results)

    # KNN
    if KNN:
        knn_clf = KNeighborsClassifier()
        knn_results = basic_train(knn_clf, X_train, y_train, 'K-Nearest Neighbors Classifier', top_model_scores, feature_selectors, feature_selection, threshold)
        basic_results.append(knn_results)

    # Make dataframe of model scores and print results
    scores = pd.DataFrame.from_dict(top_model_scores,
                                    orient='index',
                                    columns=['Model', 'Selector', 'Train Accuracy', 'CV Accuracy', 'Train AUROC',
                                             'Test AUROC']).reset_index(drop=True).sort_values(by='Test AUROC', ascending=False)
    # Print the top results for each model
    print("\nBest results per model:")
    print(scores.head(len(scores)))

    ### Make a summary of feature selection method performance
    all_results_df = pd.concat(basic_results, ignore_index=True) # Combine results
    all_results_sorted = all_results_df.sort_values(by=['Selector', 'Model']).reset_index(drop=True)
    plot_fs_performance(all_results_sorted, graphs_dir) # Plot

    ### Determine the best performing models to take to the tuning phase
    # Check the maximum number of models that are switched on
    model_flags = [Logistic_regression, SVM, Random_forest, AdaBoost, Gradient_boosting, XGBoost, KNN]
    n_models_safe = min(sum(model_flags), n_models_to_tune)
    # Extract best model and feature selection from top_model_scores
    for i in range(0, n_models_safe): # todo think this errors if more than the number of models - should pick the minimum of those values
        model = scores.iloc[i, 0]
        fs = scores.iloc[i, 1]
        best_models_fs[model] = fs
    # Print results
    hypertune = pd.DataFrame(best_models_fs.items(), columns=['Model', 'Selector'])
    print("\nThe models taken to the hyperparameter tuning stage are:\n", hypertune)
else:
    # Set model and feature selector that perform the best - do this manually by adding entries below based on the results of prior basic_training runs
    best_models_fs[model_choice] = fs_choice
    # IMPROVE could add option to specify more by doing the config param as a dict instead
    hypertune = pd.DataFrame(best_models_fs.items(), columns=['Model', 'Selector'])
    print("\nThe models taken to the hyperparameter tuning stage are:\n", hypertune)

### OBJECTIVE FUNCTION FOR HYPEROPT PARAMETER TUNING ###################################################################
# For selected models, define a parameter params['type'] for the model name. Then evaluate parameters and calculate the cross-validated accuracy.

# Dictionary to store the best model accuracies
best_roc = {
    'svm': 0.0,
    'rf': 0.0,
    'logreg': 0.0,
    'xgb': 0.0,
    'gb': 0.0,
    'ada': 0.0,
    'knn': 0.0
}

# Conversion to longer format # Note: to change this you need to change the format of the top model scores in basic train (i.e the model cell)
# Conversion to longer format # Note: to change this you need to change the format of the top model scores in basic train (i.e the model cell)
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
    del params['type']  # Remove from the search space

    # Set feature selector and parameters based on classifier type (as defined by basic_train)
    fs_params = params.pop('fs_params', {}) # Remove FS params from classifier search space

    # Get selector configuration
    selector_type = best_models_fs[type_translation[classifier_type]]
    selector_config = feature_selectors[selector_type]

    ### Build the feature selector
    if selector_type == 'NONE':
        selector = 'passthrough'
    else:
        # Merge base parameters with tuned parameters - the base_params are overwritten by the hyperopt fs_params
        all_params = {**selector_config['base_params'], **fs_params}

        ### Convert feature selection parameters to integers where required
            # Note: same conversion is done for different selector types
        if 'min_features_to_select' in all_params:
            all_params['min_features_to_select'] = int(all_params['min_features_to_select'])
        if 'n_features_to_select' in all_params:
            all_params['n_features_to_select'] = int(all_params['n_features_to_select'])

        # Update params
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
        clf = AdaBoostClassifier(**best_params)
        # clf = AdaBoostClassifier(**params)
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
        pipe = Pipeline([ #todo was this meant to be deleted? immediately reassigned below
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

        # Use 5-fold cross validation to compute the mean AUROC
        roc_score_mean = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='roc_auc').mean()  # Reduced to 5-fold for speed

        # Log the best accuracy for each model type if improved
        if roc_score_mean > best_roc[classifier_type]:
            best_roc[classifier_type] = roc_score_mean
            mlflow.log_metric(f"best_{classifier_type}_AUROC", roc_score_mean)

    # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
    return {'loss': -roc_score_mean, 'status': STATUS_OK}

### DEFINE SEARCH SPACES PER FEATURE SELECTOR ########################################################################## # TODO - go over documentation and check which options to include for each parameter, and decide whether to go in base params or the search space - AI-gened for now
selector_param_spaces = { # Note: for new data, values may need to be tweaked as in feature selection parameter tuning, some fits can fail and crash the script
    'SFM_RF': {
        'threshold': hp.choice('sfm_rf_threshold', [None, 'median', 'mean', 1e-6, 1e-5, 1e-4])
    },
    'RFECV_SVC': {
        'step': hp.uniform('rfecv_step', 0.01, 0.3),
        'min_features_to_select': hp.quniform('rfecv_min_feat', 5, 30, 1)
    },
    'SFM_XGB': {
        'threshold': hp.choice('xgb_threshold', [None, 'median', 'mean', 1e-6, 1e-5, 1e-4])
    },
    'RFECV_LR': {
        'step': hp.choice('rfecv_lr_step', [0.01, 0.1, 1]),
        'cv' : hp.choice('refcv_cv', [StratifiedKFold(5), StratifiedKFold(10)]),
        'scoring' : hp.choice('refcv_scoring', ['f1', 'accuracy', 'r2', 'roc_auc']),

    },
    'RFECV_RF': {
        'step': hp.choice('rfecv_lr_step', [0.01, 0.1, 1]),
        'cv' : hp.choice('refcv_cv', [StratifiedKFold(5), StratifiedKFold(10)]),
        'scoring' : hp.choice('refcv_scoring', ['f1', 'accuracy', 'roc_auc']),
    },
    'RFECV_XGB': {
        'step': hp.choice('rfecv_lr_step', [0.01, 0.1, 1]),
        'cv' : hp.choice('refcv_cv', [StratifiedKFold(5), StratifiedKFold(10)]),
        'scoring' : hp.choice('refcv_scoring', ['f1', 'accuracy'
            , 'roc_auc']),
    },
    'SFM_LR': {
        'threshold': hp.choice('sfm_lr_threshold', [None, 'median', 'mean', 1e-6, 1e-5, 1e-4])
    },
    'SFM_SVC': {
        'threshold': hp.choice('sfm_svc_threshold', [None, 'median', 'mean', 1e-6, 1e-5, 1e-4])
    },
    'SFM_LAS': {
        'threshold': hp.choice('sfm_las_threshold', [None, 'median', 'mean', 1e-6, 1e-5, 1e-4])
    },
    'SFS_LR': {
        'n_features_to_select': hp.quniform('sfs_lr_n_features', 5, min(50, X_train.shape[1]), 1),
        'tol': hp.uniform('sfs_tol', 1e-3, 0.01),
        'direction': hp.choice('sfs_direction', ['forward', 'backward']),
        'scoring' : hp.choice('refcv_scoring', ['f1', 'accuracy', 'roc_auc' 'r2']),
        'cv' : hp.choice('refcv_cv', [StratifiedKFold(5), StratifiedKFold(10)])
    },
    'SFS_LSVC': {
        'n_features_to_select': hp.quniform('sfs_lsvc_n_features', 5, min(50, X_train.shape[1]), 1),
        'tol': hp.uniform('sfs_tol', 1e-3, 0.01),
        'direction': hp.choice('sfs_direction', ['forward', 'backward']),
        'scoring' : hp.choice('refcv_scoring', ['f1', 'accuracy', 'roc_auc', 'r2']),
        'cv' : hp.choice('refcv_cv', [StratifiedKFold(5), StratifiedKFold(10)])
    },
    'SFS_XGB': {
        'n_features_to_select': hp.quniform('sfs_xgb_n_features', 5, min(50, X_train.shape[1]), 1),
        'tol': hp.uniform('sfs_tol', 1e-3, 0.01),
        'direction': hp.choice('sfs_direction', ['forward', 'backward']),
        'scoring' : hp.choice('refcv_scoring', ['f1', 'accuracy', 'roc_auc']),
        'cv' : hp.choice('refcv_cv', [StratifiedKFold(5), StratifiedKFold(10)])
    },
    'NONE': {}
}

### DEFINE SEARCH SPACES PER MODEL #####################################################################################
  # Define each search space per model type. If in the top performing models (determined in basic train/manually), add to the overall search space

best_spaces = [] # Initialise list
### Define space for each model  - input into objective as 'params'
# SVM
if type_translation['svm'] in best_models_fs:
    best_spaces.append({
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1.0),
        'kernel': hp.choice('svm_kernel', ['linear', 'rbf']),
        'gamma': hp.choice('svm_gamma', ['scale', 'auto']),
        'class_weight': hp.choice('svm_class_weight', [None, 'balanced']),
        'random_state': 42,
        'probability' : True,
        'fs_params': selector_param_spaces[best_models_fs[type_translation['svm']]]
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
        'fs_params': selector_param_spaces[best_models_fs[type_translation['rf']]] # The FS shorthand name, e.g. SFM_RF # WARNING - reviewing code and not sure if only rf models are having fs_params tuned explored? why do the search spaces not ahve this
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
       'max_iter': 3000,
       'fs_params': selector_param_spaces[best_models_fs[type_translation['logreg']]]
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
        'fs_params': selector_param_spaces[best_models_fs[type_translation['xgb']]]
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
        'fs_params': selector_param_spaces[best_models_fs[type_translation['gb']]]
    })
# AdaBoost
if type_translation['ada'] in best_models_fs:
    best_spaces.append({
        'type': 'ada',
        'n_estimators': hp.uniform('ada_n_estimators', 30, 1000),
        'learning_rate': hp.uniform('ada_learning_rate', 0.1, 1.0),
        'estimator': hp.choice('ada_base_estimator', [
            DecisionTreeClassifier(random_state=42),
            LinearSVC(random_state=42, dual=True, max_iter=3000), # WARNING: LinearSVC is untested
            LogisticRegression(random_state=42),
        ]),
        'random_state': 42,
        'algorithm': 'SAMME', # IMPROVE: This is added to prevent warnings on the HPC, which is 1.4.2 vs local 1.6.1 scikit-learn. Ideally update the container and then handle any new warnings, but for now just ensuring the HPC version runs smoothly. (Also true for other warnings)
        'fs_params': selector_param_spaces[best_models_fs[type_translation['ada']]]
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
        'fs_params': selector_param_spaces[best_models_fs[type_translation['knn']]]
    })


# Define the search space over hyperparameters (for classifier only; feature selection is determined elsehwere)
search_space = hp.choice('classifier_type', best_spaces)

### MLFLOW TRACKING ####################################################################################################
# Make folder for tracking runs
os.makedirs('./mlruns', exist_ok=True)

if enable_tracking:
    if not port_in_use(host, port):
        print(f"Running tracking server on {host}:{port}")
        subprocess.Popen(["mlflow", "server", "--backend-store-uri", "./mlruns", "--host", host, "--port", f"{port}"])
    else:
        print(f"MLflow tracking server already listening on {host}:{port}")

    # Pause to allow the server to boot up
        time.sleep(5)

# Set MLFLow tracking URI
if enable_tracking:
    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

### HYPEROPT TUNING WITH MLFLOW ########################################################################################
print("\nNow tuning hyperparameters...\n")
mlflow.set_experiment("Oxygen Prediction - Hyperparams") # Note: Could use same experiment ID as the final model in order to compare; for now I find it easier to keep them separate.
with mlflow.start_run(run_name=hyperopt_name) as run:
    mlflow.set_tag("Phase", "Hyperopt parameter tuning")
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials()
    )
    # Print run id
    hyper_run_id = run.info.run_id
    store_hyp_id = f"Run {run_name} for hyperparameter training completed. Run ID is {hyper_run_id}. See nested runs for individual trials" #TODO should this be hyperopt_run_name not run_name

# Print the best accuracies for each model type
print("\nHighest model AUROC on train data:")
best_roc_df = pd.DataFrame(list(best_roc.items()), columns=['Models', 'Highest AUROC'])
print(best_roc_df)

# Extract and print the best hyperparameter configuration
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

### TRAIN FINAL MODEL ##################################################################################################
# Create a new MLflow Experiment
if enable_tracking: # Have to use a unique name or it creates issues with artifact tracking
    exp_name = "Oxygen Prediction Traditional ML - Surrey"
else:
    exp_name = "Oxygen Prediction Traditional ML - Surrey - Offline"

artifact_path = f"mlartifacts"
os.makedirs(artifact_path, exist_ok=True)
client = MlflowClient()
existing_experiment = client.get_experiment_by_name(exp_name)

# Create new experiment if it doesn't exist
if existing_experiment is None:
    print(f"Creating new experiment for {exp_name}")
    client.create_experiment(name=exp_name, artifact_location=artifact_path)
else:
    print(f"Using existing experiment for {exp_name}")

mlflow.set_experiment(exp_name)


# Train final model using the full training data
mlflow.sklearn.autolog()
store_final_id = None # Initialise value to store run ID to print at end
final_run_id = None
with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("Run name", run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("ML type", "Traditional ML")
    mlflow.set_tag("Phase", "Final model training")
    mlflow.set_tag("Hyperopt MLflow run", hyperopt_name)
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    final_exp_id = run.info.experiment_id # Get experiment id for folder management

    # Extract the best classifier type
    classifier_type = best_config['type']

    # Get parameters for the classifier and feature selector
    fs_params = best_config.get('fs_params', {})  # Feature selector params
    # Convert to integers where needed
    if 'min_features_to_select' in fs_params:
        fs_params['min_features_to_select'] = int(fs_params['min_features_to_select'])
    if 'n_features_to_select' in fs_params:
        fs_params['n_features_to_select'] = int(fs_params['n_features_to_select'])
    # Assign best parameters for the model
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

    # Save input features for validation - JSON (human-readable) and joblib
    with open(f"{output_data_dir}/input_features.json", "w") as f:
        json.dump(X_train.columns.tolist(), f)
    joblib.dump(X_train.columns.tolist(), f"{output_data_dir}/input_features.joblib")

    # Train on full training data
    final_pipeline.fit(X_train, y_train)

    # Track retained features post-preprocessing
    preprocessor = final_pipeline.named_steps['preprocessor']
    var_thresh = preprocessor.named_steps['var_thresh']
    retained_mask = var_thresh.get_support()
    retained_features = X_train.columns[retained_mask]
    print("Features after thresholding:", len(retained_features.tolist()))
    show_features = False # Enable or disable as required
    if show_detail:
        if show_features:
            print(retained_features.tolist())
        else:
            print("show_features is disabled in model_building.py. To view features as a list, enable this bool.")

    # Print the selected features post-feature selection method # WARNING Not tested with all methods
    try:
        selector = final_pipeline.named_steps['feature_selector']
        if selector == 'passthrough':
            selected_features = retained_features.tolist()
        else:
            if hasattr(selector, 'get_support'): # Standard scikit-learn selector
                support_mask = selector.get_support()
                selected_features = retained_features[support_mask].tolist()
            elif hasattr(selector, 'support_'): # Other selector types
                support_mask = selector.support_
                selected_features = retained_features[support_mask].tolist()
            else: # For other selector types, get features via transformation
                print("Feature selection method is incompatible with current handling to extract features - results are not printed.")
                selected_features = retained_features.tolist()  # Set selected features to full X_train if not assigned by a feature selector
            # Print features
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features)
    except Exception as e:
        print(f"Unable to print features for this feature selection method: {str(e)}")

    # Save selected features for validation - JSON (human-readable) and joblib
    with open(f"{output_data_dir}/selected_features.json", "w") as f:
        json.dump(selected_features, f)
    joblib.dump(selected_features, f"{output_data_dir}/selected_features.joblib")
    ### Log the final pipeline model
    # Create input example
    input_example = X_train.iloc[:1]
    # Infer model signature
    signature = infer_signature(X_train, final_pipeline.predict(X_train))
    model_info = mlflow.sklearn.log_model(final_pipeline, "best_model", signature=signature, input_example=input_example)
    model_id = model_info.model_uuid # Get model ID to copy over if in HPC

    # Evaluate the final model on the test set
    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix
    plot_confusion_matrix(cm, graphs_dir)

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) # aka TPR, recall
    specificity = tn / (tn + fp) # aka TNR
    precision = tp / (tp + fp) # aka PPV
    npv = tn / (tn + fn)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["O2 not required", "O2 required"]))

    # Print and log metrics
    print(f"\nTest accuracy with best model ({classifier_type}): {test_accuracy:.4f}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Test F1 score with best model ({classifier_type}): {test_f1:.4f}")
    mlflow.log_metric("test_roc", test_roc)
    print(f"Test AUROC score with best model ({classifier_type}): {test_roc:.4f}")
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("sensitivity-tpr-recall", sensitivity)
    mlflow.log_metric("specificity-tnr", specificity)
    mlflow.log_metric("precision-ppv", precision)
    mlflow.log_metric("npv", npv)

    # Save predictions to csv
    model_results = pd.DataFrame({'Real values': y_test,'ML predictions': y_pred}, index=X_test.index)
    model_results.to_csv(f"{output_data_dir}/Prediction_results_test_data.csv")


### GRAPHS #############################################################################################################
    # Plot PCA on the combined dataset - i.e. all data after feature selection
    with mlflow.start_run(nested=True): # Start another run to avoid auologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run

        # Combine the datasets
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

        # Call function to plot PCA on the dataset post feature selection
        pca_pre_post_fs(X_full, selected_features, y_full, graphs_dir, "After")

    # Plot learning curve
    plot_learning_curve(final_pipeline, X_train, y_train, graphs_dir)

    # Plot ROC/AUC curves
    plot_roc_auc(y_proba, y_test, graphs_dir)

    # Plot feature importance
    meta_col_names = X_test.columns[0:meta_cols].tolist()
    if meta_cols != 0:
        meta_cols = remaining_meta(meta_col_names, X_test[selected_features], sample_inves_7=None, graphs_dir=graphs_dir) # Calculate meta columns remaining in order to group features
    try: #todo putting this in a try clause for testing temporarily - this erorred in HPC but might have been fixed in an earlier run
        plot_feature_importance(classifier_type, final_pipeline, selected_features, graphs_dir, output_data_dir, best_params,
                            X_test, y_test, meta_cols)
    except:
        print("********************WARNING********************")
        print("********************WARNING********************")
        print("********************WARNING********************")
        print("********************WARNING********************")
        print("plot feature importance failed. diagnose the issue.") # todo delete after testing


    # Plot calibration curve
    plot_calibration_curve(y_proba, y_test, type_translation[classifier_type], graphs_dir)

    # Plot decision distribution
    plot_decision_distribution(y_proba, y_test, graphs_dir)

    # Plot decision tree
    class_names = np.array(['No_Oxygen_Need', 'Oxygen_Need'])
    plot_decision_tree(classifier_type, final_pipeline, retained_features, class_names, output_data_dir, graphs_dir)

    # Plot a precision-recall curve
    plot_precision_recall(y_proba, y_test, graphs_dir)

    ### Plot PCA on final predictions - Test data before and after prediction
    with mlflow.start_run(nested=True):  # Start another run to avoid autologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run
        # Plot PCA
        plot_pca_predicted(X_test, selected_features, y_test, graphs_dir, y_pred)

    # Print run id
    final_run_id = run.info.run_id
    store_final_id = f"Run {run_name} for the final traditional machine learning model completed. Run ID is {final_run_id}"

    # Log artifacts
    if enable_tracking:
        mlflow.log_artifacts(graphs_dir, artifact_path="graphs")
        mlflow.log_artifacts(output_data_dir, artifact_path="tables")
    #todo also log metric test accuracy, f1, anything else I generate

    ### SAVE DATA FOR ADDITIONAL GRAPHS ################################################################################
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index) #todo check index is correct
    y_proba_df = pd.DataFrame(y_proba, index=y_test.index)
    y_results = pd.concat([y_pred_df, y_proba_df], axis=0)
    y_results.to_csv(f"{output_data_dir}/y_results.csv")

### STORE RESULTS IN NEW FOLDER ########################################################################################
# todo: if running as a standalone script (not in wrapper) then it will copy whatever old files + NN files too. Ideally only select current run files + ML
# Move and rename runs to a new directory for easier examination - results are copied from the MLflow tracking folder
# (which is also available in the server) but renamed here for easier access based on the suffix defined in the config
# file.
# Bool to set whether to copy the runs to the final output subdirectory - for testing only this can be disabled
if track_final: #IMPROVE: take out useful individual subfolders vs whole folder contents - need to determine which bits are useful
    print("\'track_final\' has been enabled, so the model information will be copied to ./model_output for easier viewing.")

    # Determine file locations
    final_folder = Path("mlruns") / final_exp_id / final_run_id
    ml_artifacts = Path("mlartifacts") / final_run_id
    output_folder = Path("model_output") / run_name
    output_artifacts = output_folder #IMPROVE (and in NN) - remove or improve as currently not needed
    data_folder = Path(data_dir)
    graph_folder = Path(graphs_dir)

    # Copy final model folder contents # TODO not sure if an mlflow or data copying issue, but when enable_tracking is disabled there is some convoluted file structures in model_output
    shutil.copytree(final_folder, output_folder, dirs_exist_ok=True)
    print(f"\nCopying {final_folder} to {output_folder}")
    # Copy final model artifacts from mlartifacts to the model_output model file # IMPROVE - revamp for enable_tracking false so the file structure is the same (either use mlartifacts for local or /artifacts for remote tracking)
    #  Note: since setting an experiment name changes the artifacts location to mlartifacts instead of in the mlruns (run) folder, we will copy it over for our final output
    shutil.copytree(ml_artifacts, output_artifacts, dirs_exist_ok=True)
    print(f"Copying {ml_artifacts} to {output_artifacts}")
    # Copy training data and graphs folder
    shutil.copytree(data_folder, output_folder / "training_data", dirs_exist_ok=True) #IMPROVE more elegant
    print(f"Copying {data_folder} to {output_folder}/training_data")
    shutil.copytree(graph_folder, output_folder / "training_graphs", dirs_exist_ok=True)
    print(f"Copying {graph_folder} to {output_folder}/training_graphs\n")

    # For HPC use - copy model folder over (instead of logging to best_model in artifacts, it saves the contents to mlartifacts/models/m-{run_id}/artifacts
    faulty_bm_path = Path("mlartifacts") / "models" / f"{model_id}" / "artifacts"
    if os.path.exists(faulty_bm_path):
        shutil.copytree(faulty_bm_path, f"{output_artifacts}/artifacts/best_model", dirs_exist_ok=True)
        print(f"For HPC: Copying {faulty_bm_path} to {output_artifacts}/artifacts/best_model\n")

    # Make note of the corresponding hyperopt MLflow run
    hyper_run_file = final_folder / "hyperopt_run_name.txt"
    hyper_run_file.write_text(f"{hyperopt_name}")

    # Save key metrics to csv #TODO - in order to simplify for bug fixing, removed the overwrite for same run name. Restore.
    key_metrics_path = f"key_metrics_{run_name}.csv"
    key_metrics = {
        'ML Test Accuracy': test_accuracy,
        'ML Test F1': test_f1,
        'ML Test AUROC': test_roc,
             }
    run_metrics = pd.DataFrame(key_metrics, index=[run_name])
    run_metrics.to_csv(f"{output_folder}/{key_metrics_path}")

    # Save to masterlist of metrics
    all_key_metrics_path = "key_metrics.csv"
    if os.path.exists(all_key_metrics_path):
        all_metrics = pd.read_csv(all_key_metrics_path, index_col=0)
        # Update only ML metrics columns for this run
        for col in key_metrics.keys():
            all_metrics.loc[run_name, col] = run_metrics.loc[run_name, col]
            all_metrics.to_csv(all_key_metrics_path)
    else:
        run_metrics.to_csv(all_key_metrics_path)


# Print run ids
print(store_hyp_id)
print(store_final_id)

# Close all figures
plt.close('all')

# WARNING: If getting the 'too many 500 error responses' warning due to deleting files, run 'kill $(lsof -t -i tcp:8080)' in the terminal

# Example output:
# Test accuracy with best model (rf): 0.6000
# Test F1 score with best model (rf): 0.6957


# IMPROVE: Early stopping isn't implemented at all because it would work for some and not others so is more complicated to implement - but could add.
#  Could also do an ensemble model approach for the final training, and stacking/voting

# IMPROVE once final model is obtained, I'll likely want to plot more model-specific graphs
