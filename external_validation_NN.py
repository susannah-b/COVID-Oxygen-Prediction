import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss, roc_auc_score
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
import yaml
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import warnings
from functions import basic_train, port_in_use, pca_original, plot_learning_curve, \
    plot_roc_auc, plot_feature_importance, plot_calibration_curve, plot_decision_tree, plot_precision_recall, \
    plot_pca_predicted, plot_confusion_matrix, plot_fs_performance
import re
import os
from datetime import datetime
import subprocess
import time
import shutil
import json
import joblib
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import clone

### SET RANDOM SEEDS ###################################################################################################
# Set global random seeds
torch.manual_seed(42) # PyTorch CPU
torch.cuda.manual_seed_all(42) # PyTorch GPU (if available)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
run_name = args.run_name # Note this is for the original run, not the validation run (which has _validation appended)

#### READ CONFIG FILE ##################################################################################################
# Set config path based on whether the script is run standlone or part of pipeline.py (config moved to 'inputs')
if not args.from_pipeline:
    config_path = Path("config.yaml")
else:
    config_path = Path(f"inputs/NN/{run_name}/config.yaml")

# Read config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set parameters for this file:
host = config['general']['host']
port = config['general']['port']
track_final = config['model_building']['track_final']

### LOAD IN VALIDATION DATA ############################################################################################
# Set input directories
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    data_dir = 'validation_data' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    data_dir = f'inputs/NN/{run_name}/validation_data'

# Create output directories for the data
output_data_dir = f'{data_dir}/NN'
os.makedirs(output_data_dir, exist_ok=True)

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    graphs_dir = 'validation_graphs/NN' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    graphs_dir = f'inputs/NN/{run_name}/validation_graphs/NN'
os.makedirs(graphs_dir, exist_ok=True)

dataset = "ISARIC"
X_path = Path(__file__).parent / data_dir / f"{dataset}_X.csv"
y_path = Path(__file__).parent / data_dir / f"{dataset}_y.csv"
X_data = pd.read_csv(X_path, index_col=0)
y_data = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array

# Convert y to float32 for pytorch
y_data = y_data.astype(np.float32)

print(f"Validation samples: {len(X_data)}")
print(f"Feature dimensions: {X_data.shape[1]} | Classes: {y_data.nunique()}")

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
            # Convert to float32 from pandas float64 for pytorch
            float64_cols = X.select_dtypes(include=['float64']).columns
            X[float64_cols] = X[float64_cols].astype(np.float32)
        return X
# TODO: this is a duplicate of the transformer in the neural network so is now defined twice - ideally import from another script
### MLFLOW TRACKING ####################################################################################################
if not port_in_use(host, port):
    print(f"Running tracking server on {host}:{port}")
    subprocess.Popen(["mlflow", "server", "--backend-store-uri", "./mlruns", "--host", host, "--port", f"{port}"])
else:
    print(f"MLflow tracking server already listening on {host}:{port}")

# Pause to allow the server to boot up
    time.sleep(5)

# Set tracking URI
mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

### LOAD MODEL #########################################################################################################
if not args.from_pipeline:
    # Set run info to take model from manually
    model_name = "7_0719-213333_cross_validation_ES" # Name of the experiment (can be found in model_output and is printed at the end of the run) - Change as needed
else:
    model_name = run_name

# Load model by run ID # TODO - think this should be changed to the mlruns file as model_output isn't always made
model_output = f"model_output/{model_name}"
model_path = f"{model_output}/artifacts/best_model"
model = mlflow.pyfunc.load_model(model_path)

# Load input features
features_path = f"{model_output}/training_data/ML/input_features.joblib"
input_features = joblib.load(features_path)

# Load selected features
features_path_2 = f"{model_output}/training_data/ML/selected_features.joblib"
selected_features = joblib.load(features_path_2)

# Set MLflow logging details
mlflow.set_experiment("Oxygen Prediction - Validation")

### PREPARE DATA #######################################################################################################
# Filter validation data to the original features
X_data = X_data[input_features]
# Ensure same column order
X_data = X_data.reindex(columns=input_features)

### FIT MODEL ##########################################################################################################
mlflow.pytorch.autolog()

# Set run name - OG model with _validation appended
val_run_name = f"{model_name}_validation"

# Get classifier type #todo not yet implemented for NN - do after basic training
# classifier_path = Path(f"{model_output}/params/type")
# with open(classifier_path, 'r') as f:
#     classifier_type = f.read().strip()

# Start MLflow run
with mlflow.start_run(run_name=val_run_name) as run:
    print(f"Now predicting oxygen need for the validation data using the neural network model.") # todo {classifier_type} post b_t
    mlflow.set_tag("Run name", val_run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("Phase", "Model validation")
    # mlflow.set_tag("Hyperopt MLflow run", hyperopt_name) # Note: haven't included the associated hyperopt selection run for OG model but could be determined if useful
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    val_exp_id = run.info.experiment_id  # Get experiment id for folder management

    # Predict on external data
    y_proba = model.predict(X_data)
    predictions = (y_proba > 0.5).astype(int)

    # Get prediction metrics
    test_accuracy = accuracy_score(y_data, predictions)
    test_f1 = f1_score(y_data, predictions)
    test_roc_auc = roc_auc_score(y_data, y_proba)

    # Print confusion matrix
    cm = confusion_matrix(y_data, predictions)
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix
    plot_confusion_matrix(cm, graphs_dir)

    print(f"\nValidation accuracy: {test_accuracy:.4f}")
    print(f"Validation F1 score: {test_f1:.4f}")
    print(f"Validation ROC_AUC: {test_roc_auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_data, predictions, target_names=["O2 not required", "O2 required"]))

    ### Save predictions to csv
    NN_prediction_path = Path(f"{output_data_dir}/Prediction_results_validation_data.csv")

    # Define path for possible pre-existing results file
    if not args.from_pipeline:
        ML_prediction_path = Path(
            f"{data_dir}/ML/Prediction_results_validation_data.csv")  # Saved to cwd training data file - WARNING: this will add results to the latest ML results if present. The config for these may not be the same. To properly store based on run name (and same config), use the wrapper script.
    else:
        ML_prediction_path = Path(
            f'model_output/{run_name}/training_data/ML/Prediction_results_validation_data.csv')  # Saved to input storage file for ML

    # Make df of results
    NN_results = pd.DataFrame({'NN predictions': predictions}, index=X_data.index)

    # Check if same run has a prediction results file for the traditional ML model
    if not os.path.exists(ML_prediction_path):  # If it doesn't exist, make a new file
        NN_results.to_csv(f"{output_data_dir}/Prediction_results_validation_data.csv")
    else:  # If it does exist, append and copy
        ML_results = pd.read_csv(ML_prediction_path, index_col=0)  # Read in old results
        # Delete pre-existing columns (i.e. if running as a standalone script, it won't append multiple results for multiple ML runs)
        overlap = ML_results.columns.intersection(NN_results.columns)
        ML_results = ML_results.drop(columns=overlap)
        results_combined = pd.concat([ML_results, NN_results], axis=1)
        results_combined.to_csv(ML_prediction_path)
        shutil.copy2(ML_prediction_path, NN_prediction_path)  # Copy back to NN results

    ### GRAPHS #############################################################################################################
    #TODO as with regular NN, need to adapt /make required graphs

    #TODO pca is commented - works for NN on test (and normal ML) but failing with NN validation - fix later
    # Plot PCA on the combined dataset - i.e. original data after feature selection #todo for all pcas, check a few samples to confirm they're correct (label on graph)
    # with mlflow.start_run(nested=True):  # Start another run to avoid auologging conflicts
    #     mlflow.pytorch.autolog(disable=True)  # Disables autolog inside this run
    #     # Call function to plot PCA on the dataset prior to feature selection
    #     pca_original(X_data, input_features, y_data, graphs_dir)
