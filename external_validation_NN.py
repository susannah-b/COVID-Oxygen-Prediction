import torch
import torch.nn as nn
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
from functions import basic_train, port_in_use, pca_pre_post_fs, plot_learning_curve, \
    plot_roc_auc, plot_feature_importance, plot_calibration_curve, plot_decision_tree, plot_precision_recall, \
    plot_pca_predicted, plot_confusion_matrix, plot_fs_performance, plot_pca_original, plot_pca_test_unprocessed, \
    remaining_meta, grouped_shap
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
import shap
from sklearn.base import clone
from mlflow.tracking import MlflowClient

### SET RANDOM SEEDS ###################################################################################################
# Set global random seeds
torch.manual_seed(42) # PyTorch CPU
torch.cuda.manual_seed_all(42) # PyTorch GPU (if available)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

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
enable_tracking = config['general']['enable_tracking']

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
y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True).squeeze()  # Convert to 1D array

# Convert y to float32 for pytorch
y_data = y_data.astype(np.float32)

print(f"Validation samples: {len(X_data)}")
print(f"Feature dimensions: {X_data.shape[1]} | Classes: {y_data.nunique()}")

### PCA ON ORIGINAL DATA ###############################################################################################
plot_pca_test_unprocessed(X_data, y_data, graphs_dir)

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

### CREATE WRAPPER FOR MODEL ###########################################################################################
# Due to the combined sklearn and pytorch elements of the pipeline (preprocessing and NN model), the model is required to
#  be logged as pyfunc, not sklearn or pytorch. It therefore has no .predict attribute needed for SHAP KernelExplainer.

# Wrapper function to convert numpy input to pytorch tensors
def model_predict(X):
    # Convert numpy array to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Make predictions
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        # Return probabilities (sigmoid output)
        return torch.sigmoid(logits).numpy()

# IMPROVE: this is defined in both the training script and here. move to functions

### MLFLOW TRACKING ####################################################################################################
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

### LOAD MODEL #########################################################################################################
if not args.from_pipeline:
    # Set run info to take model from manually
    model_name = "15_0729-160809_HPC" # Name of the experiment (can be found in model_output and is printed at the end of the run) - Change as needed
else:
    model_name = run_name

# Load model by run ID # TODO - think this should be changed to the mlruns file as model_output isn't always made
model_output = f"model_output/{model_name}"
model_path = f"{model_output}/artifacts/best_model"
model = mlflow.pyfunc.load_model(model_path)

# Load input features
features_path = f"{model_output}/training_data/NN/input_features.joblib"
input_features = joblib.load(features_path)

# Load selected features
features_path_2 = f"{model_output}/training_data/NN/selected_features.joblib"
selected_features = joblib.load(features_path_2)

# Set MLflow logging details
if enable_tracking: # Have to use a unique name or it creates issues with artifact tracking
    exp_name = "Oxygen Prediction NN Validation - Surrey"
else:
    exp_name = "Oxygen Prediction NN Validtion- Surrey - Offline"

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

### PREPARE DATA #######################################################################################################
# Filter validation data to the original features
X_data = X_data[input_features]
# Ensure same column order
X_data = X_data.reindex(columns=input_features)
### FIT MODEL ##########################################################################################################
mlflow.pytorch.autolog()

# Set run name - OG model with _validation appended
val_run_name = f"{model_name}_validation"

# Start MLflow run
with mlflow.start_run(run_name=val_run_name) as run:
    print(f"\nNow predicting oxygen need for the validation data using the neural network model.") # todo {classifier_type} post b_t
    mlflow.set_tag("Run name", val_run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("Phase", "Model validation")
    # mlflow.set_tag("Hyperopt MLflow run", hyperopt_name) # Note: haven't included the associated hyperopt selection run for OG model but could be determined if useful
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    val_exp_id = run.info.experiment_id  # Get experiment id for folder management

    # Predict on external data
    y_proba = model.predict(X_data)
    predictions = (y_proba > 0.5).astype(int)

    # Print confusion matrix
    cm = confusion_matrix(y_data, predictions)
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix
    plot_confusion_matrix(cm, graphs_dir)

    # Get prediction metrics
    test_accuracy = accuracy_score(y_data, predictions)
    test_f1 = f1_score(y_data, predictions)
    test_roc = roc_auc_score(y_data, y_proba)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) # aka TPR, recall
    specificity = tn / (tn + fp) # aka TNR
    precision = tp / (tp + fp) # aka PPV
    npv = tn / (tn + fn)

    # Print and log metrics
    print(f"\nValidation accuracy: {test_accuracy:.4f}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Validation F1 score: {test_f1:.4f}")
    mlflow.log_metric("test_f1", test_f1)
    print(f"Validation ROC_AUC: {test_roc:.4f}")
    mlflow.log_metric("test_roc", test_roc)
    mlflow.log_metric("sensitivity-tpr-recall", sensitivity)
    mlflow.log_metric("specificity-tnr", specificity)
    mlflow.log_metric("precision-ppv", precision)
    mlflow.log_metric("npv", npv)

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
    ### GRAPHS #############################################################################################################
    # Plot PCA on the combined dataset - i.e. all data after feature selection #todo for all pcas, check a few samples to confirm they're correct (label on graph)
    with mlflow.start_run(nested=True):  # Start another run to avoid auologging conflicts
        mlflow.pytorch.autolog(disable=True)  # Disables autolog inside this run
        # Call function to plot PCA on the dataset post feature selection
        pca_pre_post_fs(X_data, selected_features, y_data, graphs_dir, "After")

    # Plot ROC/AUC curves
    plot_roc_auc(y_proba, y_data, graphs_dir)

    # SHAP
    # WARNING - having issues with 'X does not have valid feature names' - fix if needed but avoid using SHAP graphs for exval if possible.
    X_data_df = pd.DataFrame(X_data, columns=input_features)  # Convert back to df for function use
    X_data_sf_df = X_data_df[selected_features] # Filter down to processed features
    explainer = shap.KernelExplainer(model.predict, X_data_sf_df[:10])  # WARNING: As background is done on the same dataset I'm not sure of the validity
    shap_values = explainer.shap_values(X_data_sf_df, nsamples=100)  # entire dataset
    plt.figure()
    shap.summary_plot(shap_values, X_data_sf_df, feature_names=selected_features, show=False)
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}/SHAP_graph.png", dpi=300, bbox_inches='tight')
    plt.close()

    ### Repeat SHAP but this time aggregate metadata and protein data to examine influence
    # Calculate meta columns after feature selection
    starting_meta_cols_count = config['general']['training_meta_cols'] - 1  # -1 to exclude label
    meta_cols_before = X_data_df.iloc[:, :starting_meta_cols_count].columns
    protein_cols_before = X_data_df.iloc[:, starting_meta_cols_count:].columns
    meta_cols_surrey = remaining_meta(meta_cols_before.tolist(), X_data_df[selected_features], sample_inves_7=False, graphs_dir=None)  # Note: The graph produced here isn't really needed but kept in to visualise selected features
    metadata_features = selected_features[:meta_cols_surrey]
    proteomics_features = selected_features[meta_cols_surrey:]

    # Split SHAP based on class
    shap_groups = {"Metadata": metadata_features, "Proteomics data": proteomics_features}
    shap_grouped = grouped_shap(shap_values, selected_features, shap_groups)
    plt.figure()
    shap.summary_plot(shap_grouped.values, feature_names=shap_grouped.columns, show=False)
    plt.tight_layout()
    plt.savefig(f"{graphs_dir}/SHAP_graph_grouped.png", dpi=300, bbox_inches='tight')
    with open(f"{graphs_dir}/SHAP_warning.txt", "w") as f:
        f.write("Warning: Due to issues with SHAP graph generation, it is recommended to avoid using the external "
                "validation SHAP graphs. If needed, ideally return to the script to fix the issue with feature names.")
    # Plot calibration curve
    classifier_type = 'neural_network'
    plot_calibration_curve(y_proba, y_data, classifier_type, graphs_dir)

    # Plot a precision-recall curve
    plot_precision_recall(y_proba, y_data, graphs_dir)

    ### Plot PCA on final predictions - Test data before and after prediction
    with mlflow.start_run(nested=True):  # Start another run to avoid autologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run
        # Plot PCA
        plot_pca_predicted(X_data, selected_features, y_data, graphs_dir, predictions) # todo fix for NN

    # Print run id
    val_run_id = run.info.run_id
    store_val_id = f"Run {val_run_name} for validation predictions completed. Run ID is {val_run_id}"

    # Log artifacts
    if enable_tracking:
        mlflow.log_artifacts(graphs_dir, artifact_path="val_graphs")
        mlflow.log_artifacts(data_dir, artifact_path="val_tables")

### STORE RESULTS IN NEW FOLDER ########################################################################################
# Move and rename runs to a new directory for easier examination - results are copied from the MLflow tracking folder
# (which is also available in the server) but renamed here for easier access based on the original model name.
# Bool to set whether to copy the runs to the final output subdirectory - for testing only this can be disabled
if track_final: #IMPROVE: take out useful individual subfolders vs whole folder contents - need to determine which bits are useful
    print(f"\'track_final\' has been enabled, so the model information will be copied to ./model_output/ML under the original experiment {model_name} for easier viewing.")

    # Determine file locations
    val_folder = Path("mlruns")  / val_exp_id / val_run_id
    ml_artifacts = Path("mlartifacts")  / val_run_id # Find artifacts for current run
    output_folder = Path(f"{model_output}/external_validation") # Put within original model folder under external_validation
    output_artifacts = output_folder
    data_folder = Path(data_dir)
    graph_folder = Path(graphs_dir)

    # Copy final model folder contents
    shutil.copytree(val_folder, output_folder, dirs_exist_ok=True)
    print(f"\nCopying {val_folder} to {output_folder}")
    # Copy validation model artifacts from mlartifacts to the model_output external validation file
        #  Note: since setting an experiment name changes the artifacts location to mlartifacts instead of in the mlruns (run) folder, we will copy it over for our final output
    shutil.copytree(ml_artifacts, output_artifacts, dirs_exist_ok=True)
    print(f"Copying {ml_artifacts} to {output_artifacts}")
    # Copy training data and graphs folder
    shutil.copytree(data_folder, output_folder / "training_data", dirs_exist_ok=True)  # IMPROVE more elegant
    print(f"Copying {data_folder} to {output_folder}/training_data")
    shutil.copytree(graph_folder, output_folder / "training_graphs", dirs_exist_ok=True)
    print(f"Copying {graph_folder} to {output_folder}/training_graphs\n")

    # Read in key metrics from training and update for validation
    key_metrics_path = f"{model_output}/key_metrics_{model_name}.csv"
    existing_metrics = pd.read_csv(key_metrics_path, index_col=0)
    key_metrics = {
        'NN Validation Accuracy': test_accuracy,
        'NN Validation F1': test_f1,
        'N Validation AUROC': test_roc,
    }
    # Update existing metrics
    for key, value in key_metrics.items():
        existing_metrics[key] = value
    existing_metrics.to_csv(key_metrics_path)

    # Update the master metrics file
    all_key_metrics_path = "key_metrics.csv"
    if os.path.exists(all_key_metrics_path):
        all_metrics = pd.read_csv(all_key_metrics_path, index_col=0)
        # Drop existing row if present
        all_metrics.drop(index=model_name, errors='ignore', inplace=True)
        # Update
        if run_name not in all_metrics.index:
            all_metrics = pd.concat([all_metrics, existing_metrics])
    else:
        all_metrics = existing_metrics
    all_metrics.to_csv(all_key_metrics_path)

# Print run ids
print(store_val_id)

# Close all figures
plt.close('all')