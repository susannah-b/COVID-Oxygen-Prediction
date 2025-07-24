### SCRIPT USAGE #######################################################################################################
# Run this script to test models on the external ISARIC data.
# Set the run_id to the ID of the finished model.

######### SETUP ########################################################################################################
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import time
import mlflow
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import shutil
import argparse
import yaml
import os

from functions import port_in_use, pca_pre_post_fs, plot_roc_auc, plot_feature_importance, plot_calibration_curve, \
    plot_decision_tree, plot_precision_recall, plot_pca_predicted, plot_confusion_matrix, plot_pca_original, \
    plot_pca_test_unprocessed

# Set global random seeds
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
    config_path = Path(f"inputs/ML/{run_name}/config.yaml")

# Read config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set parameters for this file:
host = config['general']['host']
port = config['general']['port']
track_final = config['model_building']['track_final']

### LOAD ISARIC DATA ###################################################################################################
# Set input directories
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    data_dir = 'validation_data' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    data_dir = f'inputs/ML/{run_name}/validation_data'

# Create output directories for the data
output_data_dir = f'{data_dir}/ML'
os.makedirs(output_data_dir, exist_ok=True)

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    graphs_dir = 'validation_graphs/ML' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    graphs_dir = f'inputs/ML/{run_name}/validation_graphs/ML'
os.makedirs(graphs_dir, exist_ok=True) # Make the ML graph that's specific to the ML outputs

dataset = "ISARIC"
X_path = Path(__file__).parent / data_dir / f"{dataset}_X.csv"
y_path = Path(__file__).parent / data_dir / f"{dataset}_y.csv"
X_data = pd.read_csv(X_path, index_col=0)
y_data = pd.read_csv(y_path, index_col=0)

# Ensure y_data is a 1-D Series #TODO why do i need to do this for exval but not m_b?
y_data = y_data.reset_index(drop=True).squeeze() # Convert to a series to avoid an issue with plot_pca_predicted

print(f"Validation samples: {len(X_data)}")
print(f"Feature dimensions: {X_data.shape[1]} | Classes: {y_data.nunique()}")

### PCA ON ORIGINAL DATA ###############################################################################################
plot_pca_test_unprocessed(X_data, y_data, graphs_dir)

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
    model_name = "173_0724-213624_graphs" # Name of the experiment (can be found in model_output and is printed at the end of the run) - Change as needed
else:
    model_name = run_name

# Load model by run ID # TODO - think this should be changed to the mlruns file as model_output isn't always made
model_output = f"model_output/{model_name}"
model_path = f"{model_output}/artifacts/best_model"
model = mlflow.sklearn.load_model(model_path)

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
mlflow.sklearn.autolog()

# Set run name - OG model with _validation appended
val_run_name = f"{model_name}_validation"

# Get classifier type
classifier_path = Path(f"{model_output}/params/type")
with open(classifier_path, 'r') as f:
    classifier_type = f.read().strip()

# Start MLflow run
with mlflow.start_run(run_name=val_run_name) as run:
    print(f"\nNow predicting oxygen need for the validation data using the {classifier_type} model.")
    mlflow.set_tag("Run name", val_run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("Phase", "Model validation")
    # mlflow.set_tag("Hyperopt MLflow run", hyperopt_name) # Note: haven't included the associated hyperopt selection run for OG model but could be determined if useful
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    val_exp_id = run.info.experiment_id  # Get experiment id for folder management

    # Predict on external data
    predictions = model.predict(X_data)
    y_proba = model.predict_proba(X_data)[:, 1]

    # Print confusion matrix
    cm = confusion_matrix(y_data, predictions)
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix
    plot_confusion_matrix(cm, graphs_dir)

    # Get prediction metrics
    test_accuracy = accuracy_score(y_data, predictions)
    test_f1 = f1_score(y_data, predictions)
    test_roc = roc_auc_score(y_data, y_proba)
    test_roc_auc = roc_auc_score(y_data, y_proba)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # aka TPR, recall
    specificity = tn / (tn + fp)  # aka TNR
    precision = tp / (tp + fp)  # aka PPV
    npv = tn / (tn + fn)

    # Print and log metrics
    print(f"\nValidation accuracy: {test_accuracy:.4f}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Validation F1 score: {test_f1:.4f}")
    mlflow.log_metric("test_f1", test_f1)
    print(f"Validation ROC_AUC: {test_roc_auc:.4f}")
    mlflow.log_metric("test_roc", test_roc_auc)
    mlflow.log_metric("sensitivity-tpr-recall", sensitivity)
    mlflow.log_metric("specificity-tnr", specificity)
    mlflow.log_metric("precision-ppv", precision)
    mlflow.log_metric("npv", npv)


    # Save predictions to csv
    model_results = pd.DataFrame({'Real values': y_data.values, 'ML predictions': predictions}, index=X_data.index)
    model_results.to_csv(f"{output_data_dir}/Prediction_results_validation_data.csv")

    ### GRAPHS #############################################################################################################
    # Plot PCA on the combined dataset - i.e. all data after feature selection #todo for all pcas, check a few samples to confirm they're correct (label on graph)
    with mlflow.start_run(nested=True): # Start another run to avoid auologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run
        # Call function to plot PCA on the dataset post feature selection
        pca_pre_post_fs(X_data, selected_features, y_data, graphs_dir, "After")

    # Plot ROC/AUC curves
    plot_roc_auc(y_proba, y_data, graphs_dir)

    # Plot feature importance
    plot_feature_importance(classifier_type, model, selected_features, graphs_dir, output_data_dir, model.get_params(),
                            X_data, y_data)

    # Plot calibration curve
    plot_calibration_curve(y_proba, y_data, classifier_type, graphs_dir)

    # Plot decision tree
    class_names = np.array(['No_Oxygen_Need', 'Oxygen_Need'])
    plot_decision_tree(classifier_type, model, X_data, class_names, output_data_dir, graphs_dir)

    # Plot a precision-recall curve
    plot_precision_recall(y_proba, y_data, graphs_dir)

    ### Plot PCA on final predictions - Test data before and after prediction
    with mlflow.start_run(nested=True):  # Start another run to avoid autologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run

        # Plot PCA
        plot_pca_predicted(X_data, selected_features, y_data, graphs_dir, predictions)

    # Print run id
    val_run_id = run.info.run_id
    store_val_id = f"Run {val_run_name} for validation predictions completed. Run ID is {val_run_id}"

    # Log artifacts
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
    ml_artifacts = Path("mlartifacts") / val_exp_id / val_run_id # Find artifacts for current run
    output_folder = Path(f"{model_output}/external_validation") # Put within original model folder under external_validation
    output_artifacts = output_folder
    data_folder = Path(data_dir)
    graph_folder = Path(graphs_dir)

    # Copy final model folder contents
    shutil.copytree(val_folder, output_folder, dirs_exist_ok=True)
    # Copy validation model artifacts from mlartifacts to the model_output external validation file
        #  Note: since setting an experiment name changes the artifacts location to mlartifacts instead of in the mlruns (run) folder, we will copy it over for our final output
    shutil.copytree(ml_artifacts, output_artifacts, dirs_exist_ok=True)
    # Copy training data and graphs folder
    shutil.copytree(data_folder, output_folder / data_dir, dirs_exist_ok=True)
    shutil.copytree(graph_folder, output_folder / graphs_dir, dirs_exist_ok=True)

    # Read in key metrics from training and update for validation
    key_metrics_path = f"{model_output}/key_metrics_{model_name}.csv"
    existing_metrics = pd.read_csv(key_metrics_path, index_col=0)
    key_metrics = {
        'ML Validation Accuracy': test_accuracy,
        'ML Validation F1': test_f1,
        'ML Validation AUROC': test_roc,
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