### SCRIPT USAGE #######################################################################################################
# Run this script to test models on the external ISARIC data.
# Set the run_id to the ID of the finished model.
from operator import truediv

######### SETUP ########################################################################################################
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import time
import mlflow
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import shutil
from functions import port_in_use, pca_original, plot_learning_curve, \
    plot_roc_auc, plot_feature_importance, plot_calibration_curve, plot_decision_tree, plot_precision_recall, \
    plot_pca_predicted, plot_confusion_matrix

### LOAD ISARIC DATA ###################################################################################################
data_dir = 'validation_data'
dataset = "ISARIC"
X_path = Path(__file__).parent / data_dir / f"{dataset}_X_train.csv" # Note: Although the ISARIC data as saved as 'train' data, this is actually the wholedataset for validation/testing
y_path = Path(__file__).parent / data_dir / f"{dataset}_y_train.csv"
X_data = pd.read_csv(X_path, index_col=0)
y_data = pd.read_csv(y_path, index_col=0)

# Ensure y_data is a 1-D Series #TODO why do i need to do this for exval but not m_b?
y_data = y_data.reset_index(drop=True).squeeze() # Convert to a series to avoid an issue with plot_pca_predicted

# Set output folders
graphs_dir = 'validation_graphs'

### MLFLOW TRACKING ####################################################################################################
# Start local tracking server
host = "127.0.0.1" # Note: for unified tracking change these to be the same as in model_building.py
port = 8080

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
# Set run info to take model from
model_name = "13_0709-2311_RF_only_practice_validation" # Name of the experiment (can be found in model_output and is printed at the end of the run) - Change as needed
# Load model by run ID
model_output = f"model_output/{model_name}"
model_path = f"{model_output}/artifacts/best_model"
model = mlflow.sklearn.load_model(model_path)

# Load input features
features_path = f"{model_output}/training_data/input_features.joblib"
input_features = joblib.load(features_path)

# Load selected features
features_path_2 = f"{model_output}/training_data/selected_features.joblib"
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
store_val_id = None # Initialise value to store run ID to print at end
val_run_id = None # ID for validation run
val_exp_id = None # ID for validation run

# Set run name - OG model with _validation appended
run_name = f"{model_name}_validation"

# Get classifier type
classifier_path = Path(f"{model_output}/params/type")
with open(classifier_path, 'r') as f:
    classifier_type = f.read().strip()

# Start MLflow run
with mlflow.start_run(run_name=run_name) as run:
    print(f"Now predicting oxygen need for the validation data using the {classifier_type} model.")
    mlflow.set_tag("Run name", run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("Phase", "Model validation")
    # mlflow.set_tag("Hyperopt MLflow run", hyperopt_name) # Note: haven't included the associated hyperopt selection run for OG model but could be determined if useful
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    val_exp_id = run.info.experiment_id  # Get experiment id for folder management

    # Predict on external data
    predictions = model.predict(X_data)

    # Get prediction metrics
    test_accuracy = accuracy_score(y_data, predictions)
    test_f1 = f1_score(y_data, predictions)

    # Print confusion matrix
    cm = confusion_matrix(y_data, predictions)
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix
    plot_confusion_matrix(cm, graphs_dir)

    print(f"\nValidation accuracy: {test_accuracy:.4f}")
    print(f"Validation F1 score: {test_f1:.4f}")

### GRAPHS #############################################################################################################
    # Plot PCA on the combined dataset - i.e. original data after feature selection #todo for all pcas, check a few samples to confirm they're correct (label on graph)
    with mlflow.start_run(nested=True): # Start another run to avoid auologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run
        # Call function to plot PCA on the dataset prior to feature selection
        pca_original(X_data, input_features, y_data, graphs_dir)

    # Plot ROC/AUC curves
    plot_roc_auc(model, X_data, y_data, graphs_dir)

    # Plot feature importance
    plot_feature_importance(classifier_type, model, selected_features, graphs_dir, data_dir, model.get_params(),
                            X_data, y_data)

    # Plot calibration curve
    plot_calibration_curve(model, X_data, y_data, classifier_type, graphs_dir)

    # Plot decision tree
    class_names = np.array(['No_Oxygen_Need', 'Oxygen_Need'])
    plot_decision_tree(classifier_type, model, X_data, class_names, data_dir, graphs_dir)

    # Plot a precision-recall curve
    plot_precision_recall(model, X_data, y_data, graphs_dir)

    ### Plot PCA on final predictions - Test data
    with mlflow.start_run(nested=True):  # Start another run to avoid autologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run

        # Plot PCA
        plot_pca_predicted(X_data, selected_features, y_data, graphs_dir, predictions)

    # Print run id
    val_run_id = run.info.run_id
    store_val_id = f"Run {run_name} for validation predictions completed. Run ID is {val_run_id}"

    # Log artifacts
    mlflow.log_artifacts(graphs_dir, artifact_path="graphs")
    mlflow.log_artifacts(data_dir, artifact_path="tables")

### STORE RESULTS IN NEW FOLDER ########################################################################################
# Move and rename runs to a new directory for easier examination - results are copied from the MLflow tracking folder
# (which is also available in the server) but renamed here for easier access based on the original model name.
# Bool to set whether to copy the runs to the final output subdirectory - for testing only this can be disabled
track_final = True #IMPROVE: take out useful individual subfolders vs whole folder contents - need to determine which bits are useful
if track_final:
    print(f"\'track_final\' has been enabled, so the model information will be copied to ./model_output under the original experiment {model_name} for easier viewing.")

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

# Print run ids
print(store_val_id)