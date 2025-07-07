### SCRIPT USAGE #######################################################################################################
# Run this script to test models on the external ISARIC data.
# Set the run_id to the ID of the finished model.

######### SETUP ########################################################################################################
import mlflow
import pandas as pd
from pathlib import Path
import subprocess
from functions import port_in_use
import time
import mlflow
import joblib

### LOAD ISARIC DATA ###################################################################################################
data_dir = 'validation_data'
dataset = "ISARIC"
X_path = Path(__file__).parent / data_dir / f"{dataset}_X_train.csv"
y_path = Path(__file__).parent / data_dir / f"{dataset}_y_train.csv"
X_data = pd.read_csv(X_path, index_col=0)
y_data = pd.read_csv(y_path, index_col=0)

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
exp_name = "6_0707-1601_RF_only_practice_validation" # Name of the experiment (can be found in model_output and is printed at the end of the run) - Change as needed
# Load model by run ID
model_output = f"model_output/{exp_name}"
model_path = f"{model_output}/artifacts/best_model"
model = mlflow.sklearn.load_model(model_path)

# Load selected features
features_path = f"{model_output}/training_data/selected_features.joblib"
selected_features = joblib.load(features_path)

# Set MLflow logging details
mlflow.set_experiment("Oxygen Prediction - Validation")

### APPLY MODEL ########################################################################################################
# Filter validation data to the original features
X_data = X_data[selected_features]
# Ensure same column order
X_data = X_data.reindex(columns=selected_features)

# Predict on external data
predictions = model.predict(X_data)
print(predictions)



#Todo can do something like this but see what i have from the training script first
# result = pd.DataFrame(X_test, columns=iris_feature_names)
# result["actual_class"] = y_test
# result["predicted_class"] = predictions


# todo: have this basic model setup done. do i get the graphs and where are they saved? ideally i want them in a new
#  file for ex val. might first just set a data_output file so i keep my folder clean, but then see about graphs.