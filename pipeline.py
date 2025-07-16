### SCRIPT USAGE #######################################################################################################
# Run this script to run the entire model pipeline; from raw data to external validation (if enabled). An 'input_storage'
# folder will be created to store the correct version of the data upon commencing a run, so multiple runs can be completed
# simultaneously without overwriting. Scripts can also be run individually, in which case the data used will be the files
# currently present in the working directory.

import subprocess
from pathlib import Path
import os
from datetime import datetime
import yaml
import shutil

### READ IN CONFIG FILE ################################################################################################
# Create config fil if it doesn't exist
config_path = Path("config.yaml")
default_config = Path("default_config.yaml")
if not os.path.exists(config_path):
    shutil.copy2(default_config, config_path)

# Read in base config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

### CREATE DIRECTORY TO STORE DATA INPUTS FOR THIS RUN #################################################################
# Set timestamp for current run (used as a unique identifier for each run)
timestamp = datetime.now().strftime("%m%d-%H%M%S")

# Set run name
run_number = config["general"]["run_number"]
run_suffix = config["general"]["run_suffix"] or "Unspecified" # Set to unspecified if empty
run_name = f"{run_number}_{timestamp}_{run_suffix}" # Unique ID for each model built
build_ML = config["traditional"]
build_NN = config["neural_network"]

### Create an input folder for the model to store the data
input_storage_prelim = Path("inputs") # Where feature_engineering.py and data_preprocessing output the data for both models
input_storage_ML = Path("inputs/ML")
input_storage_NN = Path("inputs/NN")

# Make the relevant folders
os.makedirs(input_storage_prelim, exist_ok=True)
if build_ML:
    os.makedirs(input_storage_ML, exist_ok=True)
if build_NN:
    os.makedirs(input_storage_NN, exist_ok=True)

# Create subdirectory to store data/config for the model
model_storage_prelim = Path(f"{input_storage_prelim}/{run_name}")
model_storage_ML = Path(f"{input_storage_ML}/{run_name}") # TODO Not sure I need separate input/model storage since the config/preprocessing is the same for the run and only the post-model building is different and that's in seperate output folders, but combine later if needed
model_storage_NN = Path(f"{input_storage_NN}/{run_name}")
os.makedirs(model_storage_prelim, exist_ok=True)
if build_ML:
    os.makedirs(model_storage_ML, exist_ok=True)
if build_NN:
    os.makedirs(model_storage_NN, exist_ok=True)

# Copy config folder to input storage folder and rename
config_new_ML = f"{model_storage_ML}/config.yaml"
config_new_NN = f"{model_storage_NN}/config.yaml"
if build_ML:
    shutil.copy2(config_path, config_new_ML)
if build_NN:
    shutil.copy2(config_path, config_new_NN)

# Update run number in base config - done ASAP to allow overlapping runs to access unique IDs
  # Note: as long as runs aren't started within the same second, they are guaranteed to be unique due to the timestamp, even if run number isn't updated
config["general"]["run_number"] = run_number + 1 # Update run number for next run
with open(config_path, "w") as f:
    yaml.dump(config, f, sort_keys=False)

### RUN PRELIMINARY SCRIPTS #############################################################################################
# Run feature_engineering.py - this script takes the initial data and cleans/engineers features to prepare for the
  # preprocessing required for model construction.
process = subprocess.run(['python', 'feature_engineering.py', '--run_name', run_name, '--from_pipeline'],text=True)
# Check if failed
if process.returncode != 0:
    print("Warning: feature_engineering.py failed")
    exit(1)

# # Run data_preprocessing.py - this script splits, normalises, encodes, and imputes the data to prepare the datasets for
# # model building.
# process = subprocess.run(['python', 'data_preprocessing.py', '--run_name', run_name, '--from_pipeline'],text=True)
# # Check if failed
# if process.returncode != 0:
#     print("Warning: data_preprocessing.py failed")
#     exit(1)

### MOVE FILES TO RELEVANT MODEL FOLDER ################################################################################
# In order to use the exact same data for preliminary vs ML models, the data needs to be copied into the relevant folder
if build_ML:
    shutil.copytree(model_storage_prelim, model_storage_ML, dirs_exist_ok=True)
if build_NN:
    shutil.copytree(model_storage_prelim, model_storage_NN, dirs_exist_ok=True)

# Delete original input folder now it's been copied
print("Deleting:", model_storage_prelim)
shutil.rmtree(model_storage_prelim)

#### RUN TRADITIONAL MODEL SCRIPTS #####################################################################################
if build_ML:
    # Run model_building.py - builds the model using the Surrey data
    process = subprocess.run(['python', 'model_building.py', '--run_name', run_name, '--from_pipeline'],text=True)
    # Check if failed
    if process.returncode != 0:
        print("Warning: model_building.py failed")
        exit(1)

    # Check whether to validate
    validate = config["general"]["validate"]

    if validate:
        # Run external_validation.py - fits the model to the validation data
        process = subprocess.run(['python', 'external_validation.py', '--run_name', run_name, '--from_pipeline'],text=True)
        # Check if failed
        if process.returncode != 0:
            print("Warning: external_validation.py failed")
            exit(1)

### RUN NEURAL NETWORK SCRIPTS #########################################################################################
if build_NN:
    # Run neural_network.py - builds the model using the Surrey data
    process = subprocess.run(['python', 'neural_network.py', '--run_name', run_name, '--from_pipeline'], text=True)
    # Check if failed
    if process.returncode != 0:
        print("Warning: neural_network.py failed")
        exit(1)

    # # Check whether to validate # todo adapt validation to nn (could run inside other script if small)
    # validate = config["general"]["validate"]
    #
    # if validate:
    #     # Run external_validation.py - fits the model to the validation data
    #     process = subprocess.run(['python', 'external_validation.py', '--run_name', run_name, '--from_pipeline'], text=True)
    #     # Check if failed
    #     if process.returncode != 0:
    #         print("Warning: external_validation.py failed")
    #         exit(1)