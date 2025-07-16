### SCRIPT USAGE #######################################################################################################
# Run this script to build and run a neural network to pedict oxygen need (O2 Req.).

### SETUP ##############################################################################################################
from datasets import load_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
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
import yaml
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import warnings
from functions import basic_train, IntToFloatTransformer, port_in_use, pca_original, plot_learning_curve, \
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

# Bool to show additional detail
show_detail = False

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

#### READ CONFIG FILE ##################################################################################################
# Set config path based on whether the script is run standlone or part of pipeline.py (config moved to 'inputs')
if not args.from_pipeline:
    config_path = Path("config.yaml")
else:
    config_path = Path(f"inputs/NN/{run_name}/config.yaml")

# Read config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set parameters for this file: #todo copied pasted from ML, check if used at end - some def need changing for NNs
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

#todo NN specific ones below - keep all these
batch_size = config['neural_network']['batch_size'] # Batch size to use for the neural net
n_epochs = config['neural_network']['n_epochs'] # How many epochs to run
nth_epoch = config['neural_network']['nth_epoch'] # Every nth epoch, print the loss

# Determine which models to test (set in config file)
# todo

### READ IN DATA #######################################################################################################
# Set pandas to display all columns and longer rows # IMPROVE remove in final version
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

# Create output directories for the data
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    data_dir = 'training_data' # Combine with other training graphs if using training data
else: # Put into input storage folder to prevent overwriting
    data_dir = f'inputs/NN/{run_name}/training_data'

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    graphs_dir = 'training_graphs' # Combine with other training graphs if using training data
else: # Put into input storage folder to prevent overwriting
    graphs_dir = f'inputs/NN/{run_name}/training_graphs'

### Read in data
# Train
X_path = Path(__file__).parent / data_dir / "Surrey_X_train.csv"
y_path = Path(__file__).parent / data_dir / "Surrey_y_train.csv"
X_train_df = pd.read_csv(X_path, index_col=0)
y_train_df = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Test
X_path = Path(__file__).parent / data_dir / "Surrey_X_test.csv"
y_path = Path(__file__).parent / data_dir / "Surrey_y_test.csv"
X_test_df = pd.read_csv(X_path, index_col=0)
y_test_df = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array

### PCA ON ORIGINAL DATA ###############################################################################################
try:
    # Combine train and test
    X_full = pd.concat([X_train_df, X_test_df]).values
    y_full = np.concatenate([y_train_df, y_test_df])

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
 # IMPROVE see model_building.py for note on variance thresholding
# Calculate median variance of all features
variances = X_train_df.var(axis=0)
threshold = float(var_threshold) # Effectively zero but avoids floating-point issues

# Note: Skipped VIF analysis for NN

### DEFINE FEATURE SELECTION PER MODEL #################################################################################
### Feature selection methods taken from scikit-learn documentation
# Dictionary of feature selector options. base_params are fixed parameters that also apply to basic_train, with other parameters tunable later in the search space
feature_selectors_all = {
    # RFECV with Logistic Regression
    'RFECV_LR': {
        'class': RFECV,
        'base_params': {
            'estimator': LogisticRegression(),
            'step': 1,
            'cv': StratifiedKFold(5),
            'scoring': "f1",
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
            'scoring': "f1",
            'min_features_to_select': 50,
        }
    },
    # RFECV with Random Forest
    'RFECV_RF': {
        'class': RFECV,
        'base_params': {
            'estimator': RandomForestClassifier(random_state=42),
            'step': 1,
            'cv': StratifiedKFold(5),
            'scoring': "f1",
            'min_features_to_select': 50,
        }
    },
    # RFECV with XGBoost
    'RFECV_XGB': {
        'class': RFECV,
        'base_params': {
            'estimator': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='logloss', random_state=42),
            'step': 1,
            'cv': StratifiedKFold(5),
            'scoring': "f1",
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
            'estimator': LinearSVC(),
            'n_features_to_select': 'auto',
            'tol': 0.01,
        }
    },
    # Sequential Feature Selection with XGBoost
    'SFS_XGB': {
        'class': SequentialFeatureSelector,
        'base_params': {
            'estimator': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                       eval_metric='logloss', random_state=42),
            'n_features_to_select': 'auto',
            'tol': 0.01,
        }
    },
    # No feature selection
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

# TODO these FS are currently not implemented; i need a basic_train equivalent. But I do want to do a simple train vs hyper train as with TML









### TODO ADAPT TML FRAMEWORK TO NN AS REQUIRED - BUT FIRST MAKE BASIC NEURAL NET

### CREATE DATALOADERS #################################################################################################
# Convert X data to PyTorch tensors
X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
X_test = torch.tensor(X_test_df.values, dtype=torch.float32)

# Convert labels and ensure proper shape (64-bit integer to 1D label tensor)
y_train = torch.tensor(y_train_df.values, dtype=torch.long).squeeze()
y_test = torch.tensor(y_test_df.values, dtype=torch.long).squeeze()

# Create TensorDatasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
print(f"Feature dimensions: {X_train.shape[1]} | Classes: {y_train.unique().size(0)}")

### DEFINE NEURAL NETWORK ##############################################################################################

class O2Classifier(nn.Module): # TODO decide/hyperopt layer #/activation function/# neurons/anything else
    def __init__(self, input_dim): # Initialisation - input number of numerical features
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected (dense) layer with 128 neurons.
        self.relu = nn.ReLU()  # Activation function to add non-linearity.
        self.fc2 = nn.Linear(128, 1)  # Output layer outputting the probability of class 1

    def forward(self, input): # Forward pass
        # Pass through the network
        x = self.fc1(input) # Passes the pooled embedding through the first dense layer.
        x = self.relu(x) # Applies the ReLU activation function
        output = self.fc2(x) # Outputs the logits for the number of classes.
        return output


# Initialize the model
input_dim = X_train.shape[1]
model = O2Classifier(input_dim)

### DEFINE OPTIMISER AND LOSS FUNCTION #################################################################################
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with built-in sigmoid
optimiser = torch.optim.Adam(model.parameters(), lr=5e-4) # Updates the model’s parameters to minimize the loss function

### DEFINE THE TRAINING LOOP ###########################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Automatically uses a GPU if available; otherwise, defaults to the CPU
model.to(device) # Moves the model to the selected device for computation

num_epochs = n_epochs
for epoch in range(num_epochs):
    model.train() # Sets the model to training mode, enabling operations like dropout (if present)
    total_loss = 0
    for batch in train_loader:
        # Move the batch data to the same device as the model (GPU or CPU)
        features = batch[0].to(device) # Selects input features #todo understand
        labels = batch[1].to(device).float()  # Selects labels and converts #todo do i need to convert?

        # Forward pass
        outputs = model(features).squeeze() # todo why squeeze
        loss = criterion(outputs, labels) # Calculates the classification error between predictions (outputs) and true labels (labels) using the cross-entropy loss

        # Backward pass and optimization
        optimiser.zero_grad() # Resets gradients from the previous iteration to prevent accumulation
        loss.backward() # Computes the gradients of the loss with respect to the model parameters via backpropagation
        optimiser.step() # Updates the model parameters using the computed gradients

        total_loss += loss.item() # Accumulates the total loss for the epoch to monitor training progress

    if (epoch == 0) or ((epoch + 1) % nth_epoch == 0) or (epoch == num_epochs - 1): # Print every nth epoch or first/last epoch
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}") # Logs the average loss per epoch to track improvement

### EVALUATE THE MODEL #################################################################################################
model.eval() # Puts the model in evaluation mode
correct = 0
total = 0

with torch.no_grad(): # Disables gradient computation to save memory and speed up evaluation
    for batch in test_loader:
        features = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model(features).squeeze() # Passes the input IDs through the model to compute logits (unnormalized scores for each class)
        probabilities = torch.sigmoid(outputs) # Convert to probabilities
        predictions = (probabilities > 0.5).float()
        correct += (predictions == labels).sum().item() # Counts the correct predictions in the current batch
        total += labels.size(0) # Tracks the total samples processed

accuracy = correct / total # Computes overall accuracy as the ratio of correct predictions to total samples
print(f"Test Accuracy: {accuracy:.3f}")

### LOAD IN VALIDATION DATA ############################################################################################
# Set input directories
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    data_dir = 'validation_data' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    data_dir = f'inputs/NN/{run_name}/validation_data'

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    graphs_dir = 'validation_graphs' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    graphs_dir = f'inputs/NN/{run_name}/validation_graphs'

dataset = "ISARIC"
X_path = Path(__file__).parent / data_dir / f"{dataset}_X.csv"
y_path = Path(__file__).parent / data_dir / f"{dataset}_y.csv"
X_data_df = pd.read_csv(X_path, index_col=0)
y_data_df = pd.read_csv(y_path, index_col=0)

### PREPARE DATA #######################################################################################################
# Get columns from X_train # todo temporary measure to filter to same features - see TML ex val for final example once full structure is implemented
X_train_col = X_train_df.columns
X_data_filtered = X_data_df[X_train_col]
# Ensure same column order
X_data_df = X_data_filtered.reindex(columns=X_train_col)

# TODO this is done differently for TML so can be changed if separated into scripts/feature selected in this script - cwd will be FS-ed, but wrapper script running is not (yet)

### CREATE DATALOADERS #################################################################################################
# Convert X and ydata to PyTorch tensors
X_data = torch.tensor(X_data_df.values, dtype=torch.float32)
y_data = torch.tensor(y_data_df.values, dtype=torch.long).squeeze()

# Create TensorDatasets
val_dataset = TensorDataset(X_data, y_data)

# Create DataLoaders
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # TODO can I change batch size since val is a lot smaller?

print(f"Validation samples: {len(val_dataset)}")
print(f"Feature dimensions: {X_data.shape[1]} | Classes: {y_data.unique().size(0)}")

### RUN MODEL ON NEW DATA ##############################################################################################
if validate:
    def classify_O2(model, val_loader):
        model.eval()
        correct = 0
        total = 0
        all_probabilities = []
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(features).squeeze()
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                # Collect results
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                # Store results for analysis todo
                all_probabilities.extend(probabilities.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = correct / total
        return accuracy, all_probabilities, all_predictions, all_labels

    # Apply model to validation dataset
    val_accuracy, probabilities, predictions, true_labels = classify_O2(model, val_loader)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # TODO AI GEN TEMP GRAPHS
    # Convert to class predictions and labels (to binary numbers from np.floats)
    class_predictions = [1 if p > 0.5 else 0 for p in probabilities] # Predicted values
    class_labels = [int(label) for label in true_labels] # True labels
    print("class_predictions:\n", class_predictions)
    print("class_labels:\n", class_labels)


    # Confusion matrix
    cm = confusion_matrix(class_labels, class_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(class_labels, class_predictions, target_names=["No O2", "O2"]))
    ##########




# TODO
#  Needs to be updated for TML model framework; current version is a basic version to test concept - including validation file

# Adjust hyperparams:
    # learning rate (smaller = better but more computational)
    # batch size (larger = smoother gradients more RAM, smaller = may generalise better)
    # epochs (better learning but watch for overfitting (monitor validation performance))
# Model architecture:
    # More layers is more complex patterns but require more data and careful regularization
    # Activation Functions: Experiment with ReLU, LeakyReLU, or GELU for better non-linearity and learning dynamics
# Overfitting vs underfitting:
    # If performs well on test but not train/val: Add dropout, use regularization (e.g., L2), or gather more data
    # If failing to capture patterns in training: Add layers, increase training time, or adjust hyperparameters

# Batch size and epoch number can be chosen by trial and error (or function?)

# Needed for my NN:
    # This data doesn't have headers, which idk if I can preserve or load later
    # How many layers do I need? 'heuristics or copy others'
    # In forward you can: skip connections, attention mechanisms, use conditionals, and do multiple inputs or outputs. Likely more.

# Add regularization (dropout, weight decay)
# Use learning rate schedulers
# Cross-validation
# Tune architecture and hyperparameters





























# TODO
# Adjust hyperparams:
    # learning rate (smaller = better but more computational)
    # batch size (larger = smoother gradients more RAM, smaller = may generalise better)
    # epochs (better learning but watch for overfitting (monitor validation performance))
# Model architecture:
    # More layers is more complex patterns but require more data and careful regularization
    # Activation Functions: Experiment with ReLU, LeakyReLU, or GELU for better non-linearity and learning dynamics
# Overfitting vs underfitting:
    # If performs well on test but not train/val: Add dropout, use regularization (e.g., L2), or gather more data
    # If failing to capture patterns in training: Add layers, increase training time, or adjust hyperparameters

# Batch size and epoch number can be chosen by trial and error (or function?)

# Needed for my NN:
    # This data doesn't have headers, which idk if I can preserve or load later
    # How many layers do I need? 'heuristics or copy others'
    # In forward you can: skip connections, attention mechanisms, use conditionals, and do multiple inputs or outputs. Likely more.

# Add regularization (dropout, weight decay)
# Use learning rate schedulers
# Cross-validation
# Tune architecture and hyperparameters