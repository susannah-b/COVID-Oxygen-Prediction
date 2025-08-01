### SCRIPT USAGE #######################################################################################################
# Run this script to build and run a neural network to pedict oxygen need (O2 Req.).

### SETUP ##############################################################################################################
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFECV, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
import mlflow
import yaml
import mlflow.sklearn
import matplotlib.pyplot as plt
from functions import port_in_use, pca_pre_post_fs, plot_learning_curve, \
    plot_roc_auc, plot_calibration_curve, plot_precision_recall, \
    plot_pca_predicted, plot_confusion_matrix, plot_decision_distribution, remaining_meta, grouped_shap, \
    plot_pca_original, plot_pca_test_unprocessed, set_graph_style
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
import shap
from mlflow.tracking import MlflowClient

# todo clean up at end

# Bool to show additional detail
show_detail = False

# Apply graph styles
set_graph_style()

### SET RANDOM SEEDS ###################################################################################################
# Function to set random seeds - certain operations advance random state, and due to the small dataset I've found better results (on both test and exval) with certain seeds
def reset_seeds(seed=44): #TODO can actually be hypertuned
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
host = config['general']['host'] # Host for tracking server
port = config['general']['port'] # Port for local tracking server
selector_type = config['neural_network']['feature_selection']['selector'] # Feature selector used
max_evals = config['model_building']['max_evals'] # How many evaluations to do in hyperopt tuning
track_final = config['model_building']['track_final'] # Whether to copy the model_output to the designated folder for easier browsing
batch_size = config['neural_network']['batch_size'] # Batch size to use for the neural net
n_epochs = config['neural_network']['n_epochs'] # How many epochs to run
nth_epoch = config['neural_network']['nth_epoch'] # Every nth epoch, print the loss
learn_rate = float(config['neural_network']['learning_rate']) # Learning rate for NN #todo hypertune?
early_stopping = config['neural_network']['early_stopping'] # Whether to implement early stopping
validation_size = config['neural_network']['validation_size'] # Proportional size of the validation set for cross validation/early stopping
use_set_search_space = config['neural_network']['use_set_search_space']
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
    data_dir = f'inputs/NN/{run_name}/training_data'

# Create output directories for the data
output_data_dir = f'{data_dir}/NN'
os.makedirs(output_data_dir, exist_ok=True)

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    graphs_dir = 'training_graphs/NN' # Combine with other training graphs if using training data
else: # Put into input storage folder to prevent overwriting
    graphs_dir = f'inputs/NN/{run_name}/training_graphs/NN'
os.makedirs(graphs_dir, exist_ok=True) # Make the ML graph that's specific to the ML outputs

### Read in data
# Train
X_path = Path(__file__).parent / data_dir / "Surrey_X_train.csv"
y_path = Path(__file__).parent / data_dir / "Surrey_y_train.csv"
X_train_full = pd.read_csv(X_path, index_col=0)
y_train_full = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Test
X_path = Path(__file__).parent / data_dir / "Surrey_X_test.csv"
y_path = Path(__file__).parent / data_dir / "Surrey_y_test.csv"
X_test = pd.read_csv(X_path, index_col=0)
y_test = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array

# Convert y to float32 for pytorch
y_train = y_train_full.astype(np.float32)
y_test = y_test.astype(np.float32)

# Print summary
print(f"Total training samples: {len(X_train_full)} | Test samples: {len(X_test)}")
print(f"Feature dimensions: {X_train_full.shape[1]} | Classes: {y_train.nunique()}\n")

# Do test/validation split for the early stopping check (final model is trained on X_train_full)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=validation_size, stratify=y_train_full, random_state=42)
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu" # Automatically uses a GPU if available; otherwise, defaults to the CPU

#TODO check that X_train is now correct elsewhere in code, post 2nd split (eg for graphs, anytihng that required len(X_train)? - is pca right?
### PCA ON ORIGINAL DATA ###############################################################################################
# Full dataset
plot_pca_original(X_train_full, X_test, y_train_full, y_test, graphs_dir)
# Test only (to compare to the post-processed before and after predictions graphs)
plot_pca_test_unprocessed(X_test, y_test, graphs_dir)

### VARIANCE THRESHOLDING ##############################################################################################
  # Applied in the scikit-learn pipeline

# Calculate median variance of all features
variances = X_train.var(axis=0)
threshold = float(var_threshold) # Effectively zero but avoids floating-point issues

# Note: Skipped VIF analysis for NN

### CONVERT INTEGER COLUMNS TO FLOAT ###################################################################################
  # Safely handles missing values
class IntToFloatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Only convert if DataFrame (preserves column names)
        if isinstance(X, pd.DataFrame):
            int_cols = X.select_dtypes(include=['int', 'int32', 'int64']).columns
            X.loc[:, int_cols] = X.loc[:, int_cols].astype(float)
            # Convert to float32 from pandas float64 for pytorch
            float64_cols = X.select_dtypes(include=['float64']).columns
            X.loc[:, float64_cols] = X.loc[:, float64_cols].astype(np.float32)
        return X

### DEFINE FEATURE SELECTION PER MODEL #################################################################################
### Feature selection methods taken from scikit-learn documentation
# Dictionary of feature selector options. base_params are fixed parameters, with other parameters tunable later in the search space
feature_selectors = {
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
    # RFECV with Random Forest
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
    # RFECV with XGBoost
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

# IMPROVE: for the neural set FS methods are simply specified vs automatically evaluated, due to time constraints during
#  development. Ideally, this would also be implemented for NNs.

### ESTIMATE BEST NN MODEL WITH BASIC SETTINGS #########################################################################
# IMPROVE: The 'basic_train' which evaluates a few different model versions as with the traditional ML model is not yet
#  implemented. WOuld be interesting to automate layer/activation fucntion/etc experimentation.

### DEFINE NEURAL NETWORK ##############################################################################################
def create_model(params, input_dim):
    class DynamicO2Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, int(params['layer1']))
            self.fc2 = nn.Linear(int(params['layer1']), int(params['layer2']))
            self.fc3 = nn.Linear(int(params['layer2']), int(params['layer3']))
            self.fc4 = nn.Linear(int(params['layer3']), 1)

            # Activation selection
            activation_name = params['activation']
            if activation_name == 'leaky_relu':
                self.act = nn.LeakyReLU(0.01)
            elif activation_name == 'mish':
                self.act = nn.Mish()
            elif activation_name == 'sigmoid':
                self.act = nn.Sigmoid()
            elif activation_name == 'softplus':
                self.act = nn.Softplus()
            elif activation_name == 'softsign':
                self.act = nn.Softsign()
            elif activation_name == 'selu':
                self.act = nn.SELU()
            elif activation_name == 'elu':
                self.act = nn.ELU()
            else:  # Default to ReLU
                self.act = nn.ReLU()

            self.dropout = nn.Dropout(params['dropout'])

        def forward(self, x):
            x = self.dropout(x)
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            return self.fc4(x).squeeze(1)

    return DynamicO2Classifier()

### EARLY STOPPING #####################################################################################################
# Basic setup for early stopping criteria
patience = 5  # Epochs to wait after no improvement
delta = 0.01  # Minimum change in the loss
best_val_loss = float("inf")  # Best validation loss to compare against
no_improvement_count = 0 # Count epochs since improvement

# Define class
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:  #If loss is improving
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else: # Track duration without improvement and trigger break
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

### FUNCTION TO TRAIN FINAL MODEL ######################################################################################
# Includes options for cross validation and early stopping
# todo move this to functions & imprve structure, eg final train bool could also turn of ES etc, validation set should be set up better instead of requiring/returning none
# Function to train the model - applied to both cross validation and the final model with no validation/early stopping
def train_model(model, train_loader, val_loader, es_handler, optimiser, verbose=1):
    all_probs = []
    all_labels = []
    auc_roc = 0 # Initiliase so when training without calculating auc_roc, it will return a value instead of erroring
    epoch_stop = n_epochs  # Initialise as max range of epochs
    training_losses = []
    validation_losses = []
    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()  # Sets the model to training mode, enabling operations like dropout (if present)
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move the batch data to the same device as the model (GPU or CPU)
            features = batch[0].to(device)  # Selects input features
            labels = batch[1].to(device)  # Selects labels

            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs,
                             labels)  # Calculates the classification error between predictions (outputs) and true labels (labels) using the cross-entropy loss

            # Backward pass and optimisation
            optimiser.zero_grad()  # Resets gradients from the previous iteration to prevent accumulation
            loss.backward()  # Computes the gradients of the loss with respect to the model parameters via backpropagation
            optimiser.step()  # Updates the model parameters using the computed gradients

            train_loss += loss.item()  # Accumulates the total loss for the epoch to monitor training progress

        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Print loss
        if verbose:
            if (epoch == 0) or ((epoch + 1) % nth_epoch == 0) or (
                    epoch == n_epochs - 1):  # Print every nth epoch or first/last epoch
                print(
                    f"Epoch {epoch + 1}, Loss: {avg_train_loss}")  # Logs the average loss per epoch to track improvement

        # Validation phase - for CV only
        if es_handler is not None:  # Run early stopping
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch[0].to(device)
                    labels = batch[1].to(device)
                    output = model(features).squeeze()
                    loss = criterion(output, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            if early_stopping:
                # Check early stopping condition
                es_handler.check_early_stop(avg_val_loss)
                if es_handler.stop_training:
                    print(f"Early stopping at epoch {epoch}")
                    epoch_stop = epoch + 1
                    break

        # Calculate final AUROC score on validation set when supplied
        if val_loader is not None:
            model.eval()
            val_loss = 0 #IMPROVE - for loss curve this is calculated again here (same as for early stopping check above) in the cross validation loop - function needs to be condensed to be more efficient!
            with torch.no_grad():
                for batch in val_loader:
                    features = batch[0].to(device)
                    labels = batch[1].to(device)
                    output = model(features).squeeze() # Raw outputs from the model
                    loss = criterion(output, labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(output)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            # Calculate AUROC
            auc_roc = roc_auc_score(all_labels, all_probs)
            # Store validation losses for loss curve
            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss) # Stores loss per epoch

    if val_loader is not None: # If supplying a validation set, ie for ES check or CV
        return auc_roc, epoch_stop, training_losses, validation_losses
    else:
        return auc_roc, epoch_stop

criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with built-in sigmoid

### OBJECTIVE FUNCTION FOR HYPEROPT PARAMETER TUNING ###################################################################

# Define classifier type
  # IMPROVE: This is used to be similar in structure to the traditional model; could be used after a similar basic_train
  #  function is implemented. For now, just stores best result for the defined model
classifier_type = 'neural_network'

# Dictionary to store the best model accuracies
best_roc = {
    'neural_network': 0.0,
}

def objective(params):
    # Set feature selector and parameters based on classifier type
    fs_params = params.pop('fs_params', {})  # Remove FS params from classifier search space
    # Get selector configuration
    selector_config = feature_selectors[selector_type]

    # Check for invalid layer setup
    if params['layer1'] < params['layer2'] or params['layer2'] < params['layer3']:
        return {'loss': 0.0, 'status': STATUS_FAIL}

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

    # Start MLFlow run for each trial
    with mlflow.start_run(nested=True):
        mlflow.set_tag("Corresponding final model",f"{run_name}")  # Can find the name of the trained model using this tag
        # Log trial hyperparameters
        mlflow.log_params({**params, "type": classifier_type, **{"fs_" + k: v for k, v in fs_params.items()}})

        # Incorporate required preprocessing/FS steps and the model
        hp_preprocessor = Pipeline([
            ('int_to_float', IntToFloatTransformer()),
            ('var_thresh', VarianceThreshold(threshold=0.0)),
            ('scaler', StandardScaler()),
            ('feature_selector', selector if feature_selection else 'passthrough')
        ])

        # Call function to train model #IMPROVE - exact copy of CV script - more elegant way to implement this eg function
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Set to three due to small dataset sizes
        roc_scores = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):

            # Create datasets for this fold
            X_train_hp = X_train.iloc[train_idx]
            y_train_hp = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            X_val_hp = X_train.iloc[val_idx]
            y_val_hp = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

            # Fit preprocessor
            X_train_hp_processed = hp_preprocessor.fit_transform(X_train_hp, y_train_hp)
            X_val_hp_processed = hp_preprocessor.transform(X_val_hp)

            # Create data loaders for this fold
            train_dataset_hp = TensorDataset(
                torch.FloatTensor(X_train_hp_processed),
                torch.FloatTensor(y_train_hp.values if hasattr(y_train_hp, 'values') else y_train_hp)
            )
            val_dataset_hp = TensorDataset(
                torch.FloatTensor(X_val_hp_processed),
                torch.FloatTensor(y_val_hp.values if hasattr(y_val_hp, 'values') else y_val_hp)
            )

            train_loader_hp = DataLoader(train_dataset_hp, batch_size=int(params['batch_size']), shuffle=True)
            val_loader_hp = DataLoader(val_dataset_hp, batch_size=12, shuffle=False)  # todo batch size - will also error if it creates a size of 1 at any point

            # Initialize fresh model and early stopping for this fold (note early stopping is only determined for the final model in the cross validation section and not stored for this iteration)
            input_dim_hp = X_train_hp_processed.shape[1]  # Get input dim from processed data

            # Create model
            model = create_model(params, input_dim_hp)
            model.to(device)

            # Optimiser setup
            opt_class = {
                'Adam': torch.optim.Adam,
                'SGD': torch.optim.SGD,
                'RMSprop': torch.optim.RMSprop,
                'Adagrad': torch.optim.Adagrad,
                'Adamax': torch.optim.Adamax,
                'Nadam': torch.optim.NAdam,
            }[params['optimiser']]

            optimiser = opt_class(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

            # Reset early stopping for this fold
            if early_stopping:
                es_handler = EarlyStopping(patience=patience, delta=delta, verbose=False)
            else:
                es_handler = None
            # Train and get AUROC score #IMPROVE stores returned training and validation loss but these aren't used elsewhere - training function needs tidying
            roc, epoch_stop, tl, vl = train_model(model, train_loader_hp, val_loader_hp, es_handler=None, optimiser=optimiser, verbose=False)
            roc_scores.append(roc)

        # Evaluate the model after CV with AUROC
        roc_score_mean = np.mean(roc_scores).astype(np.float32)

        # Log the best AUROC for each model type if improved
        if roc_score_mean > best_roc[classifier_type]:
            best_roc[classifier_type] = roc_score_mean
            mlflow.log_metric(f"best_{classifier_type}_AUROC", roc_score_mean)

    # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
    return {'loss': -roc_score_mean, 'status': STATUS_OK}

### DEFINE SEARCH SPACES PER FEATURE SELECTOR ##########################################################################
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
    'NONE': {

    }

}

### DEFINE SEARCH SPACE FOR THE MODEL ##################################################################################
search_space = {
    # Layer sizes: use stepwise reduction
    'layer1': hp.quniform('layer1', 64, 512, 32),
    'layer2': hp.quniform('layer2', 32, 256, 16),
    'layer3': hp.quniform('layer3', 16, 128, 8),

    # Regularisation
    'dropout': hp.uniform('dropout', 0.0, 0.8),
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-2)),

    # Optimisation
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(0.1)),
    'optimiser': hp.choice('optimiser', ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adamax', 'Nadam']),
    'batch_size': hp.choice('batch_size', [64, 48, 32]),

    # Activation
    'activation': hp.choice('activation', ['relu', 'leaky_relu', 'mish', 'sigmoid', 'softplus', 'softsign',
        'selu', 'elu' ]),

    # Feature selection
    'fs_params': selector_param_spaces[selector_type]

}

fixed_search_space = {
    'layer1': 256,
    'layer2': 128,
    'layer3': 64,
    'dropout': 0.7,
    'weight_decay': 0,
    'lr': 5e-4,
    'optimiser': 'Adam',
    'batch_size': 32,
    # 'epochs': 300, # Removed epochs term from search space, but despite being probably too high this had good results
    'activation': 'mish',
    'fs_params': selector_param_spaces[selector_type]
}
# Set search space to use - used to bypass hyperopt and test fixed parameters
  # IMPROVE: This was implemented because the results from hyperopt weren't exceed my results with set parameters - ideally resolve this (e.g. changing input params, perhaps how hyperopt evaluates 'best' model, or editing/removing CV in hyperopt)
if use_set_search_space:
    search_space = fixed_search_space
    max_evals = 1

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

mlflow.set_experiment("Oxygen Prediction NN - Hyperparams") # Note: Could use same experiment ID as the final model in order to compare; for now I find it easier to keep them separate.
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
    store_hyp_id = f"Run {run_name} for hyperparameter training completed. Run ID is {hyper_run_id}. See nested runs for individual trials"

# Print the best accuracies for each model type
print("\nHighest network AUROC on train data:")
best_roc_df = pd.DataFrame(list(best_roc.items()), columns=['Models', 'Highest AUROC'])
print(best_roc_df)

# Extract and print the best hyperparameter configuration
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

# Define tuned classifier
class O2Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, int(best_config['layer1']))
        self.fc2 = nn.Linear(int(best_config['layer1']), int(best_config['layer2']))
        self.fc3 = nn.Linear(int(best_config['layer2']), int(best_config['layer3']))
        self.fc4 = nn.Linear(int(best_config['layer3']), 1)
        self.dropout = nn.Dropout(best_config['dropout'])

        # Instantiate the activation function
        activation_name = best_config['activation']
        if activation_name == 'leaky_relu':
            self.act = nn.LeakyReLU(0.01)
        elif activation_name == 'mish':
            self.act = nn.Mish()
        elif activation_name == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation_name == 'softplus':
            self.act = nn.Softplus()
        elif activation_name == 'softsign':
            self.act = nn.Softsign()
        elif activation_name == 'selu':
            self.act = nn.SELU()
        elif activation_name == 'elu':
            self.act = nn.ELU()
        else:  # Default to ReLU
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x).squeeze(1)

### CREATE WRAPPER FOR MODEL ###########################################################################################
# Due to the combined sklearn and pytorch elements of the pipeline (preprocessing and NN model), the model is required to
#  be logged as pyfunc, not sklearn or pytorch. It therefore has no .predict attribute needed for SHAP KernelExplainer.
#TODO - can i just use the model as wrapped by pyfunc? or does it need to be loaded in after being logged

# Wrapper function to convert numpy input to pytorch tensors
def model_predict(X):
    # Convert numpy array to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Make predictions
    final_model.eval()
    with torch.no_grad():
        logits = final_model(X_tensor)
        # Return probabilities (sigmoid output)
        return torch.sigmoid(logits).numpy()

### TRAIN FINAL MODEL IN MLFLOW ########################################################################################
#todo many parts were deleted from TML framework, so copy in later
# todo edit code comment blocks/move around

# Create a new MLflow Experiment
if enable_tracking: # Have to use a unique name or it creates issues with artifact tracking
    exp_name = "Oxygen Prediction Neural Network - Surrey"
else:
    exp_name = "Oxygen Prediction Neural Network - Surrey - Offline"

artifact_path = f"mlartifacts" #todo what happens with multiple runs
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
mlflow.pytorch.autolog()
store_final_id = None # Initialise value to store run ID to print at end
final_run_id = None
final_exp_id = None
# Start MLFlow run to track
with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("Run name", run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("ML type", "Neural network")
    mlflow.set_tag("Phase", "Final model training")
    mlflow.set_tag("Hyperopt MLflow run", hyperopt_name)
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    final_exp_id = run.info.experiment_id # Get experiment id for folder management

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
    selector_config = feature_selectors[selector_type]

    # Build selector
    if selector_type == 'NONE':
        selector = 'passthrough'
    else:
        all_params = {**selector_config['base_params'], **fs_params}
        selector = selector_config['class'](**all_params)

    # Log the best hyperparameters
    mlflow.log_params(best_config)

    # Create the preprocessing pipeline
    preprocessor = Pipeline([
        ('int_to_float', IntToFloatTransformer()),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler()),
        ('feature_selector', selector if feature_selection else 'passthrough')
    ])

    # Fit preprocessor on full training data
    preprocessor.fit(X_train_full, y_train_full)
    # Transform all datasets using the preprocessor
    X_train_original = X_train.copy() # Store original data
    X_train_full_processed = preprocessor.transform(X_train_full)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    ### CREATE DATALOADERS #################################################################################################
    # Convert X data to PyTorch tensors #todo for this section, not sure if these are reassigned. eg X_train i think isn't used but X_full might be?
    X_train_dl = torch.tensor(X_train_processed, dtype=torch.float32)
    X_train_full_dl = torch.tensor(X_train_full_processed, dtype=torch.float32)
    X_val_dl = torch.tensor(X_val_processed, dtype=torch.float32)
    X_test_dl = torch.tensor(X_test_processed, dtype=torch.float32)

    # Convert labels and ensure proper shape (64-bit integer to 1D label tensor)
    y_train_dl = torch.tensor(y_train.values, dtype=torch.float32).squeeze()
    y_train_full_dl = torch.tensor(y_train_full.values, dtype=torch.float32).squeeze()
    y_val_dl = torch.tensor(y_val.values, dtype=torch.float32).squeeze()
    y_test_dl = torch.tensor(y_test.values, dtype=torch.float32).squeeze()

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_dl, y_train_dl)
    train_full_dataset = TensorDataset(X_train_full_dl, y_train_full_dl)
    val_dataset = TensorDataset(X_val_dl, y_val_dl)
    val_sample_len = len(val_dataset)
    test_dataset = TensorDataset(X_test_dl, y_test_dl)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_full_loader = DataLoader(train_full_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False) #TODO custom batch size or hyperopt

    print(f"Training samples: {len(train_dataset)} | Validation samples: {val_sample_len} | Test samples: {len(test_dataset)}")
    print(f"Feature dimensions: {X_train.shape[1]} | Classes: {len(np.unique(y_train))}")

    ### CROSS-VALIDATE #################################################################################################
    # Call function to train model # WARNING this is technically a different model to the final model - worth keeping?
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Set to three due to small dataset sizes
    roc_scores = []
    stopping_epochs = [] # best epochs per fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):

        # Create datasets for this fold
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

        # Clone preprocessor and fit
        preprocessor_fold = clone(preprocessor)  # Create a fresh copy to avoid corrupting original preprocessor
        X_train_fold_processed = preprocessor_fold.fit_transform(X_train_fold, y_train_fold)
        X_val_fold_processed = preprocessor_fold.transform(X_val_fold)

        # Create data loaders for this fold
        train_dataset_fold = TensorDataset(
            torch.FloatTensor(X_train_fold_processed),
            torch.FloatTensor(y_train_fold.values if hasattr(y_train_fold, 'values') else y_train_fold)
        )
        val_dataset_fold = TensorDataset(
            torch.FloatTensor(X_val_fold_processed),
            torch.FloatTensor(y_val_fold.values if hasattr(y_val_fold, 'values') else y_val_fold)
        )

        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=12, shuffle=False) # todo batch size - will also error if it creates a size of 1 at any point

        # Initialize fresh model and early stopping for this fold
        input_dim_fold = X_train_fold_processed.shape[1]  # Get input dim from processed data
        model = O2Classifier(input_dim_fold).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=5e-4)

        # Reset early stopping for this fold
        if early_stopping:
            es_handler = EarlyStopping(patience=patience, delta=delta, verbose=True)
        else:
            es_handler = None

        # Train and get AUROC score/losses for training curve
        roc, epoch_stop, train_loss_list, val_loss_list = train_model(model, train_loader_fold, val_loader_fold, es_handler, optimiser, verbose=False)
        print(f"Fold {fold + 1} AUROC Score: {roc:.4f}\n")
        mlflow.log_metric(f"CV/Fold {fold + 1} AUROC Score", roc)
        roc_scores.append(roc)
        stopping_epochs.append(epoch_stop)

     # Find average best_epoch from CV
    avg_best_epoch = int(np.mean(stopping_epochs))
    print(f"Average best epoch: {avg_best_epoch}")

    # Evaluate the model after CV with AUROC
    mean = np.mean(roc_scores)
    std = np.std(roc_scores)
    print("Cross-Validation Results: %.2f%% (+/- %.2f%%)" % (mean * 100, std * 100))

    ### TRAIN FINAL MODEL ON TRAINING SET ##############################################################################
    # Fit preprocessor on full training data
    input_dim_final = X_train_full_processed.shape[1]

    print(f"Final model input dimension: {input_dim_final}")
    print(f"Test data dimension after preprocessing: {X_test_processed.shape[1]}")
    final_model = O2Classifier(input_dim_final).to(device) # Initialise fresh model for final training on full training dataset (no validation set)
    optimiser = torch.optim.Adam(final_model.parameters(), lr=5e-4)  # Updates the model’s parameters to minimize the loss function

    # Train final model for the average best epoch count (no early stopping needed)
    print(f"\nNow training final model with {avg_best_epoch} epochs.\n")
    n_epochs = avg_best_epoch  # Set to average best epoch #Improve this feeds into train_model but should be fed into function, not set in script
    train_model(final_model, train_full_loader, None, es_handler=None, optimiser=optimiser, verbose=True)

    # Save input features for validation - JSON (human-readable) and joblib
    with open(f"{output_data_dir}/input_features.json", "w") as f:
        json.dump(X_train_original.columns.tolist(), f)
    joblib.dump(X_train_original.columns.tolist(), f"{output_data_dir}/input_features.joblib")

    if show_detail: #IMPROVE especially for NN which has a separate pipeline this might not be necessary - as in there's simpler ways
        # Track retained features post-preprocessing
        preprocessor = preprocessor.named_steps['preprocessor']
        var_thresh = preprocessor.named_steps['var_thresh']
        retained_mask = var_thresh.get_support()
        retained_features = X_train_original.columns[retained_mask] #todo X_train_original instead of X train i think, as X_train was preprocessed and converted to numpy
        print("Features after thresholding:", len(retained_features.tolist()))
        show_features = False  # Enable or disable as required
        if show_features:
            print(retained_features.tolist())
        else:
            print("show_features is disabled in neural_network.py. To view features as a list, enable this bool.")

    # Print the selected features post-feature selection method # WARNING Not tested with all methods
    try:
        selector = preprocessor.named_steps['feature_selector']
        if hasattr(selector, 'get_support'):  # Standard scikit-learn selector
            support_mask = selector.get_support()
            selected_features = X_train_full.columns[support_mask].tolist()
        elif hasattr(selector, 'support_'):  # Other selector types
            selected_features = X_train_full.columns[selector.support_].tolist()
        else:  # For other selector types, get features via transformation
            print("Feature selection method is incompatible with current handling to extract features - results are not printed.")
            selected_features = X_train_full.columns.tolist() # Set selected features to full X_train if not assigned by a feature selector
        # Print features
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features)
    except Exception as e:
        print(f"Unable to print features for this feature selection method: {str(e)}")

    # Save selected features for validation - JSON (human-readable) and joblib
    with open(f"{output_data_dir}/selected_features.json", "w") as f:
        json.dump(selected_features, f)
    joblib.dump(selected_features, f"{output_data_dir}/selected_features.joblib")

    # # Reconstruct dataframe after preprocessing/dtype conversion #TODO not sure if this is needed or interferes - check. Doing it now because I understand the workflow, but can delete if not used
    # X_train = X_train_original[selected_features]

    ### Log the final pipeline preprocessor and model
    # Save preprocessor
    preprocessor_path = f"{data_dir}/preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    # Save model state
    final_model.to("cpu")
    model_path = f"{output_data_dir}/model.pt"
    torch.save({'model_state_dict': final_model.state_dict(),'input_dim': input_dim_final}, model_path)

    # Log artifacts
    mlflow.log_artifact(preprocessor_path)
    mlflow.log_artifact(model_path)

    # Log custom combined predictor
    class OxygenPredictor(mlflow.pyfunc.PythonModel):
        def __init__(self):
            super().__init__()
            self.input_dim = None

        def load_context(self, context):
            self.preprocessor = joblib.load(context.artifacts["preprocessor"])
            # Load model metadata
            model_data = torch.load(context.artifacts["pytorch_model"])
            self.input_dim = model_data['input_dim']
            # Reconstruct model
            self.model = O2Classifier(self.input_dim)
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.eval()

        def predict(self, context, model_input):
            processed = self.preprocessor.transform(model_input)
            tensor = torch.tensor(processed, dtype=torch.float32)
            with torch.no_grad():
                outputs = self.model(tensor)
                return torch.sigmoid(outputs).numpy()


    # Log combined model
    model_info = mlflow.pyfunc.log_model(artifact_path="best_model",
                            python_model=OxygenPredictor(),
                            artifacts={
                                "preprocessor": preprocessor_path,
                                "pytorch_model": model_path
                            })
    model_id = model_info.model_uuid  # Get model ID to copy over if in HPC

    ### APPLY MODEL TO TEST DATA #######################################################################################
    final_model.eval()
    with torch.no_grad():
        logits = final_model(X_test_dl)
        y_proba = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)  # todo check this is right!
    print("Confusion Matrix:")
    print(cm)

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

    # Print and log metrics
    print(f"\nTest accuracy with best model: {test_accuracy:.4f}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Test F1 with best model: {test_f1:.4f}")
    mlflow.log_metric("test_f1", test_f1)
    print(f"Test AUROC with best model: {test_roc:.4f}\n")
    mlflow.log_metric("test_roc", test_roc)
    mlflow.log_metric("sensitivity-tpr-recall", sensitivity)
    mlflow.log_metric("specificity-tnr", specificity)
    mlflow.log_metric("precision-ppv", precision)
    mlflow.log_metric("npv", npv)

    # TODO AI GEN TEMP OUTPUTS - find and make my own (eg CM already has a function)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["O2 not required", "O2 required"]))

    ### Save predictions to csv
    NN_prediction_path = Path(f"{output_data_dir}/Prediction_results_test_data.csv")

    # Define path for possible pre-existing results file
    if not args.from_pipeline:
        ML_prediction_path = Path(
            f"{data_dir}/ML/Prediction_results_test_data.csv")  # Saved to cwd training data file - WARNING: this will add results to the latest ML results if present. The config for these may not be the same. To properly store based on run name (and same config), use the wrapper script.
    else:
        ML_prediction_path = Path(
            f'model_output/{run_name}/training_data/ML/Prediction_results_test_data.csv')  # Saved to input storage file for ML

    # Make df of results
    NN_results = pd.DataFrame({'NN predictions': y_pred}, index=X_test.index)

    # Check if same run has a prediction results file for the traditional ML model
    if not os.path.exists(ML_prediction_path):  # If it doesn't exist, make a new file
        NN_results.to_csv(f"{output_data_dir}/Prediction_results_test_data.csv")
    else:  # If it does exist, append and copy
        ML_results = pd.read_csv(ML_prediction_path, index_col=0)  # Read in old results
        # Delete pre-existing columns (i.e. if running as a standalone script, it won't append multiple results for multiple ML runs)
        overlap = ML_results.columns.intersection(NN_results.columns)
        ML_results = ML_results.drop(columns=overlap)
        results_combined = pd.concat([ML_results, NN_results], axis=1)
        results_combined.to_csv(ML_prediction_path)
        shutil.copy2(ML_prediction_path, NN_prediction_path)  # Copy back to NN results

    # TODO check over model_building for TML to see if anything extra is missed from here - and vice versa
    ### GRAPHS #############################################################################################################
    # Plot PCA on the combined dataset - i.e. all data after feature selection
    with mlflow.start_run(nested=True):  # Start another run to avoid auologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run

        # Combine the datasets
        X_full = pd.concat([X_train_full, X_test])
        y_full = pd.concat([y_train_full, y_test]).reset_index(drop=True)

        # Call function to plot PCA on the dataset post feature selection
        pca_pre_post_fs(X_full, selected_features, y_full, graphs_dir, "After")

    # Plot learning curve
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (BCE with logits)')
    plt.legend()
    plt.savefig(f'{graphs_dir}/loss_curve.png')

    # Plot ROC curve
    plot_roc_auc(y_proba, y_test, graphs_dir)

    # SHAP #IMPROVE can dimensionality reduce first due to high feature number
    explainer = shap.KernelExplainer(model_predict, X_train_full_processed[:50])
    shap_values = explainer.shap_values(X_test_processed, nsamples=100)
    plt.figure()
    shap.summary_plot(shap_values, X_test_processed, feature_names=selected_features, show=False)
    fig = plt.gcf()  # Get current figure created by shap
    plt.savefig(f"{graphs_dir}/SHAP_graph.png")
    plt.close(fig)

    ### Repeat SHAP but this time aggregate metadata and protein data to examine influence
    # Calculate meta columns after feature selection
    starting_meta_cols_count = config['general']['training_meta_cols'] - 1 # Before preprocessing. -1 Due to removed of Label (O2 req.) column for X vs y
    meta_cols_before = X_train_full.iloc[:, :starting_meta_cols_count].columns # Slice dataset for meta and MS data before processing and get columns
    protein_cols_before = X_train_full.iloc[:, starting_meta_cols_count:].columns
    X_train_full_processed_df = pd.DataFrame(X_train_full_processed, columns=selected_features) # Convert back to df for function use
    meta_cols_surrey = remaining_meta(meta_cols_before.tolist(), X_train_full_processed_df, sample_inves_7=False, graphs_dir=None) # Note: The graph produced here isn't really needed but kept in to visualise selected features
    metadata_features = selected_features[:meta_cols_surrey]
    proteomics_features = selected_features[meta_cols_surrey:]
    # Split SHAP based on class
    shap_groups = {"Metadata" : metadata_features,
                   "Proteomics data" : proteomics_features
                   }
    shap_grouped = grouped_shap(shap_values, selected_features, shap_groups)
    plt.figure()
    shap.summary_plot(shap_grouped.values, feature_names=shap_grouped.columns, show=False)
    fig = plt.gcf()  # Get current figure created by shap
    plt.savefig(f"{graphs_dir}/SHAP_graph_grouped.png")
    plt.close(fig)

    # Plot precision-recall curve
    plot_precision_recall(y_proba, y_test, graphs_dir)

    # Plot calibration curve
    classifier_whitespace = "Neural network" # Input a name to present on the graph
    plot_calibration_curve(y_proba, y_test, classifier_whitespace, graphs_dir)

    # Plot decision distribution
    plot_decision_distribution(y_proba, y_test, graphs_dir)

    ### Plot PCA on final predictions - Test data before and after prediction
    with mlflow.start_run(nested=True):  # Start another run to avoid autologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run
        # Plot PCA
        plot_pca_predicted(X_test, selected_features, y_test, graphs_dir, y_pred)

    # Print run id
    final_run_id = run.info.run_id
    store_final_id = f"Run {run_name} for the neural network model completed. Run ID is {final_run_id}"

    # Log artifacts
    if enable_tracking:
        mlflow.log_artifacts(graphs_dir, artifact_path="graphs")
        mlflow.log_artifacts(output_data_dir, artifact_path="tables")

    ### SAVE DATA FOR ADDITIONAL GRAPHS ################################################################################
    y_pred_df = pd.DataFrame(y_pred, index=y_test.index) #todo check index is correct
    y_proba_df = pd.DataFrame(y_proba, index=y_test.index)
    y_results = pd.concat([y_pred_df, y_proba_df], axis=0)
    y_results.to_csv(f"{output_data_dir}/y_results.csv")

### STORE RESULTS IN NEW FOLDER ########################################################################################
# todo: if running as a standalone script (not in wrapper) then it will copy whatever old files + ML files too. Ideally only select current run files + NN
# Move and rename runs to a new directory for easier examination - results are copied from the MLflow tracking folder
# (which is also available in the server) but renamed here for easier access based on the suffix defined in the config
# file.
# Bool to set whether to copy the runs to the final output subdirectory - for testing only this can be disabled
if track_final: #IMPROVE: take out useful individual subfolders vs whole folder contents - need to determine which bits are useful
    print("\'track_final\' has been enabled, so the model information will be copied to ./model_output for easier viewing.")

    # Determine file locations
    final_folder = Path("mlruns") / final_exp_id / final_run_id
    ml_artifacts = Path("mlartifacts") / final_run_id
    output_folder = Path("model_output") / run_name #TODO could output to NN/ML subdirectories for easier comparision - but need to check that works with whole script and wrapper too
    output_artifacts = output_folder
    data_folder = Path(data_dir)
    graph_folder = Path(graphs_dir)

    # Copy final model folder contents
    shutil.copytree(final_folder, output_folder, dirs_exist_ok=True)
    print(f"\nCopying {final_folder} to {output_folder}")
    # Copy final model artifacts from mlartifacts to the model_output model file
    #  Note: since setting an experiment name changes the artifacts location to mlartifacts instead of in the mlruns (run) folder, we will copy it over for our final output
    shutil.copytree(ml_artifacts, output_artifacts, dirs_exist_ok=True)
    print(f"Copying {ml_artifacts} to {output_artifacts}")
    # Copy training data and graphs folder
    shutil.copytree(data_folder, output_folder / "training_data", dirs_exist_ok=True)  # IMPROVE more elegant
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

    # Save key metrics to csv # Note: When scripts are run as standalone, run_name will change between ML and NN even if config is the same. Run using the wrapper script to present together.
    key_metrics_path = f"key_metrics_{run_name}.csv"
    key_metrics = {
        'NN Test Accuracy': test_accuracy,
        'NN Test F1': test_f1,
        'NN Test AUROC': test_roc,
    }
    # Either update the existing key metrics, or create a new file
    if os.path.exists(key_metrics_path):
        existing_metrics = pd.read_csv(key_metrics_path, index_col=0)
        for key, value in key_metrics.items():
            existing_metrics[key] = value
        existing_metrics.to_csv(key_metrics_path)
        key_metrics_df = existing_metrics
    else:
        key_metrics_df = pd.DataFrame({k: [v] for k, v in key_metrics.items()}, index=[run_name])
        key_metrics_df.to_csv(f"{output_folder}/{key_metrics_path}")

    # Save to masterlist of metrics
    all_key_metrics_path = "key_metrics.csv"
    if os.path.exists(all_key_metrics_path):
        all_metrics = pd.read_csv(all_key_metrics_path, index_col=0)
        # Drop existing row if present
        all_metrics.drop(index=run_name, errors='ignore', inplace=True)
        # Update
        if run_name not in all_metrics.index:
            all_metrics = pd.concat([all_metrics, key_metrics_df])
    else:
        all_metrics = key_metrics_df
    all_metrics.to_csv(all_key_metrics_path)

# Print run ids
print(store_hyp_id)
print(store_final_id)

# Close all figures
plt.close('all')

# WARNING: If getting the 'too many 500 error responses' warning due to deleting files, run 'kill $(lsof -t -i tcp:8080)' in the terminal

# TODO - is FS also applied within hyperopt in the TML? Runs with slow selectors take a long time which I don't remember from TML - but could be due to difference in selectors










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
# Early stopping for overfitting