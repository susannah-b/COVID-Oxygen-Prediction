### SCRIPT USAGE #######################################################################################################
# Run this script to build and run a neural network to pedict oxygen need (O2 Req.).

### SETUP ##############################################################################################################
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from torch.utils.data import TensorDataset, DataLoader
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

# Bool to show additional detail
show_detail = False

### SET RANDOM SEEDS ###################################################################################################
# todo do for other scripts

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
# model_choice = config['model_building']['specify_model']['model_type'] # Model type if not basic training # TODO update with best once determined
fs_choice = config['model_building']['specify_model']['fs'] #Feature selector if not basic training # TODO update with best once determined
max_evals = config['model_building']['max_evals'] # How many evaluations to do in hyperopt tuning
track_final = config['model_building']['track_final'] # Whether to copy the model_output to the designated folder for easier browsing

#todo NN specific ones below - keep all these
batch_size = config['neural_network']['batch_size'] # Batch size to use for the neural net
n_epochs = config['neural_network']['n_epochs'] # How many epochs to run
nth_epoch = config['neural_network']['nth_epoch'] # Every nth epoch, print the loss
learn_rate = float(config['neural_network']['learning_rate']) # Learning rate for NN #todo hypertune?

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
    graphs_dir = 'training_graphs' # Combine with other training graphs if using training data
else: # Put into input storage folder to prevent overwriting
    graphs_dir = f'inputs/NN/{run_name}/training_graphs'
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

# Convert y to float32 for pytorch
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Feature dimensions: {X_train.shape[1]} | Classes: {y_train.nunique()}")

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
 # IMPROVE see model_building.py for note on variance thresholding
# Calculate median variance of all features
variances = X_train.var(axis=0)
threshold = float(var_threshold) # Effectively zero but avoids floating-point issues

# Note: Skipped VIF analysis for NN

# WARNING Create dataloaders removed from here (see old commit) - convert to float, ensure 1D for y_train

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

### ESTIMATE BEST [NN MODEL EQUIVALENT] WITH BASIC SETTINGS ############################################################
# todo - set up for NN - using 'X' instead of model; will set up according to what's best for NN but for now just laying out framework of TML
#  below is a temporary bypass to account for a NN basic_train equivalent not yet being built
best_X_fs = {}
best_X_fs['TODO'] = 'NONE' #todo - set to an arbitrary FS method so hyperopt FS can be set up, but needs a basic_train that chooses the best one - see TML

### DEFINE NEURAL NETWORK ##############################################################################################
#todo: need a basic train that will select [something, presumably a basic neural net] and corresponding feature selector. for now, using this predefined one.
# todo so has to be selected in hyperopt, which might make this bit redundant
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
        output = self.fc2(x).squeeze(1) # Outputs the logits for the number of classes.
        return output

### OBJECTIVE FUNCTION FOR HYPEROPT PARAMETER TUNING ###################################################################
# TODO: again see TML for structure. After/during setting up NN hyperopt refer back so theyre similar in structure

def objective(params):
    # ... todo set up for NN. Needs to also implement the fs space.
    pass


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
}

### DEFINE SEARCH SPACES PER MODEL #####################################################################################
  # TODO need TML equivalent for NN

### MLFLOW TRACKING ####################################################################################################
# Make folder for tracking runs
os.makedirs('./mlruns', exist_ok=True)

if not port_in_use(host, port):
    print(f"Running tracking server on {host}:{port}")
    subprocess.Popen(["mlflow", "server", "--backend-store-uri", "./mlruns", "--host", host, "--port", f"{port}"])
else:
    print(f"MLflow tracking server already listening on {host}:{port}")

# Pause to allow the server to boot up
    time.sleep(5)

# Set MLFLow tracking URI
mlflow.set_tracking_uri(uri=f"http://{host}:{port}")

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

### HYPEROPT TUNING WITH MLFLOW ########################################################################################
#todo adapt TML for NN

### TRAIN FINAL MODEL ##################################################################################################
#todo many parts were deleted from TML framework, so copy in later

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu" # Automatically uses a GPU if available; otherwise, defaults to the CPU

# Create a new MLflow Experiment
mlflow.set_experiment("Oxygen Prediction Neural Network - Surrey")

# Train final model using the full training data
mlflow.pytorch.autolog()
store_final_id = None # Initialise value to store run ID to print at end
final_run_id = None
final_exp_id = None
with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("Run name", run_name) # Set tag to custom run id so it's searchable in the MLFlow UI
    mlflow.set_tag("ML type", "Neural network")
    mlflow.set_tag("Phase", "Final model training")
    mlflow.set_tag("Hyperopt MLflow run", 'hyperopt_name') # todo should be hyperopt_name variable not str but not currently set up
    mlflow.log_param("mlflow_run_name", run.info.run_name)
    final_exp_id = run.info.experiment_id # Get experiment id for folder management

    # Get selector configuration
    selector_type = best_X_fs['TODO'] #todo stand in to replace basic_train which isn't yet implemented
    selector_config = feature_selectors[selector_type]

    # Build selector
    if selector_type == 'NONE':
        selector = 'passthrough'
    else:
        all_params = {**selector_config['base_params']} # TODO ', **fs_params' was removed from here as hyperopt training hasn't been done yet - will just use the default FS
        selector = selector_config['class'](**all_params)

    # Create the preprocessing pipeline
    preprocessor = Pipeline([
        ('int_to_float', IntToFloatTransformer()),
        ('var_thresh', VarianceThreshold(threshold=0.0)),
        ('scaler', StandardScaler()),
        ('feature_selector', selector if feature_selection else 'passthrough')
    ])

    # Fit preprocessor
    X_train_original = X_train.copy() # Stores original df
    X_train_processed = preprocessor.fit_transform(X_train, y_train) # Converts to a numpy array on output
    X_test_processed = preprocessor.transform(X_test)
    input_dim = X_train_processed.shape[1]

    ### CREATE DATALOADERS #################################################################################################
    # Convert X data to PyTorch tensors
    X_train_dl = torch.tensor(X_train_processed, dtype=torch.float32)
    X_test_dl = torch.tensor(X_test_processed, dtype=torch.float32)

    # Convert labels and ensure proper shape (64-bit integer to 1D label tensor)
    y_train_dl = torch.tensor(y_train.values, dtype=torch.float32).squeeze()
    y_test_dl = torch.tensor(y_test.values, dtype=torch.float32).squeeze()

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_dl, y_train_dl)
    test_dataset = TensorDataset(X_test_dl, y_test_dl)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Training samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Feature dimensions: {X_train.shape[1]} | Classes: {len(np.unique(y_train))}")

    # Train the neural networks
    model = O2Classifier(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with built-in sigmoid
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4)  # Updates the model’s parameters to minimize the loss function

    for epoch in range(n_epochs):
        model.train()  # Sets the model to training mode, enabling operations like dropout (if present)
        total_loss = 0
        for batch in train_loader:
            # Move the batch data to the same device as the model (GPU or CPU)
            features = batch[0].to(device)  # Selects input features
            labels = batch[1].to(device)  # Selects labels

            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)  # Calculates the classification error between predictions (outputs) and true labels (labels) using the cross-entropy loss

            # Backward pass and optimization
            optimiser.zero_grad()  # Resets gradients from the previous iteration to prevent accumulation
            loss.backward()  # Computes the gradients of the loss with respect to the model parameters via backpropagation
            optimiser.step()  # Updates the model parameters using the computed gradients

            total_loss += loss.item()  # Accumulates the total loss for the epoch to monitor training progress

        if (epoch == 0) or ((epoch + 1) % nth_epoch == 0) or (epoch == n_epochs - 1):  # Print every nth epoch or first/last epoch
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")  # Logs the average loss per epoch to track improvement

    # Save input features for validation - JSON (human-readable) and joblib
    with open(f"{output_data_dir}/input_features.json", "w") as f:
        json.dump(X_train_original.columns.tolist(), f)
    joblib.dump(X_train_original.columns.tolist(), f"{output_data_dir}/input_features.joblib")

    if show_detail:
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
            selected_features = X_train.columns[support_mask].tolist()
        elif hasattr(selector, 'support_'):  # Other selector types
            selected_features = X_train.columns[selector.support_].tolist()
        else:  # For other selector types, get features via transformation
            print("Feature selection method is incompatible with current handling to extract features - results are not printed.")
            selected_features = X_train.columns.tolist() # Set selected features to full X_train if not assigned by a feature selector
        # Print features
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features)
    except Exception as e:
        print(f"Unable to print features for this feature selection method: {str(e)}")

    # Save selected features for validation - JSON (human-readable) and joblib
    with open(f"{output_data_dir}/selected_features.json", "w") as f:
        json.dump(selected_features, f)
    joblib.dump(selected_features, f"{output_data_dir}/selected_features.joblib")

    # Reconstruct dataframe after preprocessing/dtype conversion #TODO not sure if this is needed or interferes - check. Doing it now because I understand the workflow, but can delete if not used
    X_train = X_train_original[selected_features]

    ### Log the final pipeline preprocessor and model
    # Save preprocessor
    preprocessor_path = f"{data_dir}/preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    # Save model state
    model.to("cpu")
    model_path = f"{data_dir}/model.pt"
    torch.save({'model_state_dict': model.state_dict(),'input_dim': input_dim}, model_path)

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
    mlflow.pyfunc.log_model(artifact_path="best_model",
                            python_model=OxygenPredictor(),
                            artifacts={
                                "preprocessor": preprocessor_path,
                                "pytorch_model": model_path
                            })

    # Apply model to test

    # 3. Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(X_test_dl)
        y_proba = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\nTest accuracy with best model: {test_accuracy:.4f}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Test F1 with best model: {test_f1:.4f}")
    mlflow.log_metric("test_f1", test_f1)
    print(f"Test ROC_AUC with best model: {test_roc_auc:.4f}\n")
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    # TODO AI GEN TEMP OUTPUTS - find and make my own (eg CM already has a function)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred) # todo check this is right!
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix
    plot_confusion_matrix(cm, graphs_dir)

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

    # TODO 'GRAPHS' section needs to be adapted from TML to here (might need diff functions) - as well as check over whole script again
    ### GRAPHS #############################################################################################################
    # Plot PCA on the combined dataset - i.e. original data after feature selection
    with mlflow.start_run(nested=True):  # Start another run to avoid auologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run

        # Combine the datasets
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

        # Call function to plot PCA on the dataset prior to feature selection
        pca_original(X_full, selected_features, y_full, graphs_dir)

    ### Plot PCA on final predictions - Test data
    with mlflow.start_run(nested=True):  # Start another run to avoid autologging conflicts
        mlflow.sklearn.autolog(disable=True)  # Disables autolog inside this run
        # Plot PCA
        plot_pca_predicted(X_test, selected_features, y_test, graphs_dir, y_pred)

    # Print run id
    final_run_id = run.info.run_id
    store_final_id = f"Run {run_name} for final neural network model completed. Run ID is {final_run_id}"

    # Log artifacts
    mlflow.log_artifacts(graphs_dir, artifact_path="graphs")
    mlflow.log_artifacts(output_data_dir, artifact_path="tables")

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
    ml_artifacts = Path("mlartifacts") / final_exp_id / final_run_id
    output_folder = Path("model_output") / run_name #TODO could output to NN/ML subdirectories for easier comparision - but need to check that works with whole script and wrapper too
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

    # # Make note of the corresponding hyperopt MLflow run #todo commented out while hyperopt isn't set up
    # hyper_run_file = final_folder / "hyperopt_run_name.txt"
    # hyper_run_file.write_text(f"{hyperopt_name}")

# Print run ids
# print(store_hyp_id) #todo commented out while hyperopt isn't set up
print(store_final_id)

# WARNING: If getting the 'too many 500 error responses' warning due to deleting files, run 'kill $(lsof -t -i tcp:8080)' in the terminal











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