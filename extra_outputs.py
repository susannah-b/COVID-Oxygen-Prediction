### SCRIPT USAGE #######################################################################################################
# This file is used to generate addtional metrics for my report on the oxygen prediction model. Ideally these would be
# incorporated into the original model code (such as bootstrapped AUROC), however to save time in running the scripts
# again they are instead calculated here. #IMPROVE incorporate into the main scripts

import subprocess
from pathlib import Path
import os
import yaml
import shutil
import sys
import joblib
import pandas as pd
import numpy as np
import mlflow
import torch
import warnings

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, roc_curve, auc, brier_score_loss
from sklearn.utils import resample
from scipy.stats import norm
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from functions import plot_roc_auc, plot_calibration_curve, \
    plot_pca_predicted, plot_confusion_matrix, plot_pca_original, \
    plot_pca_test_unprocessed, remaining_meta, set_graph_style

# Ignore future warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Series.__getitem__ treating keys as positions")

### SET RANDOM SEEDS ###################################################################################################
# Set global random seeds
torch.manual_seed(42) # PyTorch CPU
torch.cuda.manual_seed_all(42) # PyTorch GPU (if available)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Apply graph styles
set_graph_style()

# Set pandas to display all columns and longer rows # IMPROVE remove in final version
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

#  WARNING: Manually change these for each run:
# Manually state model name and best_model logged folder (HPC ver) - determined from the .out SLURM file
model_name = "1_0803-180856_V+D-T2B[M-P-C-R-]M+Mo[All]FS[All-2]NFS[NONE]E+_V-_best_config" # Specify model name here
best_model_TML = "mlartifacts/models/m-dd7d75e4b0464159b7e7cff6d08805fa/artifacts" # Specify where the best TML model is logged to (this format is specfic to the way the files are tracked on the HPC)
best_model_NN = "mlartifacts/models/m-8ab5e1d5a5914b75a806b9d080d25da2/artifacts" # Specify where the best NN model is logged to

print(f"Producing extra data for model {model_name}")
### READ IN CONFIG FILE ################################################################################################
# Create config fil if it doesn't exist
config_path = Path(f"inputs/ML/{model_name}/config.yaml") # Specified ML but same as NN file; the inputs system needs a cleanup/overhaul to be more logical
if not os.path.exists(config_path):
    print("No config detected; quitting.")
    exit(1)

# Read in base config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Check whether to validate
validate = config["general"]["validate"]

### DATA DIRECTORIES ###################################################################################################
# Create output directories for the data
output_data_dir = f'extra_outputs/{model_name}'
os.makedirs(output_data_dir, exist_ok=True)

# Read in data
model_output = f"model_output/{model_name}"
training_data = f"{model_output}/training_data"
if validate:
    validation_data = f"{model_output}/external_validation/validation_data"

# Read in train and test files
X_path_train = Path(__file__).parent / training_data / f"Surrey_X_train.csv"
y_path_train = Path(__file__).parent / training_data / f"Surrey_y_train.csv"
X_train = pd.read_csv(X_path_train, index_col=0)
y_train = pd.read_csv(y_path_train, index_col=0).astype(np.float32).squeeze()
X_path_test = Path(__file__).parent / training_data / f"Surrey_X_test.csv"
y_path_test = Path(__file__).parent / training_data / f"Surrey_y_test.csv"
X_test = pd.read_csv(X_path_test, index_col=0)
y_test = pd.read_csv(y_path_test, index_col=0).astype(np.float32).squeeze()
if validate:
    X_path_val = Path(__file__).parent / validation_data / f"ISARIC_X.csv"
    y_path_val = Path(__file__).parent / validation_data / f"ISARIC_y.csv"
    X_val = pd.read_csv(X_path_val, index_col=0)
    y_val = pd.read_csv(y_path_val, index_col=0).astype(np.float32).squeeze()

# Print details on the dataset
print(f"Training samples: {len(X_train)}")
print(f"                  {np.sum(y_train == 1)} O2 required samples (=1)")
print(f"                  {np.sum(y_train == 0)} O2 not required samples (=0)")
print(f"Testing samples: {len(X_test)}")
print(f"                  {np.sum(y_test == 1)} O2 required samples (=1)")
print(f"                  {np.sum(y_test == 0)} O2 not required samples (=0)")
if validate:
    print(f"Validation samples: {len(X_val)}")
    print(f"                  {np.sum(y_val == 1)} O2 required samples (=1)")
    print(f"                  {np.sum(y_val == 0)} O2 not required samples (=0)")
#TODO print feature count before and after - or is this done in prev .out?

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

### LOAD IN MODELS #####################################################################################################
model_TML = mlflow.sklearn.load_model(best_model_TML)
model_NN = mlflow.pyfunc.load_model(best_model_NN)

def filter_to_inputs(model_type, X):
    # Load input features
    features_path = f"{model_output}/training_data/{model_type}/input_features.joblib"
    input_features = joblib.load(features_path)

    # Filter data to the original features
    X = X[input_features]
    # Ensure same column order
    X = X.reindex(columns=input_features)

    return X

X_train = filter_to_inputs("ML", X_train)
X_test = filter_to_inputs("ML", X_test)
if validate:
    X_val = filter_to_inputs("ML", X_val)

### CREATE WRAPPER FOR NN MODEL ########################################################################################
# Due to the combined sklearn and pytorch elements of the pipeline (preprocessing and NN model), the model is required to
#  be logged as pyfunc, not sklearn or pytorch. It therefore has no .predict attribute needed for SHAP KernelExplainer.

# Wrapper function to convert numpy input to pytorch tensors
def model_predict(X):
    # Convert numpy array to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Make predictions
    model_NN.eval()
    with torch.no_grad():
        logits = model_NN(X_tensor)
        # Return probabilities (sigmoid output)
        return torch.sigmoid(logits).numpy()


### FUNCTIONS FOR CIs ##################################################################################################
# Calculate BCa confidence intervals
def bca_ci(data, stat, alpha=0.05):
    bootstat = np.sort(data)
    n = len(data)

    # Bias-correction
    z0 = norm.ppf((bootstat < stat).mean())

    # Jackknife estimates for acceleration
    jackknife = np.array([np.mean(np.delete(data, i)) for i in range(n)])
    jack_mean = np.mean(jackknife)
    num = np.sum((jack_mean - jackknife) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    # Adjusted quantiles
    z_alpha1 = norm.ppf(alpha / 2)
    z_alpha2 = norm.ppf(1 - alpha / 2)

    pct1 = norm.cdf(z0 + (z0 + z_alpha1) / (1 - a_hat * (z0 + z_alpha1)))
    pct2 = norm.cdf(z0 + (z0 + z_alpha2) / (1 - a_hat * (z0 + z_alpha2)))

    lower = np.percentile(bootstat, 100 * pct1)
    upper = np.percentile(bootstat, 100 * pct2)
    return lower, upper


def classification_metrics_with_bca_ci(y_test, y_proba, y_pred, n_bootstraps=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)

    def compute_metrics(y_t, y_s, y_p):
        auc = roc_auc_score(y_t, y_s)
        acc = accuracy_score(y_t, y_p)
        f1 = f1_score(y_t, y_p)
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        return auc, acc, f1, sensitivity, specificity, ppv, npv

    # Original metrics
    auc, acc, f1, sens, spec, ppv, npv = compute_metrics(y_test, y_proba, y_pred)

    # Bootstrapping
    boot_results = {key: [] for key in ["auc", "accuracy", "f1", "sens", "spec", "ppv", "npv"]}
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_test), len(y_test))
        if len(np.unique(y_test[indices])) < 2:
            continue
        y_t = y_test[indices]
        y_s = y_proba[indices]
        y_p = y_pred[indices]
        m = compute_metrics(y_t, y_s, y_p)
        for key, val in zip(boot_results.keys(), m):
            boot_results[key].append(val)

    # BCa CI calculation
    results = {}
    metric_names = {
        "auc": "AUROC",
        "accuracy": "Accuracy",
        "f1": "F1 Score",
        "sens": "Sensitivity",
        "spec": "Specificity",
        "ppv": "PPV",
        "npv": "NPV"
    }
    metric_values = {
        "auc": auc,
        "accuracy": acc,
        "f1": f1,
        "sens": sens,
        "spec": spec,
        "ppv": ppv,
        "npv": npv
    }

    alpha = 1 - ci
    for key in boot_results.keys():
        boot_vals = np.array(boot_results[key])
        lower, upper = bca_ci(boot_vals, metric_values[key], alpha)
        results[metric_names[key]] = {
            "original value": metric_values[key],
            "bootstrapped median": np.median(boot_vals),
            "ci_lower": lower,
            "ci_upper": upper,
        }

    return results

### FIT TML TO TRAIN ###################################################################################################
classifier_path = Path(f"{model_output}/params/type")
with open(classifier_path, 'r') as f:
    classifier_type = f.read().strip()

# Translation of short type to presentable string
type_translation = {
    'svm': 'Support Vector Classifier',
    'rf': 'RandomForestClassifier',
    'logreg': 'Logistic Regression',
    'xgb': 'XGBoost Classifier',
    'gb': 'GradientBoosting Classifier',
    'ada': 'AdaBoost Classifier',
    'knn': 'K-Nearest Neighbors Classifier'
    }

print(f"\nNow predicting oxygen need for the test data using the {type_translation[classifier_type]}.")
# Predict on test data
y_preds_TML = model_TML.predict(X_test)
y_proba_TML = model_TML.predict_proba(X_test)[:, 1]

# Print confusion matrix
cm = confusion_matrix(y_test, y_preds_TML)
print("Confusion Matrix:\n", cm)

# Calculate metrics using BCa bootstrapping
metrics = classification_metrics_with_bca_ci(y_test, y_proba_TML, y_preds_TML)
for m, vals in metrics.items():
    print(f"{m}: Original {vals['original value']:.3f}, Bootstrapped median {vals['bootstrapped median']:.3f}, "
          f"(95% BCa CI {vals['ci_lower']:.3f}–{vals['ci_upper']:.3f})")

print("Test metrics with 95% CIs:")
print(pd.DataFrame.from_dict(metrics))
### FIT TML ON VALIDATION DATA #########################################################################################
if validate:
    print(f"\nNow predicting oxygen need for the validation data using the {type_translation[classifier_type]}.")
    # Predict on test data
    y_preds_TML_v = model_TML.predict(X_val)
    y_proba_TML_v = model_TML.predict_proba(X_val)[:, 1]

    # Print confusion matrix
    cm = confusion_matrix(y_val, y_preds_TML_v)
    print("Confusion Matrix:\n", cm)

    # Calculate metrics using BCa bootstrapping
    metrics_v = classification_metrics_with_bca_ci(y_val, y_proba_TML_v, y_preds_TML_v)
    for m, vals in metrics_v.items():
        print(f"{m}: Original {vals['original value']:.3f}, Bootstrapped median {vals['bootstrapped median']:.3f}, "
              f"(95% BCa CI {vals['ci_lower']:.3f}–{vals['ci_upper']:.3f})")
    print("Validation metrics with 95% CIs:")
    print(pd.DataFrame.from_dict(metrics_v))

# IMPROVE: As this really should be implemented as part of the original pipeline instead, I won't track with MLflow. instead, update training scripts.

### FIT NN TO TRAIN ####################################################################################################
# Predict on test data
y_proba_NN = model_NN.predict(X_test)
y_preds_NN = (y_proba_NN > 0.5).astype(int)

# Print confusion matrix
cm = confusion_matrix(y_test, y_preds_NN)
print("Confusion Matrix:\n", cm)

# Calculate metrics using BCa bootstrapping
metrics_n = classification_metrics_with_bca_ci(y_test, y_proba_NN, y_preds_NN)
for m, vals in metrics_n.items():
    print(f"{m}: Original {vals['original value']:.3f}, Bootstrapped median {vals['bootstrapped median']:.3f}, "
          f"(95% BCa CI {vals['ci_lower']:.3f}–{vals['ci_upper']:.3f})")
print("Test metrics with 95% CIs:")
print(pd.DataFrame.from_dict(metrics_n))

### FIT NN TO VALIDATION ###############################################################################################
if validate:
    # Predict on validation data
    y_proba_NN_v = model_NN.predict(X_val)
    y_preds_NN_v = (y_proba_NN_v > 0.5).astype(int)

    # Print confusion matrix
    cm = confusion_matrix(y_val, y_preds_NN_v)
    print("Confusion Matrix:\n", cm)

    # Calculate metrics using BCa bootstrapping
    metrics_n_v = classification_metrics_with_bca_ci(y_val, y_proba_NN_v, y_preds_NN_v)
    for m, vals in metrics_n_v.items():
        print(f"{m}: Original {vals['original value']:.3f}, Bootstrapped median {vals['bootstrapped median']:.3f}, "
              f"(95% BCa CI {vals['ci_lower']:.3f}–{vals['ci_upper']:.3f})")
    print("Validation metrics with 95% CIs:")
    print(pd.DataFrame.from_dict(metrics_n_v))

### SAVE FILES #########################################################################################################
# Confidence interval metrics
pd.DataFrame.from_dict(metrics).to_csv(f"{output_data_dir}/CIs_Test_TML.csv")
if validate:
    pd.DataFrame.from_dict(metrics_v).to_csv(f"{output_data_dir}/CIs_Val_TML.csv")
pd.DataFrame.from_dict(metrics_n).to_csv(f"{output_data_dir}/CIs_Test_NN.csv")
if validate:
    pd.DataFrame.from_dict(metrics_n_v).to_csv(f"{output_data_dir}/CIs_Val_NN.csv")

# Y probabilities
pd.DataFrame(y_proba_TML, columns=["probability"]).to_csv(f"{output_data_dir}/y_proba_TML.csv", index=False)
pd.DataFrame(y_proba_NN, columns=["probability"]).to_csv(f"{output_data_dir}/y_proba_NN.csv", index=False)
if validate:
    pd.DataFrame(y_proba_TML_v, columns=["probability"]).to_csv(f"{output_data_dir}/y_proba_TML_v.csv", index=False)
    pd.DataFrame(y_proba_NN_v, columns=["probability"]).to_csv(f"{output_data_dir}/y_proba_NN_v.csv", index=False)

### SAVE ROC DATA FOR PLOTTING #########################################################################################
def save_roc(y_proba, y_test, ID):
    roc_data = {}
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Store data
    roc_data[ID] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    # Save data to disk
    joblib.dump(roc_data, f'{output_data_dir}/roc_data_{ID}.pkl')

save_roc(y_proba_TML, y_test, "TML")
if validate:
    save_roc(y_proba_TML_v, y_val, "TML_val")
save_roc(y_proba_NN, y_test, "NN")
if validate:
    save_roc(y_proba_NN_v, y_val, "NN_val")


### PLOT CALIBRATION CURVE #############################################################################################
os.makedirs(f"{output_data_dir}/TML_cal_curve", exist_ok=True)
if validate:
    os.makedirs(f"{output_data_dir}/TMLv_cal_curve", exist_ok=True)
os.makedirs(f"{output_data_dir}/NN_cal_curve", exist_ok=True)
if validate:
    os.makedirs(f"{output_data_dir}/NNv_cal_curve", exist_ok=True)

# Plot calibration curves # IMPROVE This shouldn't be needed but currently if running TML and NN it overwrites - fix that, but for speed I'm replotting here
plot_calibration_curve(y_proba_TML, y_test, type_translation[classifier_type], f"{output_data_dir}/TML_cal_curve")
if validate:
    plot_calibration_curve(y_proba_TML_v, y_val, type_translation[classifier_type], f"{output_data_dir}/TMLv_cal_curve")
plot_calibration_curve(y_proba_NN, y_test, 'Neural network', f"{output_data_dir}/NN_cal_curve")
if validate:
    plot_calibration_curve(y_proba_NN_v, y_val, 'Neural network', f"{output_data_dir}/NNv_cal_curve")
### LIMIT DATASETS TO SELECTED FEATURES ################################################################################
# Load selected features
def selected_features(model_type):
    features_path_2 = f"{model_output}/training_data/{model_type}/selected_features.joblib"
    selected_features = joblib.load(features_path_2)
    return selected_features

selected_features_ML = selected_features("ML")
selected_features_NN = selected_features("NN")

X_train["O2 req."] = y_train
if validate:
    X_val["O2 req."] = y_val

# Combine dataset to present statistics for the full dataset
X_full = pd.concat([X_train, X_test], axis=0)

# Filter to selected features for each model framework
X_full_TML = X_full[selected_features_ML + ["O2 req."]]
X_full_NN = X_full[selected_features_NN + ["O2 req."]]
if validate:
    X_val_TML = X_val[selected_features_ML + ["O2 req."]]
    X_val_NN = X_val[selected_features_NN + ["O2 req."]]

### PRODUCE STATISTICS ON FULL DATA ####################################################################################
def generate_feature_summary(df, target_col='O2 req.'):
    # Initialize results dictionary
    results = {
        'Feature': [],
        'Mean (O2+)': [],
        'SD (O2+)': [],
        'Mean (O2-)': [],
        'SD (O2-)': [],
        'Sig. diff.': [],
        'P value': []
    }

    # Separate groups
    o2_plus = df[df[target_col] == 1]
    o2_minus = df[df[target_col] == 0]

    # Define categorical columns
    categorical_cols = ['Airway Disease', 'Covid Positive Hospital Swab (Y/N)', 'Smoking Status']

    for feature in df.columns:
        if feature == target_col: # Skip target column
            continue

        # Get data for both groups
        plus_vals = o2_plus[feature].dropna()
        minus_vals = o2_minus[feature].dropna()

        # Calculate means and SDs
        mean_plus = plus_vals.mean()
        sd_plus = plus_vals.std()
        mean_minus = minus_vals.mean()
        sd_minus = minus_vals.std()

        # Statistical testing
        p_value = np.nan
        sig_diff = "No difference"

        # Determine feature type
        if feature in categorical_cols or plus_vals.nunique() <= 5:
            # Categorical data (chi-square test)
            contingency = pd.crosstab(df[feature], df[target_col])
            if contingency.size > 0:
                _, p_value, _, _ = stats.chi2_contingency(contingency)

        elif plus_vals.nunique() == 2 and set(plus_vals.unique()) == {0, 1}:
            # Binary feature (z-test for proportions)
            count = [plus_vals.sum(), minus_vals.sum()]
            nobs = [len(plus_vals), len(minus_vals)]
            _, p_value = proportions_ztest(count, nobs)

        else:
            # Continuous/numerical feature (t-test)
            _, p_value = stats.ttest_ind(plus_vals, minus_vals, nan_policy='omit')

        # Determine significance direction for O2+
        if not np.isnan(p_value) and p_value < 0.05:
            if mean_plus > mean_minus:
                sig_diff = "Greater"
            else:
                sig_diff = "Lesser"

        # Store results
        results['Feature'].append(feature)
        results['Mean (O2+)'].append(mean_plus)
        results['SD (O2+)'].append(sd_plus)
        results['Mean (O2-)'].append(mean_minus)
        results['SD (O2-)'].append(sd_minus)
        results['Sig. diff.'].append(sig_diff)
        results['P value'].append(p_value)

    # Create DataFrame and set index
    result_df = pd.DataFrame(results)
    result_df.set_index('Feature', inplace=True)
    return result_df

summary_train_ML = generate_feature_summary(X_full_TML) # Note: using train but is the train+test data
if validate:
    summary_val_ML = generate_feature_summary(X_val_TML)
summary_train_NN = generate_feature_summary(X_full_NN)
if validate:
    summary_val_NN = generate_feature_summary(X_val_NN)

summary_train_ML.to_csv(f"{output_data_dir}/summary_train_ML.csv")
if validate:
    summary_val_ML.to_csv(f"{output_data_dir}/summary_val_ML.csv")
summary_train_NN.to_csv(f"{output_data_dir}/summary_train_NN.csv")
if validate:
    summary_val_NN.to_csv(f"{output_data_dir}/summary_val_NN.csv")














