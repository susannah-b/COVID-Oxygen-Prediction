### SCRIPT USAGE #######################################################################################################
# This file is used to generate addtional metrics for my report on the oxygen prediction model. Ideally these would be
# incorporated into the original model code (such as bootstrapped AUROC), however to save time in running the scripts
# again they are instead calculated here.

import subprocess
from pathlib import Path
import os
import joblib
import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from MLstatkit.stats import Delong_test
from functions import plot_roc_auc, set_graph_style, plot_metrics_heatmap

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

### DATA DIRECTORIES ###################################################################################################
# Create output directories for the data
output_data_dir = f'extra_outputs/graphs'
os.makedirs(output_data_dir, exist_ok=True)

# Set ROC paths and names
file_paths = [
    "/Users/s.blundell/LIFE703-Project/COVID_Oxygen_Prediction/extra_outputs/1_0803-180856_V+D-T2B[M-P-C-R-]M+Mo[All]FS[All-2]NFS[NONE]E+_V-_best_config/roc_data_TML.pkl",
    "/Users/s.blundell/LIFE703-Project/COVID_Oxygen_Prediction/extra_outputs/1_0803-180856_V+D-T2B[M-P-C-R-]M+Mo[All]FS[All-2]NFS[NONE]E+_V-_best_config/roc_data_TML_val.pkl",
]
model_ids = ["1", "2"]  # TODO Update with curve names/run ids

# Set up plot
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], 'k--', lw=1)  # Diagonal reference line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')

# Load and plot each ROC curve
for i, file_path in enumerate(file_paths):
    # Construct full file path
    model_id = model_ids[i]

    # Load ROC data
    try:
        roc_data = joblib.load(file_path)
        model_id_in_file = list(roc_data.keys())[0]
        data = roc_data[model_id_in_file]

        # Plot ROC curve
        plt.plot(data['fpr'], data['tpr'],
                 label=f'{model_id} (AUC = {data["auc"]:.3f})')

    except FileNotFoundError:
        print(f"File not found: {file_path}")

# Finalize plot
plt.legend(loc='lower right')
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(output_data_dir, 'combined_roc_curves.png'))

### PLOT HEATMAP OF METRICS ############################################################################################
metrics_path = "/Users/s.blundell/LIFE703-Project/COVID_Oxygen_Prediction/extra_outputs/all_results_metrics.csv"
metrics = pd.read_csv(metrics_path)
test_metrics = metrics.iloc[:, [1] + list(range(3, 9))].set_index('Summary')
validation_metrics = metrics.iloc[0:3, [1] + list(range(-6, 0))].set_index('Summary')
plot_metrics_heatmap(test_metrics, output_data_dir, "Test")
plot_metrics_heatmap(validation_metrics, output_data_dir, "Validation")

### DELONG'S TEST ######################################################################################################
# Load in data for ROC
y_proba_model1 = pd.read_csv()
y_proba_model2 = pd.read_csv()
y_true = pd.read_csv()

# Perform test
z_score, p_value = Delong_test(y_true, y_proba_model1, y_proba_model2)

# Print result
print(f"Delong's Result: Z-score {z_score} P-value {p_value}")
