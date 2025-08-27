### SCRIPT USAGE #######################################################################################################
# This file is used to generate addtional metrics for my report on the oxygen prediction model. Ideally these would be
# incorporated into the original model code (such as bootstrapped AUROC), however to save time in running the scripts
# again they are instead calculated here.

# WARNING: File paths have been removed for publication, and certain inputs were made manually outside of the scripts
#  and the script will therefore unable to run

########################################################################################################################
import os
import joblib
import pandas as pd
import torch
from matplotlib import pyplot as plt
from MLstatkit.stats import Delong_test
from functions import set_graph_style, plot_metrics_heatmap, get_palette
import seaborn as sns

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
    "Example.pkl",
    "Example_2.pkl",
    "Example_3.pkl"
]
model_ids = ["Run 1", "Run 2", "Run 3"]

# Set up plot
sns.set_palette(get_palette(str(len(model_ids))), desat=1)
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

# ### PLOT HEATMAP OF METRICS ############################################################################################
metrics_path = "[file_path/all_results_metrics.csv]" # WARNING: Example file only, will not run
# TODO: all_results_metrics is currently made manually, although should be the all_key_metrics.csv file just with the
#  addition of a 'Summary' column inserted to manaully label (could be automated here instead) (plus 'Best Test AUROC'
#  but this wasn't used in the scripts at all). Used a manual version as I had some issues with generating the all_key_metrics
#  file in the HPC - investigate this.
metrics = pd.read_csv(metrics_path)
test_metrics = metrics.iloc[:, [1] + list(range(3, 9))].set_index('Summary')
validation_metrics = metrics.iloc[0:3, [1] + list(range(-6, 0))].set_index('Summary')
conf_int_test = pd.read_csv("extra_outputs/confidence_intervals_test.csv").set_index('Summary')
conf_int_val = pd.read_csv("extra_outputs/confidence_intervals_val.csv").set_index('Summary')
# TODO: confidence_intervals files were also generated manually in the below format, from the CIs files generated in
#  extra_outputs.py. This should also be automated instead
#  Example:
# Summary	ML Test Accuracy	ML Test F1	ML Test AUROC	NN Test Accuracy	NN Test F1	NN Test AUROC
# 1. V+ M+ D0	(0.53-0.83)	(0.43-0.86)	(0.51-0.95)	(0.50-0.83)	(0.43-0.86)	(0.50-0.93)
# 2. V+ M- D0	(0.40-0.73)	(0.31-0.76)	(0.49-0.88)	(0.57-0.87)	(0.45-0.89)	(0.50-0.91)
# 3. V+ M- D+	(0.40-0.70)	(0.38-0.74)	(0.44-0.80)	(0.48-0.78)	(0.36-0.76)	(0.43-0.83)
# 4. V- M+  B+ D0	(0.63-0.93)	(0.48-0.92)	(0.63-0.97)	(0.40-0.77)	(0.36-0.80)	(0.58-0.96)
# 5. V- M+ B- D0	(0.57-0.87)	(0.45-0.88)	(0.66-0.97)	(0.57-0.87)	(0.45-0.88)	(0.48-0.93)
# 6. V- M- B- D0	(0.50-0.83)	(0.38-0.83)	(0.59-0.94)	(0.50-0.83)	(0.43-0.86)	(0.52-0.92)
# 7. V- M- B- D+	(0.48-0.78)	(0.41-0.79)	(0.53-0.86)	(0.53-0.80)	(0.42-0.81)	(0.45-0.85)

plot_metrics_heatmap(test_metrics, output_data_dir, "Test", conf_int_test)
plot_metrics_heatmap(validation_metrics, output_data_dir, "Validation", conf_int_val)

### DELONG'S TEST ######################################################################################################
# Load in data for ROC # WARNING: Example file paths where [] are used. Change model name, and for delong's can change which y_proba files are used
y_proba_model1 = pd.read_csv("extra_outputs/[Example_model_name]/y_proba_TML.csv", header=0).squeeze().to_numpy()
y_proba_model2 = pd.read_csv("extra_outputs/[Example_model_name_2]/y_proba_TML.csv", header=0).squeeze().to_numpy()
y_true = pd.read_csv("[file_path]/Surrey_y_test.csv", header=0) # WARNING y_true for D0 - comment if using D+ data
# y_true = pd.read_csv("/Users/s.blundell/LIFE703-Project/COVID_Oxygen_Prediction/RESULTS/7_0803-181957_V-D+T2B[M-P-C-R-]M-Mo[All]FS[All-2]NFS[NONE]E+_V-_no_meta_D+/training_data/Surrey_y_test.csv", header=0) # WARNING y_true for D+ - comment if using D0 data only
y_true = y_true.iloc[:, 1].squeeze().to_numpy()
# Perform test
z_score, p_value = Delong_test(y_true, y_proba_model1, y_proba_model2)
# Print result
print(f"Delong's Result: Z-score {z_score} P-value {p_value}")
