### SCRIPT USAGE #######################################################################################################
# Run this script to perform necessary preprocessing steps on the cleaned Surrey and ISARIC data produced by
# feature_engineering.py.

######### SETUP ########################################################################################################
# Import libraries
import pandas as pd
from pathlib import Path
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
from functions import convert_categories, normalise_MS, impute_MICE, encode_categorical, encode_y, \
    plot_missingness_ms

# WARNING untested for validate False after some changes to the file structure
# WARNING This script is largely untested after changes to structure due to computational limitations. Check it runs fine on HPC - is the output data what you expect?

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Bool for any checking of the data that isn't needed for general use
show_testing = False

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
    config_path = Path(f"inputs/{run_name}/config.yaml")

# Read config file
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set parameters for this file:
validate = config['general']['validate'] # Whether to make the Surrey dataset compatible with the valdiation dataset
impute = config['data_preprocessing']['impute'] # Whether to impute (vs load a pre-imputed file)
drop_metadata = config['data_preprocessing']['drop_metadata'] # Whether to drop the metadata from the model
num_datasets = config['data_preprocessing']['num_datasets_imputation'] # num_datasets for miceforest imputation
iterations = config['data_preprocessing']['iterations_imputation'] # iterations for miceforest imputation

np.random.seed(42)

### PREPARE THE DATA ###################################################################################################

# Create output directories for the data
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    training_data = 'training_data' # Combine with other training graphs if using training data
    validation_data = 'validation_data' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    training_data = f'inputs/{run_name}/training_data'
    validation_data = f'inputs/{run_name}/validation_data'

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    training_graphs = 'training_graphs' # Combine with other training graphs if using training data
    validation_graphs = 'validation_graphs' # Combine with other validation graphs if using training data
else: # Put into input storage folder to prevent overwriting
    training_graphs = f'inputs/{run_name}/training_graphs'
    validation_graphs = f'inputs/{run_name}/validation_graphs'


### Read in train and test data (and full dataset for testing)
# Surrey
train_path = Path(__file__).parent / training_data / "Surrey_train.csv"
test_path = Path(__file__).parent / training_data / "Surrey_test.csv"
full_path = Path(__file__).parent / training_data / "Surrey_final.csv"
train = pd.read_csv(train_path, index_col=0)
test = pd.read_csv(test_path, index_col=0)
full_dataset = pd.read_csv(full_path, index_col=0)
# ISARIC
isaric_path = Path(__file__).parent / validation_data / "ISARIC_final.csv"
isaric = pd.read_csv(isaric_path, index_col=0)

### SPLIT DATA #########################################################################################################
# Split train and test data into X and y
# Surrey
surrey_X_train = train.drop('O2 req.',axis=1)
surrey_y_train = train['O2 req.'].copy()
X_test = test.drop('O2 req.',axis=1)
y_test = test['O2 req.'].copy()
# ISARIC
isaric_X = isaric.drop('O2 req.',axis=1)
isaric_y = isaric['O2 req.'].copy()

# Check data is read in correctly (currently only does train)
if show_testing:
    print("Surrey data:")
    print(surrey_y_train.head(5)) # Should be a series with SID as indexes, name is correct target variable (O2 req.)
    print(surrey_X_train.iloc[:5, :5]) # Should be a df with SID as indexes, column values look as expected
    print("ISARIC data:")
    print(isaric_y.head(5))  # Should be a series with SID as indexes, name is correct target variable (O2 req.)
    print(isaric_X.iloc[:5, :5])  # Should be a df with SID as indexes, column values look as expected

### DETERMINE REMAINING METADATA COLUMNS ###############################################################################
meta_cols_surrey = config['general']['training_meta_cols'] - 1 # -1 Due to removed of Label (O2 req.) column for X vs y
meta_cols_isaric = config['general']['validation_meta_cols'] - 1

# Drop the metadata if enabled # IMPROVE - should this be moved to feature_engineering.py - doesn't affect results but is a bit misleading that the produced data includes metadata there for earlier files
if drop_metadata:  # Drop the metadata if bool is true
    # Surrey
    surrey_X_train = surrey_X_train.iloc[:, meta_cols_surrey:]
    meta_cols_surrey = 0
    print(f"Metadata was dropped from the Surrey dataset; if unintended, disable drop_metadata in the script.")
    # ISARIC
    isaric_X = isaric_X.iloc[:, meta_cols_isaric:]
    meta_cols_isaric = 0
    print(f"Metadata was dropped from the ISARIC dataset; if unintended, disable drop_metadata in the script.")

# Update config for later access
with open(config_path, "w") as f:
    config["general"]["training_meta_cols"] = meta_cols_surrey
    config["general"]["validation_meta_cols"] = meta_cols_isaric
    yaml.dump(config, f, sort_keys=False)

### HANDLE CATEGORICAL DATA FOR IMPUTATION #############################################################################
# Detect numeric vs categorical columns
numeric_cols_s = surrey_X_train.select_dtypes(include='number').columns.tolist()
cat_cols_s = surrey_X_train.select_dtypes(exclude='number').columns.tolist()
numeric_cols_i = isaric_X.select_dtypes(include='number').columns.tolist()
cat_cols_i = isaric_X.select_dtypes(exclude='number').columns.tolist()

if show_testing:
    print("\nNumeric features in Surrey data:", numeric_cols_s)
    print("\nCategorical features in Surrey data:", cat_cols_s)
    print("\nNumeric features in ISARIC data:", numeric_cols_i)
    print("\nCategorical features in ISARIC data:", cat_cols_i)
    # TODO still not sure if chol should be in here or not - investigate

# Check which test categories are binary/ordinal - use full dataset to check all regardless of train/test split
if show_testing:
    # Note: currently done for Surrey data only
    print("----- Test features: -----")
    for cat in cat_cols_s:
        print(f"{cat} values:")
        print(full_dataset[cat].value_counts())

        # Results:
        # - Covid Positive Hospital Swab (Y/N): Of 172 values, 2 are outside Y/N binary. Instead of one-hot encoding into
        # four columns, I think ordinal encoding of 'Inconclusive' and 'N - previously pos in ICU' as between Y/N makes more sense
        # All others are binary Y/N or M/F so will label encode

        # Binary/ordinal columns can be converted to orderered pd.Categorical, with nominal (none present) to unordered
        #  pd.Categorical. After imputation then label encode

    # Check target variable (should be binary Y/N)
    print("----- Target variable: -----")
    print(full_dataset['O2 req.'].value_counts())
    # Target is also binary so can label encode

# Filter out ordinal/binary categories for ordered categorical encoding
ordinal_cats = {'Chol' : ['N', 'Y'],
                'Bilateral CXR changes' : ['N', 'Y'],
                'CPAP' : ['N', 'Y'],
                'Clinical Covid (Y/N)' : ['N', 'Y'],
                'Covid Positive Hospital Swab (Y/N)' : ['N', 'Inconclusive', 'Y'],
                'For escalation? (Y/N)' : ['N', 'Y'],
                'Gender' : ['M', 'F'],
                'HTN' : ['N', 'Y'],
                'ICU admission' : ['N', 'Y'],
                'IHD' : ['N', 'Y'],
                'MADU admission' : ['N', 'Y'],
                'Survived Admission' : ['N', 'Y'],
                'T2DM' : ['N', 'Y']
                }

nominal_cats = [] # In this case empty, but may not be with other data sets so leaving in as framework - see MH model for example
# For both X_train and X_test (and isaric_X), convert to ordered categorical WITHOUT extracting codes

### Convert categories to pandas categorical - ordinal and nominal
# Convert series to frame # todo this only recently errored - hopefully this fix effects nothing but check
surrey_y_train = surrey_y_train.to_frame()
y_test = y_test.to_frame()
isaric_y = isaric_y.to_frame()
# Ordinal categories
surrey_X_train = convert_categories(surrey_X_train, ordinal_cats)
surrey_y_train = convert_categories(surrey_y_train, ordinal_cats)
X_test = convert_categories(X_test, ordinal_cats)
y_test = convert_categories(y_test, ordinal_cats)
isaric_X = convert_categories(isaric_X, ordinal_cats)
isaric_y = convert_categories(isaric_y, ordinal_cats)

### NORMALISE PROTEOMICS DATA ########################################################################################## #TODO is it better to impute first?
surrey_X_train_quant = normalise_MS(surrey_X_train, meta_cols_surrey)
X_test_quant = normalise_MS(X_test, meta_cols_surrey)
isaric_X_quant = normalise_MS(isaric_X, meta_cols_isaric)

### IMPUTE MISSING VALUES ##############################################################################################
# Visualise missing data in the MS data to investigate missingness type
plot_missingness_ms(surrey_X_train_quant, training_graphs, 'Surrey')
plot_missingness_ms(isaric_X_quant, validation_graphs, 'ISARIC')

imputed_surrey_train = f"{training_data}/Surrey_train_after_imputation.csv"
imputed_surrey_test = f"{training_data}/Surrey_test_after_imputation.csv"
imputed_isaric = f"{validation_data}/ISARIC_after_imputation.csv"

if impute:
    surrey_X_train = impute_MICE(surrey_X_train, imputed_surrey_train, 'Surrey_Train', num_datasets, iterations, training_graphs)
    X_test = impute_MICE(X_test, imputed_surrey_test, 'Surrey_Test', num_datasets, iterations, training_graphs)
    isaric_X = impute_MICE(isaric_X, imputed_isaric, 'ISARIC',num_datasets , iterations, validation_graphs)

else: # If not imputing, read in the data
    print("Skipping imputation; using already produced imputed file. Otherwise set impute = True")
    surrey_X_train = pd.read_csv(imputed_surrey_train, index_col=0)
    X_test = pd.read_csv(imputed_surrey_test, index_col=0)
    isaric_X = pd.read_csv(imputed_isaric, index_col=0)

# Convert all categorical types back to pandas categorical
# WARNING: This might be an issue caused by the temporary imputation I did for ISARIC. Comment this when run with MICE
#  to check if it still applies and if it doesn't, delete this. (also adapt for surrey_X_train, X_test, and isaric_X
# Convert to numerical or categorical
# for col in ordinal_cats:
#     if col in X_train:
#         X_train[col] = X_train[col].astype('category')
#
# # Check columns are correctly imputed for categorical data
# if show_testing:
#     for cat in ordinal_cats:
#         if cat in X_train.columns:  # Check that the column is present - allows same dict to be used for multiple datasets
#             print(f"\nUnique values in X_train {cat}: {X_train[cat].unique()}")
#             if not validate:
#                 print(f"Unique values in X_test {cat}: {X_test[cat].unique()}")
#                 # Should look the same as before imputation (probably unecessary testing now code is defined but leaving in)

### ENCODE CATEGORICAL DATA ############################################################################################
encode_categorical(surrey_X_train, ordinal_cats)
encode_categorical(X_test, ordinal_cats)
encode_categorical(isaric_X, ordinal_cats)

# Note: This dataset does not currently have any nominal categories, but otherwise one-hot encode here. See mental
# health data project for an example.

# Encode y data
surrey_y_train = encode_y(surrey_y_train)
y_test = encode_y(y_test)
isaric_y = encode_y(isaric_y)

### SAVE DATA ##########################################################################################################
# Write to csv for use in next script
surrey_X_train.to_csv(f"{training_data}/Surrey_X_train.csv", sep=",", index=True)
surrey_y_train.to_csv(f"{training_data}/Surrey_y_train.csv", sep=",", index=True)
X_test.to_csv(f"{training_data}/Surrey_X_test.csv", sep=",", index=True)
y_test.to_csv(f"{training_data}/Surrey_y_test.csv", sep=",", index=True)
isaric_X.to_csv(f"{validation_data}/ISARIC_X.csv", sep=",", index=True)
isaric_y.to_csv(f"{validation_data}/ISARIC_y.csv", sep=",", index=True)

# Close all figures
plt.close('all')