### SCRIPT USAGE #######################################################################################################
# This script is the first in the series of model development scripts. It can be run standalone (which uses files from
# the current working directory to run the script before pasting the final outputs to model_outputs), or as part of
# pipeline.py which copies the input files to a new 'inputs' folder and works from there to remove any risk of
# overwriting.
# Run this script to investigate and clean 1. The Surrey data used for training the model, and 2. the ISARIC data used
# for external validation.

# Set values in config.yaml. See the default_config.yaml for details on how each is used. Edit the config.yaml (NOT
# the default_config.yaml) to adjust the run settings. For runs executed with pipeline.py, the config file for each run
# will be stored in 'inputs' under the run name.

# WARNING: If externally validating, the script must be run first for the validation data before the training data.
#  This is due to a (temporary) fix to prevent columns dropped from ISARIC being included in the training data.
#  Ideally this will be later improved to allow flexibility.
######### SETUP ########################################################################################################
# Imports
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import os
from datetime import datetime
import yaml
import shutil
from functions import check_abnormal_SIDs, calculate_overlaps, check_columns, check_empty_cols, \
    plot_row_missingness, plot_missingness_msno, investigate_null, remaining_meta, categorise_cols, numerical_check_nan, \
    plot_class_distribution, text_to_binary, replace_values, set_graph_style

# Set pandas to display all columns and longer rows # IMPROVE remove in final version
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

# Bools to determine what to print - set here not in config for ease of changing during testing
sample_inves_1 = False # Metadata info
sample_inves_2 = False # Metadata vs quant info
sample_inves_3 = False # Overlap of metadata vs quant
sample_inves_4 = False # Combining data frames/removing columns
sample_inves_5 = False # Checking for discrepancies in the metadata
sample_inves_6 = False # Data cleaning results within columns
sample_inves_7 = False # Missingness
sample_inves_8 = False # Numerical conversion
sample_inves_9 = False # Dropped colums

# Apply graph styles
set_graph_style()

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

# Create config file if it doesn't exist
default_config = Path("default_config.yaml")
if not os.path.exists(config_path):
    print("Feature_engineering.py is creating a new config file.")
    shutil.copy2(default_config, config_path)
else:
    print("Feature_engineering.py is using an existing config file.")

# Read config file # IMPROVE Some settings (e.g. training_data directory) could be moved to the config instead of hardcoded. For now only commonly changed settings are added
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set parameters for this file:
validate = config['general']['validate'] # Whether to make the Surrey dataset compatible with the validation set and produce relevant files
day_zero = config['feature_engineering']['day_zero'] # Whether to only include D0 samples
medication_48hr = config['feature_engineering']['text_features']['Medication_48hr'] # Whether to expand the column from csv text into binary columns for each value
pre_symptoms = config['feature_engineering']['text_features']['Pre-symptoms'] # Whether to expand the column from csv text into binary columns for each value
comorbidity = config['feature_engineering']['text_features']['Comorbidity'] # Whether to expand the column from csv text into binary columns for each value
regular_meds = config['feature_engineering']['text_features']['Regular_meds'] # Whether to expand the column from csv text into binary columns for each value
min_count = config['feature_engineering']['min_count'] # Minimum frequency of a csv value in the column to convert it into a binary column (otherwise excluded from the dataset)

# Override text to binary expansions if validating
# IMPROVE currently I don't have equivalent columns set for isaric so we switch off these settings if validation is on.
#  Can be added later but requires finding/renaming matching columns if each t2b bool is enabled, and updating the rest of scripts/other scripts to acknowledge the added cols
if validate:
    medication_48hr = False
    pre_symptoms = False
    comorbidity = False
    regular_meds = False
    print("\nDisabling text-to-binary conversion due to validation being set to True.")

np.random.seed(42)

########################################################################################################################
# todo - change cwd to github folder; currently assumes it's in there already. maybe for other scripts. including pipeline.py

# Read in data #IMPROVE avoid hardcoding
quant_file = Path(__file__).parent / "Surrey_Files" / "KR_Covid_DIA_Pt_gene_Serum30_Report_Protein Quant (Pivot).xls"
s_meta_file = Path(__file__).parent / "Surrey_Files" / "Surrey_Metadata_master_spreadsheet_130622_edit2.csv"
isaric_file = Path(__file__).parent / "ISARIC_Files" / "ISARIC.csv"
phosp_file = Path(__file__).parent / "ISARIC_Files" / "PHOSP Metadata Master.csv" # Contains some ISARIC data
quant = pd.read_csv(quant_file, sep='\t').T # Transpose so sample IDs are rows
s_meta = pd.read_csv(s_meta_file)
isaric = pd.read_csv(isaric_file, index_col=0) # Avoid creating Unnamed: 0 column
phosp = pd.read_csv(phosp_file, index_col=0, low_memory=False)

# Create output directories for the data
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    training_data = 'training_data' # Combine with other training graphs if using training data
    validation_data = 'validation_data' # Combine with other validation graphs if using training data
    os.makedirs(training_data, exist_ok=True)
    os.makedirs(validation_data, exist_ok=True)
else: # Put into input storage folder to prevent overwriting
    training_data = f'inputs/{run_name}/training_data'
    validation_data = f'inputs/{run_name}/validation_data'
    os.makedirs(training_data, exist_ok=True)
    os.makedirs(validation_data, exist_ok=True)

# Create output directory for the graphs
if not args.from_pipeline: # If calling as a standalone script, save to the current working directory
    training_graphs = 'training_graphs' # Combine with other training graphs if using training data
    validation_graphs = 'validation_graphs' # Combine with other validation graphs if using training data
    os.makedirs(training_graphs, exist_ok=True)
    os.makedirs(validation_graphs, exist_ok=True)
else: # Put into input storage folder to prevent overwriting
    training_graphs = f'inputs/{run_name}/training_graphs'
    validation_graphs = f'inputs/{run_name}/validation_graphs'
    os.makedirs(training_graphs, exist_ok=True)
    os.makedirs(validation_graphs, exist_ok=True)

# Quant data preprocessing
quant.columns = quant.iloc[0] # Set protein names (now row 0) as column headers
quant = quant[1:] # Remove the first (now duplicated) row

### INVESTIGATE SAMPLE IDS - what samples do we have in each file and is this consistent? ##############################
if sample_inves_1:
    print("\nCheck the number of Surrey samples in the metadata:")
    print(len(s_meta))
    print("\nCheck the number of ISARIC samples in the metadata:")
    print(len(isaric))

# Removing duplicates in the Surrey metadata file # TODO could check quant too. And maybe go over the dropbox metadata file for more potential issues. Including these reenlistements
#  Several participants were re-enlisted: 403>434, 404>411, and 405>410. This leads to duplicate sample IDs (434 is
    #  slightly different due to one ID missing the middle number, but should still be removed as a duplicate)
    # Therefore we will remove the outdated MABRA IDs
# Define the conditions for rows to remove
conditions_to_remove = (
    ((s_meta['Sample'] == '434-28_220321') & (s_meta['MABRA ID'] == 403)) |
    ((s_meta['Sample'] == '411-0_010321') & (s_meta['MABRA ID'] == 404)) |
    ((s_meta['Sample'] == '410-0_010321') & (s_meta['MABRA ID'] == 405))
)
# Remove the rows by keeping only those that do not meet the conditions
s_meta = s_meta[~conditions_to_remove]

# Convert to a set for later analysis
s_meta_samples = set(s_meta['Sample'])

# Removing duplicates in the ISARIC metadata file
    # CCPI307B is listed twice. The data appears to be the same patient from two different time points, however as it is not possible to
    # identify which is the day 1/admission timepoint (or discharge in the equivalent phosp data which is the earliest available) both samples will
    # be dropped.
to_remove = isaric['pseudo_id'] == 'CCPI307B;_RHM01-0062;_SM2004042369'
isaric = isaric[~to_remove]

# Extract SID from isaric file and set as index (first column contains multiple CCP IDs associated with multiple time points; we want the first only)
isaric['SID'] = (isaric['pseudo_id'].str.split(';').str[0].str.strip())
isaric = isaric.drop(columns='pseudo_id') # Remove the old ID column
isaric = isaric.set_index('SID', drop=True)

# Now to investigate the metadata vs. quant file
if sample_inves_2:
    # Check the number of samples in quant
    print("Number of unique sample timepoints in quant:")
    print(len(quant))

### Rename quant indexes to a simpler sample ID
samples = quant.index.copy() # Copy of original values
# For ISARIC - strip to CCPXXXXX ID only
isaric_SID = samples.str.split('_').str[3].str.replace(".raw.PG.Quantity", "", regex=False) # Extract samples and relevant SID for ISARIC only
isaric_mask = isaric_SID.str.startswith("CCP", na=False)
isaric_ids = isaric_SID[isaric_mask]
# For Surrey
before_samples = quant.index
samples_split = before_samples.str.split('_')
start_id = samples_split.str[3].str.slice(0, 3)  # Take first three characters because some have -[timepoint] added
end_id = samples_split.str[4].fillna('')  # Some samples won't have this, in which case it creates nan
surrey_ids = start_id + "_" + end_id
surrey_ids = surrey_ids.str.replace('.raw.PG.Quantity', '', regex=False)
final_ids = surrey_ids.to_series()
final_ids[isaric_mask] = isaric_ids # Overwrite with isaric values where necessary
final = pd.Index(final_ids.values)
quant.index = final_ids.values  # Now replace in quant df with a simplified version

# Investigate the (processed) IDs to find which samples are relevant
quant_ids = quant.index.to_series()
if sample_inves_2:
    quant_ids.to_csv(f"{training_data}/Quant_IDs.csv", index=False, header=False) # For easier examination
    print("Length before processing:", len(quant_ids))
    # Result: Using the Meta_plates.csv file, we can remove samples that belong to other datasets. As these SIDs are in
    # a different format, for simplicity the trends in SIDs are simply reported here, although ideally this would be
    # validated more extensively and automated. # IMPROVE

# Extract Surrey IDs from the dataset
# Rule 1: SIDs starting with 'PHS' are from the PHOSP data set, not Surrey, and can be removed
quant_ids_1 = quant_ids[~quant_ids.str.startswith('PHS_')]
if sample_inves_2:
    print("Length after PHS removal:", len(quant_ids_1)) # 706 PHS samples removed
# Rule 2: SIDs starting with 'QC' are quality control, not Surrey, and can be removed
quant_ids_2 = quant_ids_1[~quant_ids_1.str.startswith('QC')]
if sample_inves_2:
    print("Length after QC removal:", len(quant_ids_2))  # 44 QC samples removed
# Rule 3: SIDs starting with 'HB' are Surrey samples that are not to be included according to Meta_Plates.csv
quant_ids_3 = quant_ids_2[~quant_ids_2.str.startswith('HB')]
if sample_inves_2:
    print("Final length after HB removal:", len(quant_ids_3))  # 13 HB samples removed
# Rule 4: SIDs starting with 'CCP' are from ISARIC, not Surrey, and can be removed if not validating
quant_ids_4 = quant_ids_3[~quant_ids_3.str.startswith('CCP')]
# Determine ISARIC IDs for validation
quant_ids_5 = quant_ids_3[quant_ids_3.str.startswith('CCP')]
if not validate: # If not validating, discount ISARIC
    if sample_inves_2:
        print("Length after CCP removal:", len(quant_ids_4))  # 66 ISARIC samples removed
        # Result: 204 usable Surrey samples in the quant data
else:
    if sample_inves_2:
        print("Length of final Surrey samples:", len(quant_ids_4)) # 204 samples left
        print("Length of ISARIC samples:", len(quant_ids_5))  # 66 ISARIC samples left

# Filter to just the samples from the relevant dataset:
quant_surrey = quant.loc[quant_ids_4]
quant_isaric = quant.loc[quant_ids_5]

# Check that the SIDs are as expected
if sample_inves_2:
    check_abnormal_SIDs(quant_surrey, before_samples, final, 10)
    check_abnormal_SIDs(quant_isaric, before_samples, final, 8)
    # Result: For Surrey, in this case there is only one abnormal samples, and the 'SID before' shows us that this is due to a
    # hyphenation error in the original data. Therefore, we can simply correct the faulty column.
    # For ISARIC. 2 samples are of length 9, but 'CCPI1055C' is found in the dataset as expected so is not abnormal.
    #  CCPI1052C is not found in the dataset, but will be dropped when filtering for initial timepoints.

# Fix the hyphenation error in one Surrey sample
quant_surrey = quant_surrey.rename(index={'293_': '293_240620'})

# Check the data now looks correct:
if sample_inves_2:
    abnormal_quant_SIDs = quant_surrey[quant_surrey.index.astype(str).str.len() != 10].iloc[:, 0:5]
    print("Remaining abnormalities:", len(abnormal_quant_SIDs))

# Add a column to the Surrey metadata sheet with modified sample IDs (some inconsistencies between quant and Surrey metadata so simplifying in each by removing -[timepoint])
samples_mod = s_meta['Sample']
if sample_inves_2:
    print("Surrey meta samples before underscore removal:", len(samples_mod))
    print("Surrey meta samples containg hyphens:", len(samples_mod[samples_mod.str.contains("-")]))
samples_mod = samples_mod.str.replace(r'-\d+(?=_)', '', regex=True) # Remove '-'
# Check hyphen removal is done correctly
if sample_inves_2:
    print("Meta samples after hymphen removal:", len(samples_mod))
    print("Meta samples containg hyphens after removal:", len(samples_mod[samples_mod.str.contains("-")]))
    # 0 after = correctly processed

# Remove trailing _ from SIDs
#  Some IDs have a trailing _ in the sample ID which appears accidental; remove this
samples_mod = samples_mod.str.rstrip('_')
s_meta['Sample Modified'] = samples_mod

# Check that the metadata SIDs are all in the expected format
if sample_inves_2:
    #print(samples_mod.unique()) # Can print unique values but after examining, the main issue is length of SID
    print(s_meta['Sample Modified'].astype(str).str.len().value_counts())
    # Result: 356 samples are the expected 10 characters, and 5 are only 3 characters long (IDs: 315, 316, 409, 458, 522)
abnormal_meta_SIDs = s_meta[s_meta['Sample Modified'].astype(str).str.len() == 3]
# Print rows to determine how to handle abnormal SIDs
if sample_inves_2:
    if len(abnormal_meta_SIDs) > 0:
        print("Rows with abnormal metadata SIDs:\n", abnormal_meta_SIDs)
        # Result: These rows have a lot of missing data, so can be removed instead of processed further.
# Remove abnormal rows
s_meta = s_meta[s_meta['Sample Modified'].astype(str).str.len() != 3] # Now at 356 total samples
# Notify of removal
if len(abnormal_meta_SIDs) > 0:
    print("WARNING: Some rows were removed due to abnormal SID length. Enable the sample_inves_2 bool to examine these\
 rows before removal if not already done.")

# Calculate uniques/overlaps
calculate_overlaps(quant_surrey.index, s_meta['Sample Modified'], sample_inves_3) # Need to use the modified samples IDs to match quant
calculate_overlaps(quant_surrey.index, isaric.index, sample_inves_3)
    # Result: 152 unique samples in the metadata for Surrey
    # All samples overlap in ISARIC

# Notes on Surrey data: Mabra ID 247 time point 260520 was found in the Surrey metadata and meta plates spreadsheet but
# not in quant. This can be disregarded as missing MS data. (so 151 remaining)

# Investigation of Meta_Plates.csv also shows there are three samples that are in the metadata file that were not processed:
# 369_130121, 370_130121, 373_150121

# This leaves 148 unaccounted for, although invesitgation of the original metadata spreadsheet appears to show that these
# samples are missing a lot of data, so perhaps were taken out of the study. As we need quant data to proceed with analysis,
# we shall therefore use the 204 samples that we have MS data for.

### COMBINE ISARIC AND PHOSP METADATA ##################################################################################
#  Some samples are covered in each, and the PHOSP data takes samples at admission point which is more comparable to
#  Surrey data and therefore preferred
phosp = phosp[phosp['redcap_event_name'] == 'Hospital Discharge'] # Filter to 'Hospital Discharge' time points only
isaric = isaric.drop(columns='age') # Drop to avoid duplicate columns names when merging
isaric = isaric.reset_index().rename(columns={'index': 'SID'}) # Move index to column so it can be preserved and set later
isaric_all = pd.merge(isaric, phosp, on='phosp_id', how='left') # Merge PHOSP and ISARIC
isaric_all = isaric_all.set_index('SID')

### COMBINE QUANT AND META DATA ########################################################################################
# Surrey
s_meta_mod = s_meta.set_index('Sample Modified') # Define new index with modified sample names to match quant
s_meta_mod.columns = s_meta_mod.columns.str.replace(r"(Plasma - Ig[AGM] Anti-RBD Concentration \(ng/).*", r"\1l)",
                                                    regex=True) # Adjust some faulty column names for Surrey (have an unknown symbol)
s_meta_mod.columns = s_meta_mod.columns.str.strip()  # Remove trailing whitespace
merged_surrey = s_meta_mod.join(quant_surrey, how='inner')
merged_surrey.to_csv(f"{training_data}/Surrey_data_combined_all.csv") # Note this has the unmodified column names (e.g. whitespace)
# ISARIC
merged_isaric = isaric_all.join(quant_isaric, how='inner')
merged_isaric.to_csv(f"{validation_data}/ISARIC_data_combined_all.csv")

### DROP COLUMNS FROM TRAINING DATA ####################################################################################
# Remove whitespace surrounding columns
merged_surrey.columns = merged_surrey.columns.str.strip() # Already done for Surrey metadata but added for quant data
merged_isaric.columns = merged_isaric.columns.str.strip()

# Change Sample column to be the updated SIDs currently stored as row indexes (so can reset index later)
# merged['Sample'] = merged.index # Note I later remove this column but left this in the code in case it's useful later # IMPROVE - remove if not needed
if sample_inves_4:
    print("Is the data frame the length we expect (# of overlapping samples)?")
    print(len(merged_surrey))  # Answer: Yes

    # What columns might we want to remove from the metadata?
    print("\nColumns in the Surrey metadata:")
    print(merged_surrey.columns[0:71])
    # TODO: List of columns I'm not sure on the meaning of, can I check? Might need to remove some if irrelevant
    # 'Chol', 'Airway Disease', 'For escalation? (Y/N)', 'PBMC No Calculation', 'Saliva - untargeted metabolomics' (and 2 similar).

# List unecessary columns to remove from the Surrey dataset
remove_cols = ['Sample', # Stored as row indexes
               'Agreed to be contacted for future studies', # Irrelevant to health
               'Cell Pellet', # Sample handling irrelevant to health
               'Clot', # Sample handling irrelevant to health
               'Clot Formation', # Sample handling irrelevant to health
               'Date', # Only shows date of first sample I believe, not current sample date
               'Date of Hospital Admission', # Not relevant to health
               'Date of MOST RECENT Covid Positive Swab', # Not relevant to health
               'Date of first Mabra samples collected',
               'Date of vaccination', # TODO this column is useful for time since vaccination and Y/N vaccinated, however for now removing as 1) it's highly missing anyway and 2) Need to do more processing before inclusion. But come back to
               'Ethnicity', # Largely biased towards white so would be misleading
               #'Height (cm)', # TODO on second thought kept this in, but not sure if it is eg correlated with Gender and leads to bias (dropped with isaric regardless)
               'Hospital site', # TODO Possibly could impact care but I think a confounding feature? Maybe try with and without
               'MABRA ID', # Irrelevant
               'MABRA ID.1', # Irrelevant
               'Plasma 1', # Irrelevant
               'Plasma 2', # Irrelevant
               'Plasma 3', # Irrelevant
               'Pt MABRA ID', # Irrelevant
               'Saliva - untargeted metabolomics', # Not sure but think irrelevant. Has high missingness so filtered regardless.
               'Saliva 1', # Irrelevant
               'Saliva 2', # Irrelevant
               'Saliva 3', # Irrelevant
               'Sebum - untargeted lipidomics', # Not sure but think irrelevant. Has high missingness so filtered regardless.
               'Serum - targeted metaboloimcs', # Not sure but think irrelevant. Has high missingness so filtered regardless.
               'Serum 1', # Irrelevant
               'Serum 2', # Irrelevant
               'Serum 3', # Irrelevant
               'Serum 4', # Irrelevant
               'Serum 5', # Irrelevant
               'Serum 6', # Irrelevant
               'Serum 7', # Irrelevant
               'Serum 8', # Irrelevant
               'Serum 9', # Irrelevant
               'Time between pos covid test and first sample',
               # TODO this could be a useful feature, but will be biased against future time points. Could be omitted for those and
               #  imputed, although it might be filtered for 30% missingness in that case. Could also use various features to calculate
               #  time since infection/positive for each time point, but if the target variable is only recorded at D0 then I think it's moot.
               #'Weight (kg)', #TODO kept in for now but should test with and without
               'Dates of repeat samples', # TODO might be useful for getting timepoint data alongisde other cols, but for now removing
               #'Days_between' #TODO can drop this if doing D0 only - take out of remove_isaric if so
               ]

# If not doing text to binary conversion (which is also switched off when validating), add to the removal list:
if not medication_48hr:
    remove_cols.append('Medication taken in 48 hours prior to sample')
if not pre_symptoms:
    remove_cols.append('Nature of pre admission symptoms')
if not comorbidity:
    remove_cols.append('Other')
if not regular_meds:
    remove_cols.append('Regular medications')

# Drop the columns from the Surrey dataset
merged_surrey = merged_surrey.drop(columns=remove_cols)

# Remove Surrey columns that are incompatible with the ISARIC validation set (i.e. not found in both)
remove_isaric = ['Chol', # Unsure on this columns meaning, but I can't see anything that might correspond in ISARIC
                 'Airway Disease', # ISARIC as info on COPD and asthma but not OSA or pulmonary fibrosis
                 'BMI', # Exists in PHOSP but is NaN for all
                 'Clinical Covid (Y/N)', # Couldn't find relevant column in ISARIC data
                 'For escalation? (Y/N)', # Couldn't find relevant column in ISARIC data
                 'PBMC No Calculation', # Couldn't find relevant column in ISARIC data
                 'PBMC No', # Couldn't find relevant column in ISARIC data
                 'MADU admission', # Couldn't find relevant column in ISARIC data
                 'Survived Admission', # Couldn't find relevant column in ISARIC data
                 'Weight (kg)', # Exists in PHOSP but is NaN for all
                 'Days_between' # Not relevant as only doing the first time point (day 1/admission where specific)
                 # WARNING: If I do decide to include some texttobinary (see below) columns in the validation set, edit this
                 ]

if validate:
    merged_surrey = merged_surrey.drop(columns=remove_isaric)

### HANDLE TEXT COLUMNS ################################################################################################
# Some columns have comma-separated values in a list for each cell. These can be split up into binary Y/N for each, but
# it adds high dimensionality to the data. Therefore a bool has been set up to optionally include to examine if
# inclusion helps or hinders results.
added_metadata = 0 # Tracks added metadata columns vs original dataset, so more can be added
# IMPROVE: As noted above, only do this section if not validating as I haven't yet identified the equivalent columns in ISARIC, and want to test it with Surrey to see which columns (survive FS) are worth finding and converting.
### Clean up typos in the columns
typos_path = Path("typos.yaml")
with open(typos_path, "r") as f:
    typos = yaml.safe_load(f)

# Medications
if medication_48hr:
    typos_meds = typos['medication_48hr']
    for typo, fix, in typos_meds.items():
        merged_surrey = replace_values(merged_surrey, 'Medication taken in 48 hours prior to sample', typo, fix)
# Symptoms prior to admission
if pre_symptoms:
    typos_symp = typos['pre_symptoms']
    for typo, fix, in typos_symp.items():
        merged_surrey = replace_values(merged_surrey, 'Nature of pre admission symptoms', typo, fix)
# Other diseases
if comorbidity:
    typos_como = typos['comorbidity']
    for typo, fix, in typos_como.items():
        merged_surrey = replace_values(merged_surrey, 'Other', typo, fix)
# Regular medications
if regular_meds:
    typos_regs = typos['regular_meds']
    for typo, fix, in typos_regs.items():
        merged_surrey = replace_values(merged_surrey, 'Regular medications', typo, fix)

# Clean and split strings for each text column in the surrey data
if medication_48hr:
    merged_surrey, added_cols = text_to_binary(merged_surrey, 'Medication taken in 48 hours prior to sample', 'Medication_48hr', training_data, min_count)
    added_metadata += (added_cols - 1) # -1 due to dropping of original column (note that if used in future to represent actual added columns and not the change in columns, +1 needs to be added)
if pre_symptoms:
    merged_surrey, added_cols = text_to_binary(merged_surrey, 'Nature of pre admission symptoms', 'Pre-symptoms', training_data, min_count)
    added_metadata += (added_cols - 1)
if comorbidity:
    merged_surrey, added_cols = text_to_binary(merged_surrey, 'Other', 'Comorbidity', training_data, min_count)
    # Manually drop t2dm column as also recorded elsewhere in dataset
    if 'Comorbidity: t2dm' in merged_surrey.columns:
        merged_surrey = merged_surrey.drop(columns='Comorbidity: t2dm')
        added_cols -= 1
        print(f"Comorbidity: t2dm has been dropped due to the pre-existing T2DM column. The other {added_cols} added columns remain.")
    added_metadata += (added_cols - 1)

if regular_meds:
    merged_surrey, added_cols = text_to_binary(merged_surrey, 'Regular medications', 'Regular_meds', training_data, min_count)
    added_metadata +=  (added_cols - 1)

### CLEAN UP ISARIC COLUMNS ############################################################################################
# Rename Columns in the ISARIC dataset to match Surrey (note: only done for those that pass the filtering stage later in the script) #IMPROVE: this has to be in the same order as merged_isaric (or at least the last column) or remaining_meta won't work
isaric_cols = {'age_admission' : 'Age',
               'infiltrates_faorres.day1' : 'Bilateral CXR changes', # Infiltrates are determined from x-rays (I believe) so these are somewhat analogous, if imperfect
               'crf1a_o2_cpapv' : 'CPAP',
               'swab_pcr_result' : 'Covid Positive Hospital Swab (Y/N)',
               'crf1a_crp' : 'CRP',
               'crf1a_symptom_duration' : 'Duration of Pre-Admission Symptoms',
               'crf1a_eosinophil_count' : 'Eosinophils',
               'sex' : 'Gender',
               'crf3a_rest_height' : 'Height (cm)',
               'hypertension_mhyn' : 'HTN',
               'icu_hoterm' : 'ICU admission',
               'crf1a_com_card_ihd' : 'IHD',
               'crf1a_haema_lymph' : 'Lymphocytes',
               'crf1a_o2_supp' : 'O2 req.',
               'smoking_mhyn' : 'Smoking Status', # Assumed that the Non/Ex/Current smoker levels correspond to 0/1/2 in Surrey
               'diabetes_type_mhyn' : 'T2DM',
               }
               # Note: Weight is in the PHOSP metadataset, but is NaN for all values. BMI therefore also can't be calculated.

# Rename ISARIC columns
merged_isaric.rename(columns=isaric_cols, inplace=True)
# Drop extraneous columns in ISARIC
cols_to_keep = list(isaric_cols.values()) + quant.columns.to_list()
cols_to_keep_2 = [c for c in cols_to_keep if pd.notna(c) and (c in merged_isaric.columns)] # Also remove NaN simultaneously (causes problems with duplicate nan values)
merged_isaric = merged_isaric[cols_to_keep_2]

# List columns that have been removed
if not validate:
    removed_cols = remove_cols
else:
    removed_cols = remove_cols + remove_isaric

# Calculate the number of columns of metadata remaining
# Surrey
meta_cols_original = len(s_meta_mod.columns) # Get original columns count
if added_metadata: # If text to binary conversion has been done add the new columns to the count (already has -1 from original column factored in)
    meta_cols_added = meta_cols_original + added_metadata
else:
    meta_cols_added = meta_cols_original
meta_cols_surrey = meta_cols_added - len(removed_cols) # Subtract removed columns
meta_cols_names = merged_surrey.columns[0:meta_cols_surrey] # Store for processing later in the script

# ISARIC
meta_cols_isaric = len(isaric_cols)

### CLEAN UP REMAINING COLUMNS #########################################################################################
# Check the column values are consistent/correct; i.e. which need to be cleaned (done in the next section)

# Examine remaining columns
if sample_inves_4:
    print(f"\nRemoving {len(removed_cols)} columns from the Surrey dataset. {meta_cols_surrey} columns remaining.\n")
    print(f"ISARIC metadata columns remaining after removal: {meta_cols_isaric}")

    # Note: Some Surrey columns are not processed into a machine readable format or could benefit from further processing. However,
    #  as many are later filtered due to missingness they are left as is.
    #TODO: Other observations on columns (and actions needed)
    #  For columns like T2DM, is that correlated to covid severity/oxygen need? Otherwise could be biasing data towards majority
    #  Time point information could be deduced from the features, however I need to decide if I'm only using D0 first as that's when the metadata is recorded for

# Settings to check columns
check_only_one = False # Change this if checking a specific column only, otherwise all are printed
selected_column = ['CPAP']  # Edit this column name as needed if checking single columns
check_training_set = False # Enable to check Surrey data
check_validation_set = False # Enable to check isaric data

# Check values
columns_to_check_surrey = check_columns(merged_surrey, meta_cols_surrey, removed_cols, check_only_one, selected_column, check_training_set)
columns_to_check_isaric = check_columns(merged_isaric, meta_cols_isaric, removed_cols, check_only_one, selected_column, check_validation_set)

    # TODO Some possible data handling that can be done but does not currently affect the result (Surrey):
    #  the Ig columns I won't clean because they're filtered out for missingness. But should clean if they're needed.
    #  CXR comments will be filtered as missing data so not cleaning is not a problem - but potentially useful information could be extracted (e.g. typical vs not typical COVID).
    #     However, 'no comment' vs missing data is hard to distinguish which could lead to bias

### Clean up values for Surrey metadata
# Airway Disease
merged_surrey.replace({'Airway Disease' : ['N']}, '0', inplace = True)
merged_surrey.replace({'Airway Disease' : ['Asthma']}, '1', inplace = True)
merged_surrey.replace({'Airway Disease' : ['COPD']}, '2', inplace = True)
merged_surrey.replace({'Airway Disease' : ['OSA']}, '3', inplace = True)
merged_surrey.replace({'Airway Disease' : ['Pulmonary fibrosis']}, '4', inplace = True)
# CPAP
merged_surrey.replace({'CPAP' : ['N ', ' N']}, 'N', inplace = True)
merged_surrey.replace({'CPAP' : ['Y  ']}, 'Y', inplace = True)
# CRP
merged_surrey.replace({'CRP' : ['<4.0', '<4']}, '4', inplace = True) # IMPROVE To keep this numerical I changed the value as listed. But I am not confident this is the 'correct' approach.
# Covid Positive Hospital Swab (Y/N)
merged_surrey.replace({'Covid Positive Hospital Swab (Y/N)' : ['Not done ']}, np.nan, inplace = True)
merged_surrey.replace({'Covid Positive Hospital Swab (Y/N)' : ['N - previously pos in ICU']}, 'Inconclusive', inplace = True) # Combined this data point with the other 'Inconclusive' as miceforest recommends grouping very rare categories
# Duration of Pre-Admission Symptoms
merged_surrey.replace({'Duration of Pre-Admission Symptoms' : ['23/05/2020']}, np.nan, inplace = True) # IMPROVE could handle more robustly for future samples - in this case the patient wasn't admitted so should be NaN.
# For escalation? (Y/N)
merged_surrey.replace({'For escalation? (Y/N)' : ['Yes']}, 'Y', inplace = True)
merged_surrey.replace({'For escalation? (Y/N)' : ['No']}, 'N', inplace = True)
# Gender
merged_surrey.replace({'Gender' : ['Male', 'Male ']}, 'M', inplace = True)
merged_surrey.replace({'Gender' : ['Female']}, 'F', inplace = True)
# HTN
merged_surrey.replace({'HTN' : ['y']}, 'Y', inplace = True)
# ICU admission
merged_surrey.replace({'ICU admission' : ['Yes']}, 'Y', inplace = True)
merged_surrey.replace({'ICU admission' : ['No']}, 'N', inplace = True)
# IHD
merged_surrey.replace({'IHD' : ['Atrial fibrillation, heart failure', 'Coronary artery disease, heart failure']}, 'Y', inplace = True)
# MADU admission
merged_surrey.replace({'MADU admission' : ['Yes', 'yes']}, 'Y', inplace = True)
merged_surrey.replace({'MADU admission' : ['No']}, 'N', inplace = True)
# PBMC No
merged_surrey.replace({'PBMC No' : ['Too many to count']}, 600, inplace = True) # IMPROVE this is estimated based on the highest value being 590, so as not to lose the fact that it's high vs putting NaN. But really this should be determined using the instrument specs.
# Survived Admission
merged_surrey.replace({'Survived Admission' : ['Y ']}, 'Y', inplace = True)
merged_surrey.replace({'Survived Admission' : ['N ']}, 'N', inplace = True)
if not validate: # Otherwise these columns are dropped - PMBC No Calculation is the only one that doesn't silently error so must be handled separately
    # PBMC No Calculation
    merged_surrey.replace({'PBMC No Calculation': [' N/A ']}, np.nan, inplace=True)  # TODO I think this is a valid approach; N/A is used when PMBC is either empty or too high so it does combine those two which might not be ideal, but I want to keep ordinality
    merged_surrey['PBMC No Calculation'] = pd.to_numeric(merged_surrey['PBMC No Calculation'].str.replace(',', ''), errors='coerce')  # Convert strings to numeric, else NaN - IMPROVE beware if doing for new data to not accidentally convert any text to NaN

### Clean up values for ISARIC data - change to match Surrey
# Bilateral CXR changes
merged_isaric.replace({'Bilateral CXR changes': ['NO']}, 'N', inplace=True)
merged_isaric.replace({'Bilateral CXR changes': ['YES']}, 'Y', inplace=True)
# CPAP
merged_isaric.replace({'CPAP': ['No']}, 'N', inplace=True)
merged_isaric.replace({'CPAP': ['Yes']}, 'Y', inplace=True)
merged_isaric.replace({'CPAP': ['N/K']}, np.nan, inplace=True)
# Covid Positive Hospital Swab (Y/N)
merged_isaric.replace({'Covid Positive Hospital Swab (Y/N)': ['Positive']}, 'Y', inplace=True)
# Gender
merged_isaric.replace({'Gender': ['Male', 'Male ']}, 'M', inplace=True)
merged_isaric.replace({'Gender': ['Female']}, 'F', inplace=True)
# HTN
merged_isaric.replace({'HTN': ['NO']}, 'N', inplace=True)
merged_isaric.replace({'HTN': ['YES']}, 'Y', inplace=True)
# ICU admission
merged_isaric.replace({'ICU admission': ['No']}, 'N', inplace=True)
merged_isaric.replace({'ICU admission': ['Yes']}, 'Y', inplace=True)
# IHD
merged_isaric.replace({'IHD': ['No']}, 'N', inplace=True)
merged_isaric.replace({'IHD': ['Yes']}, 'Y', inplace=True)
# Smoking Status
merged_isaric.replace({'Smoking Status': ['Never Smoked']}, '0', inplace=True)
merged_isaric.replace({'Smoking Status': ['Former Smoker']}, '1', inplace=True)
merged_isaric.replace({'Smoking Status': ['Yes']}, '2', inplace=True)
merged_isaric.replace({'Smoking Status': ['N/K']}, np.nan, inplace=True)
# T2DM
merged_isaric.replace({'T2DM': ['NO']}, 'N', inplace=True)
merged_isaric.replace({'T2DM': ['2']}, 'Y', inplace=True)
merged_isaric.replace({'T2DM': ['1']}, 'N', inplace=True)
# O2 req.
merged_isaric.replace({'O2 req.': ['Yes']}, 'Y', inplace=True)
merged_isaric.replace({'O2 req.': ['No']}, 'N', inplace=True)
merged_isaric.replace({'O2 req.': ['N/K']}, np.nan, inplace=True)

# Check the data is corrected
if sample_inves_6:
    if check_training_set:
        for column in columns_to_check_surrey:
            print("\nValue counts fixed:\n", merged_surrey[column].value_counts())
            print("Uniques fixed:\n", merged_surrey[column].unique())
    if check_validation_set:
        for column in columns_to_check_isaric:
            print("\nValue counts fixed:\n", merged_isaric[column].value_counts())
            print("Uniques fixed:\n", merged_isaric[column].unique())

# Remove columns with no data - the Surrey dataset also contains some duplicated NaN headers with no data which are
#  simultaneously removed
# Print the results before removal
if sample_inves_6:
    check_empty_cols(merged_surrey, 'Surrey', 'before')
    check_empty_cols(merged_isaric, 'ISARIC', 'before')

# Remove empty columns (all NaN values except header)
merged_surrey.dropna(how='all', axis=1, inplace=True)
merged_isaric.dropna(how='all', axis=1, inplace=True)

if sample_inves_6:
    check_empty_cols(merged_surrey, 'Surrey', 'after')
    check_empty_cols(merged_isaric, 'ISARIC', 'after')

# Save to csv # IMPROVE could remove this if it's not needed/later csvs are more useful
merged_surrey.to_csv(f"{training_data}/Surrey_data_selected.csv")
merged_isaric.to_csv(f"{validation_data}/ISARIC_data_selected.csv")

### CLEAN UP SAMPLE ROW MISSINGNESS ####################################################################################
# Remove any rows that are empty for O2 req, the target
merged_surrey = merged_surrey[merged_surrey['O2 req.'].notna()] # Removes 20 values from Surrey data
merged_isaric = merged_isaric[merged_isaric['O2 req.'].notna()] # Removes 2 for ISARIC

# Plot missingness across rows (note: no longer removed in order to preserve the most samples)
merged_null_before_surrey = plot_row_missingness(merged_surrey, meta_cols_surrey, sample_inves_7, training_graphs, 'Surrey')
merged_null_before_isaric = plot_row_missingness(merged_isaric, meta_cols_isaric, sample_inves_7, validation_graphs, 'ISARIC')

### CLEAN UP COLUMN MISSINGNESS ########################################################################################
# Plot missingness before filtering (full dataset) #TODO some plots are blank - possibly due to too much data
plot_missingness_msno(merged_surrey, 'Surrey', meta_cols_surrey, training_graphs)
plot_missingness_msno(merged_isaric, 'ISARIC', meta_cols_isaric, validation_graphs)

# Note: columns with 100% missingness have already been removed prior to this

# Investigate null values in each column and removing columns with over 30% missingness
merged_surrey = investigate_null(merged_surrey, 'Surrey', merged_null_before_surrey, sample_inves_7, training_data)
merged_isaric = investigate_null(merged_isaric, 'ISARIC', merged_null_before_isaric, sample_inves_7, validation_data)

# Note: final calculation and plot is done below after removing non-overlapping columns

### CONVERT TO NUMERICAL ###############################################################################################
# Initialise list of numerical vs categorical
numerical_surrey, categorical_surrey = categorise_cols(merged_surrey, sample_inves_8)
numerical_isaric, categorical_isaric = categorise_cols(merged_isaric, sample_inves_8)

# Check if any numerical columns had values converted to NaN
numerical_check_nan(merged_surrey, numerical_surrey, categorical_surrey, sample_inves_8, 'Surrey')
numerical_check_nan(merged_isaric, numerical_isaric, categorical_isaric, sample_inves_8, 'ISARIC')

### DROP EXTRA COLUMNS #################################################################################################
# TODO: Several columns are dropped from ISARIC due to missingness in the above preprocessing, which then causes an
#  error when training the model as they are found in Surrey when training, but not ISARIC. Ideally after the initial
#  filtering both the Surrey and ISARIC datasets would be checked for any columns not in either dataset and those are
#  dropped before proceeding to the next stage. This should be implemented properly but as a temporary fix, the
#  incompatible columns are dropped from the Surrey dataset here. (ISARIC has to be done first due to column renaming -
#  otherwise the two aren't comparable even if the same columns technically exist in each).

# Drop the columns from Surrey that have been filtered for missingness in ISARIC, and vice versa
if validate:
    shared_cols = merged_surrey.columns.intersection(merged_isaric.columns)
    if sample_inves_9:
        print(f"Number of columns shared between datasets: {len(shared_cols)}")
    # Subset datasets to shared columns only
    old_surrey = merged_surrey.copy()
    old_isaric = merged_isaric.copy()
    merged_surrey = merged_surrey[shared_cols]
    merged_isaric = merged_isaric[shared_cols]

    # Determine dropped columns
    dropped_surrey = set(old_surrey.columns) - set(shared_cols)
    dropped_isaric = set(old_isaric.columns) - set(shared_cols)

    # Print dropped columns
    if sample_inves_9:
        print(f"{len(dropped_surrey)} columns dropped from Surrey:")
        print(dropped_surrey)

        print(f"{len(dropped_isaric)} columns dropped from ISARIC:")
        print(dropped_isaric)

# Calculate how many metadata columns remain and plot with missingno
meta_cols_surrey = remaining_meta(meta_cols_names.tolist(), merged_surrey, sample_inves_7, training_graphs)
meta_cols_isaric = remaining_meta(isaric_cols.values(), merged_isaric, sample_inves_7, validation_graphs)

# Update config for later access
with open(config_path, "w") as f:
    config["general"]["training_meta_cols"] = meta_cols_surrey
    config["general"]["validation_meta_cols"] = meta_cols_isaric
    yaml.dump(config, f, sort_keys=False)

### LOG2 TRANSFORM THE PROTEOMICS DATA #################################################################################
merged_surrey.iloc[:,meta_cols_surrey:] = np.log2(merged_surrey.iloc[:,meta_cols_surrey:] + 1e-6) # IMPROVE This log transformation is not noted in the data headers anywhere but is used in all future processing
merged_isaric.iloc[:,meta_cols_isaric:] = np.log2(merged_isaric.iloc[:,meta_cols_isaric:] + 1e-6)

# Save final datasets to csv
merged_surrey.to_csv(f"{training_data}/Surrey_final.csv")
if validate:
    merged_isaric.to_csv(f"{validation_data}/ISARIC_final.csv")

### PLOT CLASS DISTRIBUTION ############################################################################################
plot_class_distribution(merged_surrey, training_graphs, 'Surrey')
plot_class_distribution(merged_isaric, validation_graphs, 'ISARIC')

### FILTER TO DAY 0 TIMEPOINTS ONLY FOR SURREY #########################################################################
  # As the metadata was recorded for only Day 0, the future timepoints have incorrect metadata. If using metadata only,
  # this can be disabled to allow all timepoints.
SID_dict = {}
filtered_SIDs = []
if day_zero:
    SIDs = merged_surrey.index # Get SIDs
    for sid in SIDs:
        # Split SID into components
        patient = sid.split("_")[0]
        date = sid.split("_")[1]
        # Convert to date
        date_converted = datetime.strptime(date, "%d%m%y")
        if patient not in SID_dict: # Add to dict if new SID
            SID_dict[patient] = date
        else: # Update dict if from an earlier timepoint
            if date_converted < datetime.strptime(SID_dict[patient], "%d%m%y"):
                SID_dict[patient] = date
    # Recombine the SIDs
    for k, v in SID_dict.items():
        filtered_SIDs.append(f"{k}_{v}")
    # Filter to Day 0 samples only
    merged_surrey = merged_surrey[merged_surrey.index.isin(filtered_SIDs)]

### TRAIN/TEST SPLIT - KEEP PATIENT DATA TOGETHER ######################################################################
#Train test split for Surrey only; ISARIC remains as one dataset for validation
# Extract patient IDs from the index
patient_groups = merged_surrey.index.str[:3].tolist()
# Create a grouped split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get train and test indices while keeping patients together
train_idx, test_idx = next(gss.split(merged_surrey, groups=patient_groups))

# Create your train and test sets
train = merged_surrey.iloc[train_idx]
test = merged_surrey.iloc[test_idx]

# Save to csv
train.to_csv(f"{training_data}/Surrey_train.csv")
test.to_csv(f"{training_data}/Surrey_test.csv")

# Close all figures
plt.close('all')

# TODO: Expand data exploration further (e.g. from MH model data_exploration.py)
# Can do head and tail of data to check for bizare values, eg 1000 years old
# And/or plot histograms of the data to check it looks normal (probably also statistically normal otherwise transform?)
# Remove certain obvious outliers if needed - must be actual methods for this
# In other model I plotted graphs for a lot of the variables. See what could be useful here

# TODO PCA at different stages of data cleaning/feature selection (before/after filtering, after FS - but need numerical values

# TODO Skimmed over data cleaning a lot. Transforming? Outlier removal? Which steps remain