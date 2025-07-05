### SCRIPT USAGE #######################################################################################################
# Run this script twice with the required bools (See note on line ~28) to investigate and clean 1. The Surrey data used
# for training the model, and 2. the ISARIC data used for external validation.

######### SETUP ########################################################################################################
# Imports
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import GroupShuffleSplit

# Set pandas to display all columns and longer rows # IMPROVE remove in final version
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

# Bools to determine what to print
sample_inves_1 = False # Metadata info
sample_inves_2 = False # Metadata vs quant info
sample_inves_3 = False # Overlap of metadata vs quant
sample_inves_4 = False # Combining data frames/removing columns
sample_inves_5 = False # Checking for discrepancies in the metadata
sample_inves_6 = False # Data cleaning results within columns
sample_inves_7 = False # Missingness
sample_inves_8 = False # Numerical conversion

# WARNING: Important note: As this script has evolved to handle both the training/test data (Surrey) and the external
# validation data (ISARIC), there are two options:
#  - the 'validate' bool determines whether you are running it for Surrey (False) or ISARIC (True). The script will then
#    process that dataset accordingly. This means that the script will have to be run twice if validating; once for each dataset.
#  - the 'validation_compatible' bool is intended for the Surrey dataset when you are planning to calidate with ISARIC;
#    i.e. it enables some options that are required for the model to be compatible with both datasets
# IMPROVE: Ideally the script would be split into Surrey vs ISARIC scripts (in isaric can remove extra columns)
#  from the Surrey dataset where required).

# Bool to determine whether to make the Surrey (training) data compatible with external validation data (ISARIC)
validation_compatible = True # Will drop some incompatible columns
# Bool to determine which dataset is being processed. False = Surrey, True = ISARIC.
validate = True # Switch on if processing the validation dataset

# Read in data
quant_file = Path(__file__).parent / "Surrey_Files" / "KR_Covid_DIA_Pt_gene_Serum30_Report_Protein Quant (Pivot).xls"
s_meta_file = Path(__file__).parent / "Surrey_Files" / "Surrey_Metadata_master_spreadsheet_130622_edit2.csv"
isaric_file = Path(__file__).parent / "ISARIC_Files" / "ISARIC.csv"
phosp_file = Path(__file__).parent / "ISARIC_Files" / "PHOSP Metadata Master.xlsx" # Contains some ISARIC data
quant = pd.read_csv(quant_file, sep='\t').T # Transpose so sample IDs are rows
s_meta = pd.read_csv(s_meta_file)
isaric = pd.read_csv(isaric_file, index_col=0) # Avoid creating Unnamed: 0 column
phosp = pd.read_excel(phosp_file)

# Quant data preprocessing
quant.columns = quant.iloc[0] # Set protein names (now row 0) as column headers
quant = quant[1:] # Remove the first (now duplicated) row

### INVESTIGATE SAMPLE IDS - what samples do we have in each file and is this consistent? ##############################
if sample_inves_1:
    print("\nCheck the number of Surrey samples in the metadata:")
    print(len(s_meta))
    if validate:
        print("\nCheck the number of ISARIC samples in the metadata:")
        print(len(isaric))

if not validate:
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
else:
    # Removing duplicates in the ISARIC metadata file
        # CCPI307B is listed twice. The data appears to be the same patient from two different time points, however as it is not possible to
        # identify which is the day 1/admission timepoint (or discharge in the equivalent phosp data which is the earliest available) both samples will
        # be dropped.
    to_remove = isaric['pseudo_id'] == 'CCPI307B;_RHM01-0062;_SM2004042369'
    isaric = isaric[~to_remove]

# Extract SID from isaric file and set as index (first column contains multiple CCP IDs associated with multiple time points; we want the first only)
if validate:
    isaric['SID'] = (isaric['pseudo_id'].str.split(';').str[0].str.strip())
    isaric = isaric.drop(columns='pseudo_id') # Remove the old ID column
    isaric = isaric.set_index('SID', drop=True)

# Now to investigate the metadata vs. quant file
if sample_inves_2:
    # Check the number of samples in quant
    print("Number of unique sample timepoints in quant:")
    print(len(quant))

# Rename quant indexes to a simpler sample ID
if validate: # For ISARIC - strip to CCPXXXXX ID only
    before_samples = quant.index
    samples = before_samples.str.split('_')
    id_cluttered = samples.str[3] # Take first three characters because some have -[timepoint] added
    samples = id_cluttered.str.replace('.raw.PG.Quantity', '', regex=False)
    quant.index=samples # Now replace in quant df with a simplified version
else: # For Surrey
    before_samples = quant.index
    samples = before_samples.str.split('_')
    start_id = samples.str[3].str.slice(0, 3)  # Take first three characters because some have -[timepoint] added
    end_id = samples.str[4].fillna('')  # Some samples won't have this, in which case it creates nan
    samples = start_id + "_" + end_id
    samples = samples.str.replace('.raw.PG.Quantity', '', regex=False)
    quant.index = samples  # Now replace in quant df with a simplified version

# Investigate the IDs to find which samples are relevant
quant_ids = quant.index.to_series()
if sample_inves_2:
    quant_ids.to_csv("Quant_IDs.csv", index=False, header=False) # For easier examination
    print("Length before processing:", len(quant_ids))
    # Result: Using the Meta_plates.csv file, we can remove samples that belong to other datasets. As these SIDs are in
    # a different format, for simplicity the trends in SIDs are simply reported here, although ideally this would be
    # validated more extensively and automated. # IMPROVE

# Extract Surrey IDs from the dataset
if not validate:
        # Rule 1: SIDs starting with 'PHS' are from the PHOSP data set, not Surrey, and can be removed
    quant_ids_1 = quant_ids[~quant_ids.str.startswith('PHS_')]
    if sample_inves_2:
        print("Length after PHS removal:", len(quant_ids)) # 706 PHS samples removed
        # Rule 2: SIDs starting with 'QC' are quality control, not Surrey, and can be removed
    quant_ids_2 = quant_ids_1[~quant_ids_1.str.startswith('QC')]
    if sample_inves_2:
        print("Length after QC removal:", len(quant_ids))  # 44 QC samples removed
        # Rule 3: SIDs starting with 'CCP' are from ISARIC, not Surrey, and can be removed if not validation
    quant_ids_3 = quant_ids_2[~quant_ids_2.str.startswith('CCP')]
    if sample_inves_2:
        print("Length after CCP removal:", len(quant_ids))  # 66 ISARIC samples removed
        # Rule 4: SIDs starting with 'HB' are Surrey samples that are not to be included according to Meta_Plates.csv
    quant_ids = quant_ids_3[~quant_ids_3.str.startswith('HB')]
    if sample_inves_2:
        print("Final length after HB removal:", len(quant_ids))  # 13 HB samples removed
        # Result: 204 usable Surrey samples in the quant data.
else:
    quant_ids = quant_ids[quant_ids.str.startswith('CCP')]
    if sample_inves_2:
        print("Length of ISARIC samples:", len(quant_ids))  # 66 ISARIC samples left

if not validate:
    # Filter to just the samples from the Surrey dataset:
    quant_surrey = quant.loc[quant_ids]
    quant_samples = quant_surrey
else:
    quant_isaric = quant.loc[quant_ids]
    quant_samples = quant_isaric

# Check that the SIDs are as expected
if sample_inves_2:
    # Get the samples before and after processing, filtered to the Surrey samples only
    before_samples = before_samples[samples.isin(quant_samples.index)]
    samples = samples[samples.isin(quant_samples.index)]

    # Compare SIDs before and after
    comparison_quant = pd.DataFrame({
        "SID before": before_samples,
        "Length before": before_samples.astype(str).str.len(),
        "SID after": samples,
        "Length after": samples.astype(str).str.len(),
    })
    print("Surrey samples before and after processing:\n", comparison_quant.head())

    # Examine the changes made
    print("Value counts before processing:", comparison_quant["Length before"].value_counts())
    print("Value counts after processing:", comparison_quant["Length after"].value_counts())
    # Result for Surrey: Of these samples, 1 is of abnormal length
    # Set expected length for each dataset
    if not validate:
        expected_length = 10
    else:
        expected_length = 8
    print("Abnormal SIDs in quant data:")
    print(comparison_quant[comparison_quant["Length after"] != expected_length])
    abnormal_quant_SIDs = quant_samples[quant_samples.index.astype(str).str.len() != expected_length].iloc[:, 0:5]
    print("\nThe abnormal rows in the quant data file:\n", abnormal_quant_SIDs)
    # Result: For Surrey, in this case there is only one abnormal samples, and the 'SID before' shows us that this is due to a
    # hyphenation error in the original data. Therefore, we can simply correct the faulty column.
    # For ISARIC. 2 samples are of length 9, but 'CCPI1055C' is found in the dataset as expected so is not abnormal.
    #  CCPI1052C is not found in the dataset, but will be dropped when filtering for initial timepoints.

# Fix the hyphenation error in one Surrey sample
if not validate:
    quant_samples = quant_samples.rename(index={'293_': '293_240620'})

# Check the data now looks correct:
if sample_inves_2 and not validate:
    abnormal_quant_SIDs = quant_samples[quant_samples.index.astype(str).str.len() != expected_length].iloc[:, 0:5]
    print("Remaining abnormalities:", len(abnormal_quant_SIDs))

# Add a column to the metadata sheet with modified sample IDs (some inconsistencies between quant and Surrey metadata so simplifying in each by removing -[timepoint])
if not validate: # Only necessary for Surrey data
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
if not validate:
    quant_samples_set = set(quant_samples.index) # Warning: modified this to append _set so quant_samples remains a df; check the rest of the script works as intended for Surrey
    meta_samples_modified = set(s_meta['Sample Modified']) # Need to use the modified samples IDs to match quant
    sample_overlap_quant = quant_samples_set & meta_samples_modified
    quant_unique = quant_samples_set - meta_samples_modified
    s_meta_unique_quant = meta_samples_modified - quant_samples_set
else:
    quant_samples_set = set(quant_samples.index)
    meta_samples_modified = set(isaric.index)
    sample_overlap_quant = quant_samples_set & meta_samples_modified
    quant_unique = quant_samples_set - meta_samples_modified
    s_meta_unique_quant = meta_samples_modified - quant_samples_set

if sample_inves_3:
    print("Number of overlapping samples with quant:", len(sample_overlap_quant))
    print("Number only in quant samples:", len(quant_unique))
    print("Number only in s_meta_modified samples:", len(s_meta_unique_quant))
    # Result: 152 unique samples in the metadata for Surrey
    # All samples overlap in ISARIC

    # print("\nWhat are the actual sample values?")
    # print("Overlapping samples with quant:\n", sample_overlap_quant)
    # print("\n\nQuant unique samples:\n", quant_unique)
    # print("\n\nMetadata modified unique samples:\n", s_meta_unique_quant)

# Notes on Surrey data: Mabra ID 247 time point 260520 was found in the Surrey metadata and meta plates spreadsheet but
# not in quant. This can be disregarded as missing MS data. (so 151 remaining)

# Investigation of Meta_Plates.csv also shows there are three samples that are in the metadata file that were not processed:
# 369_130121, 370_130121, 373_150121

# This leaves 148 unaccounted for, although invesitgation of the original metadata spreadsheet appears to show that these
# samples are missing a lot of data, so perhaps were taken out of the study. As we need quant data to proceed with analysis,
# we shall therefore use the 204 samples that we have MS data for.

# # Comparioson df # IMPROVE delete later but useful to check IDs
# test = pd.DataFrame(
#    0,
#    index=range(370),
#    columns=['s_meta_samples', 's_meta_mod', 'quant']
# )
# test['s_meta_samples'] = s_meta['Sample']
# test['s_meta_mod'] = s_meta['Sample Modified']
# #test['quant'] = quant_surrey
# test.to_csv("test.csv")

### COMBINE ISARIC AND PHOSP METADATA ##################################################################################
#  Some samples are covered in each, and the PHOSP data takes samples at admission point which is more comparable to
#  Surrey data and therefore preferred
if validate:
    phosp = phosp[phosp['redcap_event_name'] == 'Hospital Discharge'] # Filter to 'Hospital Discharge' time points only
    isaric = isaric.drop(columns='age') # Drop to avoid duplicate columns names when merging
    isaric = isaric.reset_index().rename(columns={'index': 'SID'}) # Move index to column so it can be preserved and set later
    isaric_all = pd.merge(isaric, phosp, on='phosp_id', how='left') # Merge PHOSP and ISARIC
    isaric_all = isaric_all.set_index('SID')

### COMBINE QUANT AND META DATA ########################################################################################
if not validate:
    s_meta_mod = s_meta.set_index('Sample Modified') # Define new index with modified sample names to match quant
    merged = s_meta_mod.join(quant_samples, how='inner')
    merged.to_csv("Surrey_data_combined_all.csv") # Note this has the unmodified column names (e.g. whitespace)
else:
    merged = isaric_all.join(quant_samples, how='inner')
    merged.to_csv("ISARIC_data_combined_all.csv")

### CLEAN UP COLUMNS ###################################################################################################
merged.columns = merged.columns.str.strip() # Remove whitespace surrounding columns
if not validate: # Some Surrey specific investigation and cleaning
    # Adjust some faulty column names for Surrey (have an unknown symbol):
    merged.columns = merged.columns.str.replace(r"(Plasma - Ig[AGM] Anti-RBD Concentration \(ng/).*", r"\1l)", regex=True)

    # Change Sample column to be the updated SIDs currently stored as row indexes (so can reset index later)
    # merged['Sample'] = merged.index # Note I later remove this column but left this in the code in case it's useful later # IMPROVE - remove if not needed

    if sample_inves_4:
        print("Is the data frame the length we expect (# of overlapping samples)?")
        print(len(merged)) # Answer: Yes

        # What columns might we want to remove from the metadata?
        print("\nColumns in the Surrey metadata:")
        print(merged.columns[0:71])
        # TODO: List of columns I'm not sure on the meaning of, can I check? Might need to remove some if irrelevant
        # 'Chol', 'Airway Disease', 'For escalation? (Y/N)', 'PBMC No Calculation', 'Saliva - untargeted metabolomics' (and 2 similar).

# List unecessary columns
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
               #'Height (cm)', # TODO on second thought kept this in, but not sure
               'Hospital site', # TODO Possibly could impact care but I think a confounding feature? Maybe try with and without
               'MABRA ID', # Irrelevant
               'MABRA ID.1', # Irrelevant
               'Medication taken in 48 hours prior to sample', # TODO Possibly could have useful info if mediction is shown to be related - investigate
               'Nature of pre admission symptoms', # TODO might be useful for certain symptoms but for now removing for simplicity - but could do e.g. Chest pain Y/N from the data. This is likely shared with ISARIC.
               'Other', # TODO certainly useful but for now removing for simplicity - but could extract related illnesses e.g. high risk for covid
               'Plasma 1', # Irrelevant
               'Plasma 2', # Irrelevant
               'Plasma 3', # Irrelevant
               'Pt MABRA ID', # Irrelevant
               'Regular medications', # TODO also useful but skipping now for simplicity
               'Saliva - untargeted metabolomics', # TODO not sure but think irrelevant. Has high missingness so filtered regardless.
               'Saliva 1', # Irrelevant
               'Saliva 2', # Irrelevant
               'Saliva 3', # Irrelevant
               'Sebum - untargeted lipidomics', # TODO not sure but think irrelevant. Has high missingness so filtered regardless.
               'Serum - targeted metaboloimcs', # TODO not sure but think irrelevant. Has high missingness so filtered regardless.
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
# Remove Surrey columns that are incompatible with the ISARIC validation set (i.e. not found in both)
if validate:
    remove_isaric = ['Chol', # Unsure on this columns meaning, but I can't see anything that might correspond in ISARIC
                     'Airway Disease', # ISARIC as info on COPD and asthma but not OSA or pulmonary fibrosis
                     'BMI', # Exists in PHOSP but is NaN for all
                     'Clinical Covid (Y/N)', # Couldn't find relevant column in ISARIC data
                     'For escalation? (Y/N)', # Couldn't find relevant column in ISARIC data
                     'PBMC No Calculation', # Couldn't find relevant column in ISARIC data
                     'PBMC No', # Couldn't find relevant column in ISARIC data
                     'Survived Admission', # Couldn't find relevant column in ISARIC data
                     'Days_between', # Irrelevant as using D0 samples only
                     'MADU admission', # Couldn't find relevant column in ISARIC data
                     'Survived Admission', # Couldn't find relevant column in ISARIC data
                     'Weight (kg)' # Exists in PHOSP but is NaN for all
                     'Days_between' # Not relevant as only doing the first time point (day 1/admission where specific)
                     ]

# And drop them from the Surrey dataset
if not validate:
    merged = merged.drop(columns=remove_cols)
    if validation_compatible:
        merged = merged.drop(columns=remove_isaric)

# Rename Columns in the ISARIC dataset to match Surrey (note: only done for those that pass the filtering stage later in the script)
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
               'smoking_mhyn' : 'Smoking Status', # Assumed that the Non/Ex/Current smoker levels correspond to 0/1/2 in Surrey
               'diabetes_type_mhyn' : 'T2DM',
               'crf1a_o2_supp' : 'O2 req.',
               }
               # Note: Weight is in the PHOSP metadataset, but is NaN for all values. BMI therefore also can't be calculated.

### Do the required ISARIC column handling
if validate:
    # Rename columns for ISARIC
    merged.rename(columns=isaric_cols, inplace=True)
    # Drop extraneous columns in ISARIC
    cols_to_keep = list(isaric_cols.values()) + quant.columns.to_list()
    cols_to_keep_2 = [c for c in cols_to_keep if pd.notna(c) and (c in merged.columns)] # Also remove NaN simultaneously (causes problems with duplicate nan values)
    merged = merged[cols_to_keep_2]

# Calculate the number of columns of Surrey metadata remaining
if not validate:
    meta_cols = 71 - len(remove_cols) # 71 columns to start with in this dataset
else:
    meta_cols = len(isaric_cols)

# Examine remaining columns
if sample_inves_4:
    if not validate:
        print(f"\nRemoving {len(remove_cols)} columns from the dataset. {meta_cols} columns remaining.\n") # Reduced from 71 to 32 columns (currently)
    print("Metadata columns remaining after removal:")
    print(meta_cols)

    # Note: Some Surrey columns are not processed into a machine readable format or could benefit from further processing. However,
    #  as many are later filtered due to missingness they are left as is.
    #TODO: Other observations on columns (and actions needed)
    #  For columns like T2DM, is that correlated to covid severity/oxygen need? Otherwise could be biasing data towards majority
    #  Time point information could be deduced from the features, however I need to decide if I'm only using D0 first as that's when the metadata is recorded for

### CLEAN UP REMAINING COLUMNS ########################################################################################

# Check the column values are consistent/correct; i.e. which need to be cleaned (done in the next section)
# Assign which columns to check
columns_to_check = [feature for feature in merged.columns[0:meta_cols] if feature not in remove_cols]
check_only_one = False # Change this as needed
if check_only_one:
    columns_to_check = ['PBMC No'] # Edit this column name as needed if checking single columns

if sample_inves_5:
    for column in columns_to_check: # To check only one at a time for easier interpretation, change check_only_one to 'True' and edit the variable to the col title as needed.
        print("\nValue counts:\n", merged[column].value_counts())
        print("Uniques:\n", merged[column].unique())

    # TODO Some possible data handling that can be done but does not currently affect the result (Surrey):
    #  the Ig columns I won't clean because they're filtered out for missingness. But should clean if they're needed.
    #  CXR comments will be filtered as missing data so not cleaning is not a problem - but potentially useful information could be extracted (e.g. typical vs not typical COVID).
    #     However, 'no comment' vs missing data is hard to distinguish which could lead to bias

if not validate:
    ### Clean up values for Surrey metadata
    # Airway Disease
    merged.replace({'Airway Disease' : ['N']}, '0', inplace = True)
    merged.replace({'Airway Disease' : ['Asthma']}, '1', inplace = True)
    merged.replace({'Airway Disease' : ['COPD']}, '2', inplace = True)
    merged.replace({'Airway Disease' : ['OSA']}, '3', inplace = True)
    merged.replace({'Airway Disease' : ['Pulmonary fibrosis']}, '4', inplace = True)
    # CPAP
    merged.replace({'CPAP' : ['N ', ' N']}, 'N', inplace = True)
    merged.replace({'CPAP' : ['Y  ']}, 'Y', inplace = True)
    # CRP
    merged.replace({'CRP' : ['<4.0', '<4']}, '4', inplace = True) # IMPROVE To keep this numerical I changed the value as listed. But I am not confident this is the 'correct' approach.
    # Covid Positive Hospital Swab (Y/N)
    merged.replace({'Covid Positive Hospital Swab (Y/N)' : ['Not done ']}, np.nan, inplace = True)
    merged.replace({'Covid Positive Hospital Swab (Y/N)' : ['N - previously pos in ICU']}, 'Inconclusive', inplace = True) # Combined this data point with the other 'Inconclusive' as miceforest recommends grouping very rare categories
    # Duration of Pre-Admission Symptoms
    merged.replace({'Duration of Pre-Admission Symptoms' : ['23/05/2020']}, np.nan, inplace = True) # IMPROVE could handle more robustly for future samples - in this case the patient wasn't admitted so should be NaN.
    # For escalation? (Y/N)
    merged.replace({'For escalation? (Y/N)' : ['Yes']}, 'Y', inplace = True)
    merged.replace({'For escalation? (Y/N)' : ['No']}, 'N', inplace = True)
    # Gender
    merged.replace({'Gender' : ['Male', 'Male ']}, 'M', inplace = True)
    merged.replace({'Gender' : ['Female']}, 'F', inplace = True)
    # HTN
    merged.replace({'HTN' : ['y']}, 'Y', inplace = True)
    # ICU admission
    merged.replace({'ICU admission' : ['Yes']}, 'Y', inplace = True)
    merged.replace({'ICU admission' : ['No']}, 'N', inplace = True)
    # IHD
    merged.replace({'IHD' : ['Atrial fibrillation, heart failure', 'Coronary artery disease, heart failure']}, 'Y', inplace = True)
    # MADU admission
    merged.replace({'MADU admission' : ['Yes', 'yes']}, 'Y', inplace = True)
    merged.replace({'MADU admission' : ['No']}, 'N', inplace = True)
    # PBMC No
    merged.replace({'PBMC No' : ['Too many to count']}, 600, inplace = True) # IMPROVE this is estimated based on the highest value being 590, so as not to lose the fact that it's high vs putting NaN. But really this should be determined using the instrument specs.
    # PBMC No Calculation
    merged.replace({'PBMC No Calculation' : [' N/A ']}, np.nan, inplace = True) #TODO I think this is a valid approach; N/A is used when PMBC is either empty or too high so it does combine those two which might not be ideal, but I want to keep ordinality
    merged['PBMC No Calculation'] = pd.to_numeric(merged['PBMC No Calculation'].str.replace(',', ''), errors='coerce') # Convert strings to numeric, else NaN - IMPROVE beware if doing for new data to not accidentally convert any text to NaN
    # Survived Admission
    merged.replace({'Survived Admission' : ['Y ']}, 'Y', inplace = True)
    merged.replace({'Survived Admission' : ['N ']}, 'N', inplace = True)

else: # For ISARIC, change values to match Surrey
    # Bilateral CXR changes
    merged.replace({'Bilateral CXR changes': ['NO']}, 'N', inplace=True)
    merged.replace({'Bilateral CXR changes': ['YES']}, 'Y', inplace=True)
    # CPAP
    merged.replace({'CPAP': ['No']}, 'N', inplace=True)
    merged.replace({'CPAP': ['Yes']}, 'Y', inplace=True)
    merged.replace({'CPAP': ['N/K']}, np.nan, inplace=True)
    # Covid Positive Hospital Swab (Y/N)
    merged.replace({'Covid Positive Hospital Swab (Y/N)': ['Positive']}, 'Y', inplace=True)
    # Gender
    merged.replace({'Gender': ['Male', 'Male ']}, 'M', inplace=True)
    merged.replace({'Gender': ['Female']}, 'F', inplace=True)
    # HTN
    merged.replace({'HTN': ['NO']}, 'N', inplace=True)
    merged.replace({'HTN': ['YES']}, 'Y', inplace=True)
    # ICU admission
    merged.replace({'ICU admission': ['No']}, 'N', inplace=True)
    merged.replace({'ICU admission': ['Yes']}, 'Y', inplace=True)
    # IHD
    merged.replace({'IHD': ['No']}, 'N', inplace=True)
    merged.replace({'IHD': ['Yes']}, 'Y', inplace=True)
    # Smoking Status
    merged.replace({'Smoking Status': ['Never Smoked']}, '0', inplace=True)
    merged.replace({'Smoking Status': ['Former Smoker']}, '1', inplace=True)
    merged.replace({'Smoking Status': ['Yes']}, '2', inplace=True)
    merged.replace({'Smoking Status': ['N/K']}, np.nan, inplace=True)
    # T2DM
    merged.replace({'T2DM': ['NO']}, 'N', inplace=True)
    merged.replace({'T2DM': ['2']}, 'Y', inplace=True)
    merged.replace({'T2DM': ['1']}, 'N', inplace=True)
    # O2 req.
    merged.replace({'O2 req.': ['Yes']}, 'Y', inplace=True)
    merged.replace({'O2 req.': ['No']}, 'N', inplace=True)
    merged.replace({'O2 req.': ['N/K']}, np.nan, inplace=True)

# Check the data is corrected
if sample_inves_6:
    for column in columns_to_check:
        print("\nValue counts fixed:\n", merged[column].value_counts())
        print("Uniques fixed:\n", merged[column].unique())

# Check the data types
# if sample_inves_6:
#     for column in columns_to_check:
#         print(f"'{column}' data type:")
#         print(merged[column].dtype)

# Remove columns with no data - the Surrey dataset also contains some duplicated NaN headers with no data which are
#  simultaneously removed
if sample_inves_6:
    print("Number of columns before removing empty columns:")
    print(len(merged.columns))
    dupe_cols = merged.columns[merged.columns.isnull() | (merged.columns == "")]
    print("Number of duplicated columns before removing empty columns:")
    print(len(dupe_cols))
    if len(dupe_cols) > 0:
        print("Duplicate cols:", dupe_cols.tolist())

# Remove empty columns (all NaN values except header)
merged.dropna(how='all', axis=1, inplace=True)

if sample_inves_6:
    print("Number of columns after removing empty columns:")
    print(len(merged.columns))
    dupe_cols = merged.columns[merged.columns.isnull() | (merged.columns == "")]
    print("Number of duplicated columns after removing empty columns:")
    print(len(dupe_cols))
    if len(dupe_cols) > 0:
        print("Duplicate cols:", dupe_cols.tolist())

# Save to csv # IMPROVE could remove this if it's not needed/later csvs are more useful
if not validate:
    merged.to_csv("Surrey_data_selected.csv")
else:
    merged.to_csv("ISARIC_data_selected.csv")

### CLEAN UP SAMPLE ROW MISSINGNESS ####################################################################################
# Remove any rows that are empty for O2 req, the target
merged = merged[merged['O2 req.'].notna()] # Removes 20 values from Surrey data, 2 for ISARIC
# Summarise missing values before removing rows
merged_null_before = merged.isnull().sum().to_frame(name='Missing_Count_Before')
merged_null_before['Missing_Percentage_Before'] = (merged_null_before['Missing_Count_Before'] / len(merged)) * 100

### Examine missingness across rows
# NOTE: Previously I filtered out highly missing rows. I no longer do this as the quant data can still be useful, but the plot is left here for investigative purposes.
# Calculate NA counts per row for selected metadata columns
row_na_counts = merged.iloc[:, :meta_cols].isna().sum(axis=1)
missing_distribution = (
    row_na_counts.value_counts()
    .sort_index()
    .reset_index(name='Rows')
    .rename(columns={'index': 'NA_Count'})
)
# Calculate percentages
missing_distribution['Percentage'] = (missing_distribution['Rows'] / len(merged) * 100).round(2)
if sample_inves_7:
    print("Missing values distribution in metadata:")
    print(missing_distribution.to_string(index=False))

### Plot missingness
plt.figure(figsize=(10, 6))
sns.barplot(x='NA_Count', y='Rows', data=missing_distribution, palette='Blues_d', edgecolor='black', hue='NA_Count',
            legend=False)
plt.title('Missing Values in the Metadata')
plt.xlabel('Number of Missing values per samples')
plt.ylabel('Number of samples')
# Add percentage labels
for index, row in missing_distribution.iterrows():
    plt.text(row.name, row.Rows + 1,  # Offset above bar
             f'{row.Percentage}%', ha='center')
plt.tight_layout()
plt.savefig('missing_distribution.png', dpi=300)

# IMPROVE plot for quant data as well?

### CLEAN UP COLUMN MISSINGNESS ########################################################################################
# Plot missingness before filtering (full dataset) #TODO some plots are blank - possibly due to too much data
fig = msno.matrix(merged)
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-data_before_filtering.png', bbox_inches = 'tight')
# Plot missingness of the metadata
fig = msno.matrix(merged.iloc[:,0:meta_cols])
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-metadata_before_filtering.png', bbox_inches = 'tight')
# Plot missingness of the quant data
fig = msno.matrix(merged.iloc[:,meta_cols:])
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-quantdata_before_filtering.png', bbox_inches = 'tight')

# Note: columns with 100% missingness have already been removed prior to this

# Investigate null values in each column
if sample_inves_7:
    print("Missing count per column:")
    print(merged.isnull().sum(), "\n")
    print("Column count before filtering for missing data:")
    print(len(merged.columns))

# Extract missingness of below 30% and keep those columns
merged_low_missing = merged_null_before[merged_null_before['Missing_Percentage_Before'] < 30].index.tolist()
merged = merged[merged_low_missing]
merged.to_csv("Surrey_data_low_missing.csv")

if sample_inves_7:
    print("\nColumn count after filtering for missing data:")
    print(len(merged.columns))

# Summarise missing values in columns after filtering
merged_null_after = merged.isnull().sum().to_frame(name='Missing_Count_After')
merged_null_after['Missing_Percentage_After'] = (merged_null_after['Missing_Count_After'] / len(merged)) * 100

# Combine the information on missingness data before and after column filtering
merged_null_summary = pd.concat([merged_null_before, merged_null_after], axis=1, sort=False).fillna('[Column removed]')

# Save missingness info before and after filtering to csv
merged_null_summary.to_csv("Missing_values_comparison.csv",
                           header=["NA Count Before","Missingness (%) Before", "NA Count After","Missingness (%) After"],
                           float_format="%.1f")

### Determine how many metadata columns remain
# List starting and filtered columns
if not validate:
    meta_columns = s_meta.columns.tolist()
else:
    meta_columns = isaric_cols.values()
existing_columns = merged.columns.tolist()
# Work backwards through the list to find the first one still present
meta_cols = 0 # Initialise
for col in reversed(meta_columns):
    if col in existing_columns:
        if sample_inves_7:
            print("\nRemaining metadata columns:")
            print(merged.columns.get_loc(col) + 1) # +1 for 1-based indexing conversion / allows for splicing where the first number is inclusive and the second exclusive
        meta_cols = merged.columns.get_loc(col) + 1
        break
else:
    if sample_inves_7:
        print("All metadata columns were removed by the missingness filter.")
    meta_cols = 0

# Plot missingness after filtering (full dataset)
fig = msno.matrix(merged)
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-data_after_filtering.png', bbox_inches = 'tight')
# Plot missingness of the metadata
fig = msno.matrix(merged.iloc[:,0:meta_cols])
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-metadata_after_filtering.png', bbox_inches = 'tight')
# Plot missingness of the quant data
fig = msno.matrix(merged.iloc[:,meta_cols:]) # Note: This is currently 546-27=519 MS data columns
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-quantdata_after_filtering.png', bbox_inches = 'tight')

### CONVERT TO NUMERICAL ###############################################################################################
# WARNING: Note that this is only done for the columns that made it through filtering. With a larger dataset (and different
#  missingness these results may be different, so recheck that everything is processed correctly.

# Initialise list of numerical vs categorical
numerical_cols = []
categorical_cols = []

# Categorise as numeric or categorical based on conversion
for col in merged.columns:
    # Check if column contains numbers (after dropping NaNs)
    non_na_values = merged[col].dropna()
    # Try to convert to numeric, if successful it's numerical
    if pd.api.types.is_numeric_dtype(non_na_values) or pd.to_numeric(non_na_values, errors='coerce').notna().any():
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)

if sample_inves_8:
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    print(f"\nNumerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

# Build a dict of pre‑conversion NaN counts
pre_missing = {col: merged[col].isna().sum() for col in numerical_cols}

# Convert to numerical or categorical
for col in numerical_cols:
    # Remove commas (thousands separators) if present
    cleaned = merged[col].astype(str).str.replace(',', '')
    # Convert to numeric, forcing non‐parseable entries to NaN
    merged[col] = pd.to_numeric(cleaned, errors='coerce')
for col in categorical_cols:
    merged[col] = merged[col].astype('category')

# Build a dict of post‐conversion NaN counts
post_missing = {col: merged[col].isna().sum() for col in numerical_cols}

# Compare and report any increases
cols_nanned = 0
for col in numerical_cols:
    before = pre_missing[col]
    after  = post_missing[col]
    if after > before: # Note this shows regardless of show_inves or not
        print(f"Column {col!r} gained {after-before} new NaN(s) (was {before}, now {after})")
        print("Investigate with .unique/.value_counts or similar to compare.")
        cols_nanned +=1

if sample_inves_8:
    if cols_nanned == 0:
        print("\nNone of the columns had any real values converted to NaN.")

### LOG2 TRANSFORM THE PROTEOMICS DATA #################################################################################
merged.iloc[:,meta_cols:] = np.log2(merged.iloc[:,meta_cols:] + 1e-6) # IMPROVE This log transformation is not noted in the data headers anywhere but is used in all future processing

# Save final dataset to csv
if not validate:
    merged.to_csv("Surrey_final.csv")
else:
    merged.to_csv("ISARIC_final.csv")

### PLOT CLASS DISTRIBUTION ############################################################################################
plt.figure(figsize=(10, 6))
sns.countplot(x='O2 req.', data=merged, palette='Blues_d', edgecolor='black', hue='O2 req.',
            legend=False)
if not validate:
    dataset = 'Surrey'
else:
    dataset = 'ISARIC'
plt.title(f'O2 requirement class distribution - {dataset} dataset')
plt.xlabel('O2 required')
plt.ylabel('Count')
plt.savefig(f'class_distribution_{dataset}.png', dpi=200)

### TRAIN.TEST SPLIT - KEEP PATIENT DATA TOGETHER ######################################################################
if not validate: # Keep ISARIC as one dataset; split Surrey
    # Extract patient IDs from the index
    patient_groups = merged.index.str[:3].tolist()
    # Create a grouped split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Get train and test indices while keeping patients together
    train_idx, test_idx = next(gss.split(merged, groups=patient_groups))

    # Create your train and test sets
    train = merged.iloc[train_idx]
    test = merged.iloc[test_idx]

    # Save to csv
    train.to_csv("Surrey_train.csv")
    test.to_csv("Surrey_test.csv")


# TODO: Expand data exploration further (e.g. from MH model data_exploration.py)
# Can do head and tail of data to check for bizare values, eg 1000 years old
# And/or plot histograms of the data to check it looks normal (probably also statistically normal otherwise transform?)
# Remove certain obvious outliers if needed - must be actual methods for this
# In other model I plotted graphs for a lot of the variables. See what could be useful here

# Improve: Could put any extra files in another subfolder so they're out the way for other analysis

# TODO PCA at different stages of data cleaning/feature selection (before/after filtering, after FS - but need numerical values

# TODO Skimmed over data cleaning a lot. Transforming? Outlier removal? Which steps remain
