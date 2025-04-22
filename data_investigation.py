# Imports
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import GroupShuffleSplit

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Bools to determine what to print
sample_inves_1 = False # Metadata vs plate info
sample_inves_2 = False # Metadata vs quant info
sample_inves_3 = False # Overlap of metadata vs quant #todo could change back to 2 when complete, just working on it now
sample_inves_4 = False # Combining data frames/removing columns
sample_inves_5 = False # Data cleaning within columns
sample_inves_6 = False # Missingness
sample_inves_7 = False # Numerical conversion

# Read in data
quant_file = Path(__file__).parent / "Surrey_Files" / "KR_Covid_DIA_Pt_gene_Serum30_Report_Protein Quant (Pivot).xls"
plates_file = Path(__file__).parent / "Surrey_Files" / "Meta_Plates.csv"
s_meta_file = Path(__file__).parent / "Surrey_Files" / "Surrey_Metadata_master_spreadsheet_130622_edit2.csv"
quant = pd.read_csv(quant_file, sep='\t').T # Transpose so sample IDs are rows
plates = pd.read_csv(plates_file)
s_meta = pd.read_csv(s_meta_file)

# Quant data preprocessing
quant.columns = quant.iloc[0] # Set protein names (now row 0) as column headers
quant = quant[1:] # Remove the first (now duplicated) row

### INVESTIGATE SAMPLE IDS - what samples do we have in each file and is this consistent? ##############################
if sample_inves_1:
    # First check that the plates and metadata samples are consistent
    print("Check that the plates Collection column contains all place names in their correct format - so we don't miss any Surrey samples:")
    print(plates['Collection'].unique())
    print("Should be only 'Surrey' for Surrey, and the the other sample origins. [Answer: Yes, no typos.]")

    print("\nNow check the number of Surrey samples in plates")
    print(len(plates[plates['Collection'] == 'Surrey']))
    # Without the extra plates bracket you get a True/False df; so we subset/filter plates again according to that rule to get
    #  the rows themselves, then do len()
    print("Now check the number of samples in Surrey Metadata - is it the same?")
    print(len(s_meta))
    print("[Answer: No, so how does it differ?]\n")

    print("Now check the sample IDs for surrey plates")

# Removing duplicates in the metadata file # TODO could check quant too. And maybe go over the dropbox metadata file for more potential issues. Including these reenlistements
#  Several participants were re-enlisted: 403>434, 404>411, and 405>410. This leads to duplicate sample IDs (434 is
    #  slightly different due to one ID missing the middle number, but should still be removed as a duplicate
    # Therefore we will remove the earlier MABRA IDs
# Define the conditions for rows to remove
conditions_to_remove = (
    ((s_meta['Sample'] == '434-28_220321') & (s_meta['MABRA ID'] == 403)) |
    ((s_meta['Sample'] == '411-0_010321') & (s_meta['MABRA ID'] == 404)) |
    ((s_meta['Sample'] == '410-0_010321') & (s_meta['MABRA ID'] == 405))
)
# Remove the rows by keeping only those that do not meet the conditions
s_meta = s_meta[~conditions_to_remove]
# TODO this does leave 3 empty spaces in the df, fix later if it matters

# Calculate uniques/overlaps
plates_samples = set(plates[plates['Collection'] == 'Surrey']['Sample'])
s_meta_samples = set(s_meta['Sample'])
sample_overlap = plates_samples & s_meta_samples
plates_unique = plates_samples - s_meta_samples
s_meta_unique = s_meta_samples - plates_samples

# Continue exploration
if sample_inves_1:
    print("Number of overlapping samples:", len(sample_overlap))
    print("Number only in plate samples:", len(plates_unique))
    print("Number only in s_meta samples:", len(s_meta_unique))

    # print("\nWhat are the actual sample values?")
    # print("Overlapping samples:\n", sample_overlap)
    # print("\n\nPlates unique samples:\n", plates_unique)
    # print("\n\nMetadata unique samples:\n", s_meta_unique)

    # TODO: There is a discrepancy with both plates and s_meta having unique sample IDs. For now I'm proceeding with the metadata
    #  sheet (the master one not dropbox one) but this should be investigated and understood. Some like '315_0'were not used in the
    #  dropbox file, so I suspect filtering out all those (red) ones would help. The plates i'm not sure on, can I find those samples
    #  in the data I have somewhere? Is it the ones with F under 'Include'?

    # Aside from the above discrepancy, the plates file contains no other useful info so can be discarded

# Now to investigate the metadata vs. quant file
if sample_inves_2:
    # Check the number of samples in quant
    print("Number of unique sample timepoints in quant:")
    print(len(quant))

# Rename quant indexes to a simpler sample ID
samples = quant.index
#print("293 sample unmodified:", samples[387]) #todo
samples = samples.str.split('_')
#print("293 sample split:", samples[387]) #todo
start_id = samples.str[3].str.slice(0, 3) # Take first three characters because some have -[timepoint] added
end_id = samples.str[4].fillna('') # Some samples won't have this, in which case it creates nan
samples = start_id + "_" + end_id
samples = samples.str.replace('.raw.PG.Quantity', '', regex=False)
#print("293 sample deleted:", samples[387]) #todo
quant.index=samples # Now replace in quant df with a simplified version
# TODO the above has prints because this one fails. It has a - instead of _. But I can't split on that always because
#  in other samples it has - AND _ which would throw it off. So need some regex that can handle:
#  123_123456, 123-1_123456, and 123-123456. Think that's all formats but could be more Surrey ones I haven't found yet

# Add a column to the metadata sheet with modified sample IDs (some inconsistencies between quant and metadata so simplifying in each by removing -[timepoint])
samples_mod = s_meta['Sample']
if sample_inves_2:
    print("Meta samples before underscore removal:", len(samples_mod))
    print("Meta samples containg hyphens:", len(samples_mod[samples_mod.str.contains("-")]))
samples_mod = samples_mod.str.replace(r'-\d+(?=_)', '', regex=True) # Remove '-'

if sample_inves_2:
    print("Meta samples after hymphen removal:", len(samples_mod))
    print("Meta samples containg hyphens after removal:", len(samples_mod[samples_mod.str.contains("-")]))

# Remove trailing _ from SIDs
#  Some IDs have a trailing _ in the sample ID which appears accidental.
samples_mod = samples_mod.str.rstrip('_')
# TODO 315, 316, 409, 458, 522 are all missing the date and hence have trailing _. These are part of the
#  samples not in the 204 found by my current extractiom which could be due to some Surrey samples existing that aren't
#  in the expected format (starting wtih digits), but for now as they're not included it should be fine to remove hyphens
#  from these too, as well as 260 which isn't matched due to the hyphen.
s_meta['Sample Modified'] = samples_mod

# Extract all surrey samples from the quant df
surrey = quant.index.to_series().str.match(r'^\d') # True/False df matcing those starting with digits TODO think this is all surrey but might be wrong given it's 204 only
quant_surrey = quant[surrey] # Actual values
#print("quant surrey length", len(quant_surrey)) # delete later TODO this is only 204 (not 361 like metadata), is that expected? Might need to check manually
# TODO It's gone between 203 and 204 -  I think just go over and manually remove PHSOP, ISARIC, etc and see what
#  I'm left with and how to convert that to the 123_123456 format and if there's duplicates doing that - I need to check
#  if all the Surrey samples are actually in my datasheet (compare to metadata)

# Calculate uniques/overlaps
quant_samples = set(quant_surrey.index)
s_meta_samples_modified = set(s_meta['Sample Modified']) # Need to use the modified samples IDs to match quant
sample_overlap_quant = quant_samples & s_meta_samples_modified
quant_unique = quant_samples - s_meta_samples_modified
s_meta_unique_quant = s_meta_samples_modified - quant_samples

if sample_inves_3:
    print("Number of overlapping samples with quant:", len(sample_overlap_quant))
    print("Number only in quant samples:", len(quant_unique))
    print("Number only in s_meta_modified samples:", len(s_meta_unique_quant))

    # print("\nWhat are the actual sample values?")
    # print("Overlapping samples with quant:\n", sample_overlap_quant)
    # print("\n\nQuant unique samples:\n", quant_unique)
    # print("\n\nMetadata modified unique samples:\n", s_meta_unique_quant)

    #TODO: Have 158 samples in metadata but not quant. Need to find them in quant, possibly the format is a bit different?
    #  or another sheet somewhere? But for now just moving ahead for time
    #TODO as also noted around line 98, there is a 293_ sample that is unique to quant (the only one!). This is due to
    #  not correctly splitting up the parts due to inconsistent formatting. If it really is just that one I would edit
    #  the spreadsheet, but probably best to figure out some kind of regex. See line 98. For now just moving on without
    #  that sample.

# # test df #todo delete later but useful to check IDs
# test = pd.DataFrame(
#    0,
#    index=range(370),
#    columns=['s_meta_samples', 's_meta_mod', 'quant']
# )
# test['s_meta_samples'] = s_meta['Sample']
# test['s_meta_mod'] = s_meta['Sample Modified']
# #test['quant'] = quant_surrey
# test.to_csv("test.csv")

### COMBINE QUANT AND META DATA ########################################################################################
s_meta_mod = s_meta.set_index('Sample Modified') # Define new index with modified sample names ot match quant
merged = s_meta_mod.join(quant_surrey, how='inner')
merged.to_csv("Surrey_data_combined_all.csv") # Note this has the unmodified column names (e.g. whitespace)

### CLEAN UP COLUMNS ###################################################################################################
merged.columns = merged.columns.str.strip() # Remove whitespace surrounding columns
# Adjust some faulty column names: # TODO this didn't work; use iloc but doing later
merged.rename(columns={"Plasma - IgA Anti-RBD Concentration (ng/�l)": "Plasma - IgA Anti-RBD Concentration (ng/l)",
                       "Plasma - IgG Anti-RBD Concentration (ng/�l)": "Plasma - IgG Anti-RBD Concentration (ng/l)",
                       "Plasma - IgM Anti-RBD Concentration (ng/�l)": "Plasma - IgM Anti-RBD Concentration (ng/l)"})

# Change Sample column to be the updated SIDs currently stored as row indexes (so can reset index later)
# merged['Sample'] = merged.index # Note I later remove this column but left in code in case it's useful later

if sample_inves_4:
    print("Is the data frame the length we expect (# of overlapping samples)?")
    print(len(merged)) # Answer: Yes

    # What columns might we want to remove from the metadata?
    print("\nColumns in the metadata:")
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
               'Date of MOST RECENT Covid Positive Swab', # Not relevant to health - # TODO but can you extract extra information based on e.g time since postive? with other data.
               'Date of first Mabra samples collected',
               'Date of vaccination', # TODO this column is useful for time since vaccination and Y/N vaccinated, however for now removing as 1) it's highly missing anyway and 2) Need to do more processing before inclusion. But come back to
               'Ethnicity', # Largely biased towards white so would be misleading
               #'Height (cm)', # TODO on second thought kept this in, but not sure
               'Hospital site', # TODO Possibly could impact care but I think a confounding feature? Maybe try with and without
               'MABRA ID', # Irrelevant
               'MABRA ID.1', # Irrelevant
               'Medication taken in 48 hours prior to sample', # TODO Possibly could have useful info if mediction is shown to be related - investigate
               'Nature of pre admission symptoms', # TODO might be useful for certain symptoms but for now removing for simplicity - but could do e.g. Chest pain Y/N from the data
               'Other', # TODO certainly useful but for now removing for simplicity - but could extract related illnesses e.g. high risk for covid
               'Plasma 1', # Irrelevant
               'Plasma 2', # Irrelevant
               'Plasma 3', # Irrelevant
               'Pt MABRA ID', # Irrelevant
               'Regular medications', # TODO also useful but skipping now for simplicity
               'Saliva - untargeted metabolomics', # TODO not sure but think irrelevant. Missing anyway.
               'Saliva 1', # Irrelevant
               'Saliva 2', # Irrelevant
               'Saliva 3', # Irrelevant
               'Sebum - untargeted lipidomics', # TODO not sure but think irrelevant. Missing anyway.
               'Serum - targeted metaboloimcs', # TODO not sure but think irrelevant. Missing anyway.
               'Serum 1', # Irrelevant
               'Serum 2', # Irrelevant
               'Serum 3', # Irrelevant
               'Serum 4', # Irrelevant
               'Serum 5', # Irrelevant
               'Serum 6', # Irrelevant
               'Serum 7', # Irrelevant
               'Serum 8', # Irrelevant
               'Serum 9', # Irrelevant
               'Time between pos covid test and first sample', # TODO think this is irrelevant
               #'Weight (kg)', #TODO keeping this in for now but now 100% sure
               'Dates of repeat samples', # TODO might be useful for getting timepoint data alongisde other cols, but for now removing
               ]

# And drop them from the dataset
merged = merged.drop(columns=remove_cols)
if sample_inves_4:
    print(merged.columns[0:32]) #Reduced from 71 to 32 columns (TODO: currently, may change)

    #TODO remove missing ROWS first to not obfuscate column missingness

    #TODO: Other observations on columns (and actions needed)
    #  CXR comments is plain text but could be interpreted, although high missingness so for now is fine to remove in missingness filter below
    #  Collection day column needs to be processed into the value for the sample itself or removed. Redundancy with other cols
    #  Covid Positive Hospital Swab (Y/N) needs 'Not done' values converted to missing - check uniques. Do for all columns really
    #  Duration of Pre-Admission Symptoms has one date in it so adjust that
    #  For escalation? (Y/N) Needs converting to Y N consistently. Same for Gender. Probably others. ICU.
    #  Hospital site I removed but there could be a difference, not sure if I should have kept it
    #  Ig columns have a symbol in col title, and multiple values/formats. Fix, but will be likely removed with high missingness anyway (Update one isnlt so is currently nonsense in my data set)
    #  Survived admission could be marked as oxygen needed - do covid tests etc confirm need for oxygen? If so check O2 req column and see if it correlates/is empty and adjust accordingly
    #  For columns like T2DM, is that correlated to covid severity/oxygen need? Otherwise could be biasing data towards majority
    #  Instead of dates of repeat samples I'm going to use the last 6 digits of SID. Really I should use the repeat dates but for time instead of cleaning/checking they're the same/right i'll just use SID.. actually for now can omit bc date isn't useful
    #    But also for dates, do as a timepoint post admission and not a date. See collection day/days between alongsite date columns and see if you can extract a defined day for eah sample
    #    Days between might actually be this already. But check it because a lot of zeroes. And would d1 vs d2 still be zero? So not an actual timepoint. Or is it from D0
    # Random thought - could plot how many who need oxygen need it on first admission vs over time (or stop needing later). If it's generally needed
    #    immediately then not much point predicting later need - but equally predicting the O2 needed upon admission. Is then
    #    equally valuable ish as future prediction, just less valuable as a project.

# Clean up the columns
# TODO: For now just doing basics like consistent Yes/No formats, but maybe needs more cleaning especially with some of the columns in the above todo.
#  Should also do standard data cleaning steps like normalisation etc, anything you can find
#  Also doesn't address quant data but that probably need some kind of cleaning too
#  Also encoding but will handle that later in the process

# Check values are consistent
if sample_inves_5:
    # Chol - Example
    print("\nValue counts:\n", merged['Chol'].value_counts())
    print("Uniques:\n", merged['Chol'].unique())
    # ... continue with all others - not showing here for brevity but to investigate change the above column header

    # TODO Some changes that need to be made:
    #  Airway Disease - On the existing 0-4 scale I guessed 1 Asthma 2 COPD 3 OSA 4 Pulmonary fibrosis. And 0 for none.
    #  the Ig columns are still poorly named and I won't clean them yet because I think they'll be filtered out for missingness. But should rename and clean
    #  Think BMI is fine but it never shows .0 just .
    #  CRP '<4' values were changed to 4 to keep them numerical
    #  CXR comments is ignored as it's complex and will be filtered as missing data. But should process into separate columns later - just hard to tell N/A vs missing

### Clean up values for metadata
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
merged.replace({'CRP' : ['<4.0', '<4']}, '4', inplace = True)
# Covid Positive Hospital Swab (Y/N)
merged.replace({'Covid Positive Hospital Swab (Y/N)' : ['Not done ']}, np.nan, inplace = True) #TODO when encoding might do ordinal; only two non Y/N samples
# Duration of Pre-Admission Symptoms
merged.replace({'Duration of Pre-Admission Symptoms' : ['23/05/2020']}, np.nan, inplace = True) #TODO could handle more robustly for future samples - in this case the patient wasn't admitted so should be NaN.
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
merged.replace({'PBMC No' : ['Too many to count']}, np.nan, inplace = True) #TODO loses the fact its too high, should i change to a high value instead?
# PBMC No Calculation
merged.replace({'PBMC No Calculation' : [' N/A ']}, np.nan, inplace = True) #TODO think this is valid, N/A is used when PMBC is either empty or too high so it does combine those two which might not be ideal, but I want to keep ordinality
merged['PBMC No Calculation'] = pd.to_numeric(merged['PBMC No Calculation'].str.replace(',', ''), errors='coerce') # Convert strings to numeric, else NaN - IMPROVE beware if doing for new data to not accidentally convert any text to NaN
# Survived Admission
merged.replace({'Survived Admission' : ['Y ']}, 'Y', inplace = True)
merged.replace({'Survived Admission' : ['N ']}, 'N', inplace = True)

# Check the data is corrected
if sample_inves_5:
    print("\nValue counts fixed:\n", merged['Chol'].value_counts())
    print("Uniques fixed:\n", merged['Chol'].unique())

# TODO should double check things are categorical/numerical as expected and not mixed - e.g. CRP I had to remove <4 but wasn't through on checking all values were now correct

# Save to csv # IMPROVE could remove this if it's not needed/later csvs are more useful
merged.to_csv("Surrey_data_selected.csv")

### CLEAN UP SAMPLE ROW MISSINGNESS ####################################################################################
# Remove any rows that are empty for O2 req, the predictor # TODO could do 'Will they need O2' as a column, in which case if patients have multiple timepoints where they need O2 in one of them, that data could still be useful. Look into this - for now just predicting current O2
merged = merged[merged['O2 req.'].notna()] # Removes 20 values from Surrey data (current dataset)

# Summarise missing values before removing rows
merged_null = merged.isnull().sum().to_frame(name='Missing_Count')
merged_null['Missing_Percentage'] = (merged_null['Missing_Count'] / len(merged)) * 100
merged_null.to_csv("Missing_values_before_row_removal.csv", header=["Missing_Count", "Missingness (%)"], float_format="%.1f") # Print to csv for easier analysis
# From the results we see many missing protein quantities, and plenty of missing metadata

# Calculate NA counts per row for first 32 columns # IMPROVE could change with new cols added
row_na_counts = merged.iloc[:, :32].isna().sum(axis=1)
missing_distribution = (
    row_na_counts.value_counts()
    .sort_index()
    .reset_index(name='Rows')
    .rename(columns={'index': 'NA_Count'})
)
# Calculate percentages
missing_distribution['Percentage'] = (missing_distribution['Rows'] / len(merged) * 100).round(2)
if sample_inves_6:
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

# Cutoff rows beyond a certain missingness
#TODO For now doing 8, not sure if I should filter beyond that - arbitrary currently
cutoff = row_na_counts < 8
merged = merged[cutoff]
merged.to_csv("Surrey_data_selected_filtered.csv") # Dataset with highly missing rows removed

# Summarise missing values in columns after removing rows
merged_null = merged.isnull().sum().to_frame(name='Missing_Count')
merged_null['Missing_Percentage'] = (merged_null['Missing_Count'] / len(merged)) * 100
merged_null.to_csv("Missing_values_after_row_removal.csv", header=["Missing_Count", "Missingness (%)"], float_format="%.1f")
# IMPROVE: Verdict, some higher some lower, might be worth doing might not? Or do a lower filter than 8

### CLEAN UP COLUMN MISSINGNESS ########################################################################################
# Note: not imputing, just filtering for high missingness
# Investigate null values #todo get back to column missingness but doing rows first
if sample_inves_6:
    print("Missing count per column:")
    print(merged.isnull().sum(), "\n")
    print("Column count before filtering for missing data:")
    print(len(merged.columns))

# Extract missingness of below 30% and keep those columns
merged_low_missing = merged_null[merged_null['Missing_Percentage'] < 30].index.tolist()
merged = merged[merged_low_missing]
merged.to_csv("Surrey_data_low_missing.csv")

if sample_inves_6:
    print("\nColumn count after filtering for missing data:")
    print(len(merged.columns))

# Summarise missing values in columns after filtering
merged_null = merged.isnull().sum().to_frame(name='Missing_Count')
merged_null['Missing_Percentage'] = (merged_null['Missing_Count'] / len(merged)) * 100
merged_null.to_csv("Missing_values_after_filtering.csv", header=["Missing_Count", "Missingness (%)"], float_format="%.1f")

# Plot missingness after filtering (before is unreadable)
fig = msno.matrix(merged)
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-data_after_filtering.png', bbox_inches = 'tight')
# Plot missingness of the metadata
fig = msno.matrix(merged.iloc[:,0:28]) # IMPROVE Now down to 28 columns of metadata
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-metadata_after_filtering.png', bbox_inches = 'tight')
# Plot missingness of the quant data
fig = msno.matrix(merged.iloc[:,28:])
fig_copy = fig.get_figure()
fig_copy.savefig('Missingness_All-quantdata_after_filtering.png', bbox_inches = 'tight')

### CONVERT TO NUMERICAL ###############################################################################################
# TODO Note that this is only done for the columns that made it through filtering. With a larger dataset (and different
#  missingness these results may be different, so recheck

#TODO dropping IgG for now - it requires more processing than I have time for
#print(merged.columns[0:28])
merged = merged.drop(merged.columns[22], axis=1)
#print(merged.columns[0:28])
merged.to_csv("Surrey_data_low_missing.csv") # Uses same name and overwrites so just delete this once you've handled IgG

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

if sample_inves_7:
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

if sample_inves_7:
    if cols_nanned == 0:
        print("\nNone of the columns had any real values converted to NaN.")

# Save final dataset to csv
merged.to_csv("Surrey_final.csv")

### TRAIN.TEST SPLIT - KEEP PATIENT DATA TOGETHER ######################################################################
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

#TODO I really should figure out IgG... which means extracting current timepoint (days between?). But doesn't exactly
# line up to IgG day. Does it always correspond to sample number though? eg 3 timepoints = 3 igg readings


# TODO: to take from other data_exploration. Skimming/skipping for now to just get a basic model made

#   print(df.describe(include='all'))

# Should still examine which are ordinal but can do later

# Can do head and tail of data to check for bizare values, eg 1000 years old
# And/or plot histograms of the data to check it looks normal (probabalby also statistically normal otherwise transform?)
# Remove certain obvious outliers if needed - must be actual methods for this

# Also some have commas so split them up into do they have X maybe - increases dimensionality though (see TODOs above)

# Train test split - but keep patient with same first three letters (same person) together

# Could put any extra files in another subfolder so they're out the way for other analysis

# Need to add actual prediction column
# And plot when this occurs - ie is it usually at first admission or later?

# In other model I plotted graphs for a lot of the variables. See what could be useful here

# PCA at different stages of data cleaning/feature selection (before/after filtering, after FS - but need numerical values

# Skimmed over data cleaning a lot. Normalising? Scaling/transforming? Outlier removal? What do I need to do.

# Impute but thats in the next .py. Better feature selection too

# Should add 'metadaa column number' as a variable so i only need to update one thing as the number of metadata cols changes.
# fine for now but with bigger data i'll want to be able to easily change (might need multiple values for before/after filtering

##### End of data_exploration.py, now go over preprocessing.py - in next doc