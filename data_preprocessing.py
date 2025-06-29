######### SETUP ########################################################################################################
# Import libraries
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import missingno as msno
import miceforest as mf
from sklearn.preprocessing import LabelEncoder
# TODO delete any unused at end

# WARNING NOTE IMPORTANT for test and train imputation, num_datsets and iterations are very low in an attempt to get it to run. must be changed and run on hpc

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Bool for any checking of the data that isn't needed for general use
show_testing = False

# Read in train and test data (and full dataset for testing) # TODO preserve patient groups? if not using D0 only - keep patient with same first three letters (same person) together
train_path = Path(__file__).parent / "Surrey_train.csv"
test_path = Path(__file__).parent / "Surrey_test.csv"
full_path = Path(__file__).parent / "Surrey_final.csv"
train = pd.read_csv(train_path, index_col=0)
test = pd.read_csv(test_path, index_col=0)
full_dataset = pd.read_csv(full_path, index_col=0)

### SPLIT DATA #########################################################################################################
# Split train and test data into X and y
X_train = train.drop('O2 req.',axis=1)
y_train = train['O2 req.'].copy()
X_test = test.drop('O2 req.',axis=1)
y_test = test['O2 req.'].copy()

# Check data is read in correctly (currently only does train)
if show_testing:
    print(y_train.head(5)) # Should be a series with SID as indexes, name is correct target variable (O2 req.)
    print(X_train.iloc[:5, :5]) # Should be a df with SID as indexes, column values look as expected

### Determine how many metadata columns remain
# Read in the original metadata file to read the column info
s_meta_file = Path(__file__).parent / "Surrey_Files" / "Surrey_Metadata_master_spreadsheet_130622_edit2.csv"
s_meta = pd.read_csv(s_meta_file)
# List starting and filtered columns
meta_columns = s_meta.columns.tolist()
existing_columns = X_train.columns.tolist()
# Work backwards through the list to find the first one still present
meta_cols = 0 # Initialise
matched = False
for col in reversed(meta_columns):
    if col in existing_columns:
        matched = True
        if show_testing:
            print("\nRemaining metadata columns:")
            print(X_train.columns.get_loc(col) + 1) # +1 for 1-based indexing conversion / allows for splicing where the first number is inclusive and the second exclusive
        meta_cols = X_train.columns.get_loc(col) + 1
        break
if not matched:
    if show_testing:
        print("All metadata columns were removed by the missingness filter.")
    meta_cols = 0

### HANDLE CATEGORICAL DATA FOR IMPUTATION #############################################################################
# Detect numeric vs categorical columns
numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
if show_testing:
    print("\nNumeric features:", numeric_cols)
    print("\nCategorical features:", cat_cols)
    # TODO still not sure if chol should be in here or not - investigate

# Check which test categories are binary/ordinal - use full dataset to check all regardless of train/test split
if show_testing:
    print("----- Test features: -----")
    for cat in cat_cols:
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
# For both X_train and X_test, convert to ordered categorical WITHOUT extracting codes

### Convert categories to pandas categorical - ordinal and nominal
# Ordinal categories
for cat, codes in ordinal_cats.items():
    # Convert to pandas category (ordered)
    X_train[cat] = pd.Categorical(X_train[cat], categories=codes, ordered=True)
    X_test[cat] = pd.Categorical(X_test[cat], categories=codes, ordered=True)

# Nominal categories - commented out for now as no nominal categories (also check as I realised my encoding above was previously wrong)
# for cat in nominal_cats.keys():
#     X_train[cat] = pd.Categorical(X_train[cat], ordered=False)
#     X_test[cat] = pd.Categorical(X_test[cat], ordered=False)

### NORMALISE PROTEOMICS DATA ########################################################################################## #TODO is it better to impute first?
# Separate into meta and MS data
X_train_meta = X_train.iloc[:, :meta_cols] # IMPROVE not currently used - could delete
X_train_quant = X_train.iloc[:, meta_cols:]

# Get sample medians per row
medians = X_train_quant.median(axis=1)
# Subtract the median (due to being log2-transformed)
X_train_quant = X_train_quant.sub(medians, axis=0)

### IMPUTE MISSING VALUES ##############################################################################################

# Plot MS missingness by average intensity to determine if MNAR
mv_ratio = X_train_quant.isna().mean() # Proportion of missing values
avg_intensity = X_train_quant.mean(skipna=True)
plot_df = pd.DataFrame({
    'Log2_Avg_Intensity': avg_intensity,
    'MV_Ratio': mv_ratio
})
plt.figure(figsize=(12,6))
sns.scatterplot(data=plot_df, x='Log2_Avg_Intensity', y='MV_Ratio', alpha=0.7)
plt.xlabel('Average Intensity (Log2)')
plt.ylabel('Proportion of Missing Values')
plt.grid(True)
plt.tight_layout()
plt.savefig('Missing_by_intensity.png')
# Result: Greater missingness at lower intensities suggests MNAR prevalence due to left censoring (below detection limit)

# Bool whether to impute - can turn this off to skip for future runs
impute = False

imputed_train = "Surrey_train_after_imputation_scaling.csv"
imputed_test = "Surrey_test_after_imputation_scaling.csv"

if impute:
    # Create a dataset to store intermediate columns for missingness handling
    train_missing = X_train.copy()

    # Store index values (have to reset for MICE)
    original_index_xtrain = X_train.index.copy()
    # Reset index for miceforest use
    train_missing = train_missing.reset_index(drop=True)
    train_missing = train_missing.replace([np.inf, -np.inf], np.nan)

    ### MAR Imputation for complete dataset with MICE    # Initialize kernel (handles categoricals natively)

    kernel = mf.ImputationKernel(data=train_missing, num_datasets=3, random_state=42) # todo could increase with more memory (and for test)

    # Run MICE with 10 iterations
    kernel.mice(iterations=5, min_data_in_leaf=3) # TODO iterations was set to 10 but memory runs out, restore if running on HPC # Default MDIL is I think 20; experiment with this.
    # kernel.plot_feature_importance(dataset=0) #todo commenting plots to try and reduce memory
    # kernel.plot_imputed_distributions()
    #
    # # Save feature importance plot
    # fig1 = kernel.plot_feature_importance(dataset=0)
    # plt.tight_layout()
    # plt.savefig('Surrey_feature_importance_plot.png')
    # plt.close(fig1)
    #
    # # Save imputed distributions plot
    # fig2 = kernel.plot_imputed_distributions()
    # plt.tight_layout()
    # plt.savefig('Surrey_imputed_distributions_plot.png')
    # plt.close(fig2)

    # # Save mean convergence plot #todo added from documentation so need to check
    # fig3 = kernel.plot_mean_convergence(dataset=0)
    # plt.tight_layout()
    # plt.savefig('Surrey_mean_convergence_plot.png')
    # plt.close(fig3)

    # Return dataset with missing values imputed
    train_missing = kernel.complete_data()

    # Restore the original index with SIDs
    train_missing.index = original_index_xtrain

    # Update X_train data with the imputed datasets
    X_train = train_missing

    # Save the dataset - not needed for future processing, just to check correct processing
    X_train.to_csv(imputed_train)

else: # If not imputing, read in the data
    print("Skipping imputation; used already produced imputed file. Otherwise set impute = True")
    X_train = pd.read_csv(imputed_train, index_col=0)



### Repeat for test data - IMPROVE this could be condensed with training - possibly with Pipeline
if impute:
    # Create a dataset to store intermediate columns for missingness handling
    test_missing = X_test.copy()

    # Store index values (have to reset for MICE)
    original_index_xtest = X_test.index.copy()
    # Reset index for miceforest use
    test_missing = test_missing.reset_index(drop=True)

    ### MAR Imputation for complete dataset with MICE
    # Initialize kernel (handles categoricals natively)
    kernel = mf.ImputationKernel(data=test_missing, num_datasets=3, random_state=42) # TODO Could increase with more memory

    # Run MICE with 10 iterations
    kernel.mice(iterations=5, min_data_in_leaf=3) # TODO was set to 10 but memory runs out, restore if running on HPC
    # kernel.plot_feature_importance(dataset=0) #todo commenting plots to try and redce memory
    # kernel.plot_imputed_distributions()
    #
    # # Save feature importance plot #todo haven't looked at these for the test or train data yet as i haven't been able to test generation, maybe remove. same for distributions
    # TODO there is also a get feature importance function that returns a matrix
    # TODO also complete_data vs MICE function in documentation? what's the difference
    # todo also tune hyperparameters? would that give better prediction
    # todo check miceforest usage examples - see github
    # fig1 = kernel.plot_feature_importance(dataset=0)
    # plt.tight_layout()
    # plt.savefig('Surrey_feature_importance_plot_test.png')
    # plt.close(fig1)
    #
    # # Save imputed distributions plot
    # fig2 = kernel.plot_imputed_distributions()
    # plt.tight_layout()
    # plt.savefig('Surrey_imputed_distributions_plot_test.png')
    # plt.close(fig2)

    # # Save mean convergence plot
    # fig3 = kernel.plot_mean_convergence(dataset=0)
    # plt.tight_layout()
    # plt.savefig('Surrey_mean_convergence_plot.png')
    # plt.close(fig3)

    # Extract completed data
    test_missing = kernel.complete_data()

    # Restore the original index with SIDs
    test_missing.index = original_index_xtest

    # Update X_test data with the imputed datasets
    X_test = test_missing

    # Save the dataset - not needed for future processing, just to check correct processing
    X_test.to_csv(imputed_test)

else: # If not imputing, read in the data
    print("Skipping imputation; used already produced imputed file. Otherwise set impute = True")
    X_test = pd.read_csv(imputed_test, index_col=0)

# Check columns are correctly imputed for categorical data (i.e still categorical)
if show_testing:
    for cat in ordinal_cats:
        print(f"\nUnique values in X_train {cat}: {X_train[cat].unique()}")
        print(f"Unique values in X_test {cat}: {X_test[cat].unique()}")
        # Should look the same as before imputation (probably unecessary testing now code is defined but leaving in)

### ENCODE CATEGORICAL DATA ############################################################################################
# Encode ordinal/binary data in X #IMPROVE can this be refined? eg sklearn OrdinalEncoder instead
for cat in ordinal_cats.keys():
    # Extract codes from the category dtype
    X_train[cat] = X_train[cat].cat.codes
    X_test[cat] = X_test[cat].cat.codes

    # Verify no missing values remain #todo might tweak this, haven't tested because of imputation memory issues
    assert X_train[cat].isna().sum() == 0
    assert X_test[cat].isna().sum() == 0

# Note: This dataset does not currently have any nominal categories, but otherwise one-hot encode here. See mental
# health data project for an example.

# Encode y data
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train_encoded= label_encoder.transform(y_train)
y_test_encoded  = label_encoder.transform(y_test)
# Convert back to df
y_train = pd.DataFrame(y_train_encoded, index=y_train.index, columns=["O2 req."])
y_test = pd.DataFrame(y_test_encoded, index=y_test.index, columns=["O2 req."])


### SAVE DATA ##########################################################################################################
# Write to csv for use in next script
X_train.to_csv("Surrey_X_train.csv", sep=",", index=True)
y_train.to_csv("Surrey_y_train.csv", sep=",", index=True)
X_test.to_csv("Surrey_X_test.csv", sep=",", index=True)
y_test.to_csv("Surrey_y_test.csv", sep=",", index=True)

#Improve: sklearn pipeline could probably improve the layout - or functions