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

# todo NOTE IMPORTANT for test and train imputation, num_datsets and iterations are very low in an attempt to get it to run. must be changed and run on hpc

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Bool for any checking of the data that isn't needed for general use
show_testing = False
# Read in train and test data (and full dataset for testing)
train_path = Path(__file__).parent / "Surrey_train.csv"
test_path = Path(__file__).parent / "Surrey_test.csv"
full_path = Path(__file__).parent / "Surrey_final.csv"
train = pd.read_csv(train_path, index_col=0)
test = pd.read_csv(test_path, index_col=0)
full_dataset = pd.read_csv(full_path, index_col=0)

# Split train data into X and y #IMPROVE could do an actual prediction column and not O2 requirment only
X_train = train.drop('O2 req.',axis=1)
y_train = train['O2 req.'].copy()
X_test = test.drop('O2 req.',axis=1)
y_test = test['O2 req.'].copy()

# Check data is read in correctly (currently only does train)
if show_testing:
    print(y_train.head(5)) # Is a series with SID as indexes, name is correct target variable (O2 req.)
    print(X_train.iloc[:5, :5]) # Index as SIDs, data looks as expected
    # Check target is not found in X
    try:
        print(X_train['O2 req.'])
        print("WARNING: Target found in X data - data processing has been done incorrectly.")
    except:
        print("Target column not found in X data, as expected")

### HANDLE CATEGORICAL DATA FOR IMPUTATION #############################################################################
# Detect numeric vs categorical columns
numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
if show_testing:
    print("Numeric:", numeric_cols)
    print("Categorical:", cat_cols)
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
                'Covid Positive Hospital Swab (Y/N)' : ['N', 'Inconclusive', 'N - previously pos in ICU', 'Y'], #IMPROVE Can I make the middle two the same order? ALthough for only two datapoints it doesn't really matter
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


### IMPUTE MISSING VALUES ##############################################################################################
#TODO current method just assumes MAR for all and uses MICE. In reality I need to investigate the data more - e.g. MNAR
# due to low abundance, vs possibly overlap with other peaks? Assess missingness using proven methods and handle accordingly
#  Also could log2 transform to reduce variacne (check validity ) and normalise if needed

# Bool whether to impute - can skip for future runs
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

    ### MAR Imputation for complete dataset with MICE
    # Initialize kernel (handles categoricals natively)
    kernel = mf.ImputationKernel(data=train_missing, num_datasets=3, random_state=42) # todo could increase with more memory (and for test)

    # Run MICE with 10 iterations
    kernel.mice(iterations=5) # TODO was set to 10 but memory runs out, restore if running on HPC
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

    # Extract completed data
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
    kernel.mice(iterations=5) # TODO was set to 10 but memory runs out, restore if running on HPC
    # kernel.plot_feature_importance(dataset=0) #todo commenting plots to try and redce memory
    # kernel.plot_imputed_distributions()
    #
    # # Save feature importance plot #todo haven't looked at these for the test or train data yet as i haven't been able to test generation, maybe remove. same for distributions
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
#TODO: will have to adjust this earlier on, but realised ordinal numerical categories like airway disease probably shouldn't be
#  scaled (although now i removed scaling here but still potentially an issue for imputing - check the post-imputation data).
#  possibly should be treated as categorical despite being numbers - e.g. don't want to impute a '3.5' between integer categories
#  Should look over the data when I have more time, and check they look as expected.
#  For now leaving but it needs to be checked/fixed.

# Encode ordinal/binary data
for cat in ordinal_cats.keys():
    # Extract codes from the category dtype
    X_train[cat] = X_train[cat].cat.codes
    X_test[cat] = X_test[cat].cat.codes

    # Verify no missing values remain #todo might tweak this, haven't tested because of imputation memory issues
    assert X_train[cat].isna().sum() == 0
    assert X_test[cat].isna().sum() == 0

# Note: This dataset does not currently have any nominal categories, but otherwise one-hot encode here. See mental
# health data project for an example.

# Encode y_train # IMPROVE currently fit separate label encoders to y_train and y_test, which could lead to inconsistent encoding if classes differ
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_encoded= label_encoder.transform(y_train)
# Convert back to df
y_train = pd.DataFrame(y_encoded, index=y_train.index, columns=["O2 req."])

# Encode y_test
label_encoder = LabelEncoder()
label_encoder.fit(y_test)
y_encoded= label_encoder.transform(y_test)
# Convert back to df
y_test = pd.DataFrame(y_encoded, index=y_test.index, columns=["O2 req."])


### SAVE DATA ##########################################################################################################
# Write to csv for use in next script
X_train.to_csv("Surrey_X_train.csv", sep=",", index=True)
y_train.to_csv("Surrey_y_train.csv", sep=",", index=True)
X_test.to_csv("Surrey_X_test.csv", sep=",", index=True)
y_test.to_csv("Surrey_y_test.csv", sep=",", index=True)

#todo
# From imputation: UserWarning: Covid Positive Hospital Swab (Y/N) have very rare categories, it is a good idea to group these, or set
# the min_data_in_leaf parameter to prevent lightgbm from outputting 0.0 probabilities.

#Improve: sklearn pipeline could probably improve the layout - or functions

#todo add msno graph to show no missing data

#todo check testing is indented where relevant