from datetime import datetime
import socket
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree, export_text
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from xgboost import to_graphviz
from xgboost import plot_tree as xgb_plot_tree
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import re
import missingno as msno
import miceforest as mf
from itertools import repeat, chain

### CHART STYLES #######################################################################################################
# Return specified colour palette so palettes can be accessed globally
def get_palette(palette): #WARNING - check that no
    # Colour palettes # IMPROVE - more distance for blues, could add less range in R-Y section
    colour_palette_10 = ["#E52B3E", "#DE5474", "#EE7956", "#F7C552", "#72C872", "#0EBC8D", "#129B95", "#167B9C",
                         "#2F6B9F", "#BA439F"] # Rainbow
    colour_palette_9 = ["#E52B3E", "#DE5474", "#EE7956", "#F7C552", "#72C872", "#0EBC8D", "#129B95", "#2F6B9F", "#BA439F"] # Near rainbow
    colour_palette_8 = ["#E52B3E", "#EE7956", "#F7C552", "#0EBC8D", "#129B95", "#167B9C", "#485BA3", "#BA439F"] #ROYGBPP
    colour_palette_7 = ["#E52B3E", "#EE7956", "#F7C552", "#0EBC8D", "#129B95", "#167B9C", "#485BA3"] # ROYGBP
    colour_palette_6 = ["#E52B3E", "#EE7956", "#F7C552", "#72C872", "#129B95", "#2F6B9F"] #ROYGGB
    colour_palette_5 = ["#E52B3E", "#EE7956", "#F7C552", "#0EBC8D", "#167B9C"] #ROYGB
    colour_palette_4 = ["#E52B3E", "#F7C552", "#0EBC8D", "#485BA3"] # RYGB
    colour_palette_3 = ["#E52B3E", "#72C872", "#485BA3"] # RGB
    colour_palette_2 = ["#E52B3E", "#485BA3"] # RB
    colour_palette_2r = colour_palette_2[::-1] # IMPROVE would have been better to do palette 2 as [0] blue (negative and [1] red (postive) but it's eaier to use an alternative palette than replace
    colour_palette_1 = ["#0EBC8D"]  # G
    colour_palette_rwb = ["#E52B3E", "#FFFFFF", "#485BA3"] # RWB for cmaps
    colour_palette_light_dark = ["#E52B3E", "#92111E", "#485BA3", "#2C3863"]  # RlRdBlBd

    # Create colourmaps
    cmap_rwb = LinearSegmentedColormap.from_list("cmap_rwb", colour_palette_rwb)

    # Dictionary lookup
    palette_dict = {
        '10': colour_palette_10,
        '9': colour_palette_9,
        '8': colour_palette_8,
        '7': colour_palette_7,
        '6': colour_palette_6,
        '5': colour_palette_5,
        '4': colour_palette_4,
        '3': colour_palette_3,
        '2': colour_palette_2,
        '2r': colour_palette_2r,
        '1': colour_palette_1,
        'rwb': colour_palette_rwb,
        'ld': colour_palette_light_dark,
        'cmap_rwb': cmap_rwb
    }
    return palette_dict[palette]



def set_graph_style():
    # Custom font
    font_path = "Merriweather_24pt-Medium.ttf"
    font_path_bold = "Merriweather_24pt-ExtraBold.ttf"
    fm.fontManager.addfont(font_path)
    fm.fontManager.addfont(font_path_bold)
    custom_font = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = custom_font


    # Seaborn settings
    sns.set_theme(
        font=custom_font,  # Font family
        font_scale=1.1,  # Font scaling factor
        rc={  # Matplotlib settings
        # Figure and font sizing
        'figure.figsize': (7, 5),
        'figure.dpi': 300,
        'figure.autolayout': True,  # Automatically adjust layout
        'font.size': 14,  # Base font size

        # Line and marker styles
        'lines.linewidth': 2.5,  # Line width
        'lines.markersize': 8,  # Marker size
        'lines.markeredgewidth': 0.5,  # Marker edge width

        # Axes
        'axes.titlesize': 14,  # Title font size
        'axes.labelsize': 10,  # Axis label font size
        'axes.linewidth': 1.5,  # Axis border width
        'axes.edgecolor': 'black',  # Axis edge color
        'axes.facecolor': 'white',  # Plot background color
        'axes.spines.right': False,  # Remove right spine
        'axes.spines.top': False,  # Remove top spine
        'axes.titleweight': 'bold',

        # Ticks
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.labelsize': 8,  # X-tick label size
        'ytick.labelsize': 8,  # Y-tick label size
        'xtick.major.size': 5,  # X-tick length
        'ytick.major.size': 5,  # Y-tick length
        'xtick.major.width': 1.5,  # X-tick width
        'ytick.major.width': 1.5,  # Y-tick width
        'xtick.minor.visible': False,  # Show minor ticks
        'ytick.minor.visible': False,  # Show minor ticks

        # Legend
        'legend.fontsize': 8,  # Legend font size
        'legend.frameon': False,  # Show legend box
        'legend.framealpha': 0,  # Legend transparency

        # Grid
        'axes.grid': True,
        'grid.color': 'lightsteelblue',  # Grid line color
        'grid.linestyle': 'solid',  # Grid line style
        'grid.alpha': 0.5,  # Grid transparency
        'grid.linewidth': 0.8,  # Grid line width

        # Colors
        'axes.prop_cycle': plt.cycler(color=get_palette("10")),

        # Saving figures
        'savefig.dpi': 300,  # Save resolution
        'savefig.format': 'png',  # Default save format
        'savefig.bbox': 'tight',  # Remove extra whitespace
        'savefig.transparent': False,  # Background transparency

        # Error bars
        'errorbar.capsize': 3,  # Error bar cap size
    })
    sns.set_palette(get_palette("2"), desat=1)

# Apply graph styles
set_graph_style() # Call in functions.py

### TYPE TRANSLATION ###################################################################################################
# Translation of short type to presentable string
type_translation = { #Note: unlike model_building.py, these are not capitalised in order to look better for graphs. #IMPROVE make the two consistent
    'svm': 'Support vector classifier',
    'rf': 'Random forest classifier',
    'logreg': 'Logistic regression',
    'xgb': 'XGBoost classifier',
    'gb': 'Gradient boosting classifier',
    'ada': 'AdaBoost classifier',
    'knn': 'K-Nearest neighbors classifier',
    'nn': 'Neural network'
    }

### FEATURE ENGINEERING.PY #############################################################################################
# Check for abnormal SIDs in the data; any unexpected lengths
def check_abnormal_SIDs(quant_samples, before_samples, samples, expected_length):
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
    print("Abnormal SIDs in quant data:")
    print(comparison_quant[comparison_quant["Length after"] != expected_length])
    abnormal_quant_SIDs = quant_samples[quant_samples.index.astype(str).str.len() != expected_length].iloc[:, 0:5]
    print("\nThe abnormal rows in the quant data file:\n", abnormal_quant_SIDs)

# Calculate unique samples IDs and detect overlaps between the SIDs
def calculate_overlaps(quant_samples, sample_index, sample_inves_3):
    quant_samples_set = set(quant_samples) # WARNING: Removed the .index - check this still works
    meta_samples_modified = set(sample_index)
    sample_overlap_quant = quant_samples_set & meta_samples_modified
    quant_unique = quant_samples_set - meta_samples_modified
    s_meta_unique_quant = meta_samples_modified - quant_samples_set
    # Print results if enabled
    if sample_inves_3:
        print("Number of overlapping samples with quant:", len(sample_overlap_quant))
        print("Number only in quant samples:", len(quant_unique))
        print("Number only in s_meta_modified samples:", len(s_meta_unique_quant))

    # Optional for testing:
    # print("\nWhat are the actual sample values?")
    # print("Overlapping samples with quant:\n", sample_overlap_quant)
    # print("\n\nQuant unique samples:\n", quant_unique)
    # print("\n\nMetadata modified unique samples:\n", s_meta_unique_quant)

# Print column value_counts and uniques for either one column or optionally all
def check_columns(merged, meta_cols, removed_cols, check_only_one, selected_column, check_set):
    # Assign which columns to check
    columns_to_check = [feature for feature in merged.columns[0:meta_cols] if feature not in removed_cols]
    if check_only_one:
        columns_to_check = selected_column

    if check_set:
        for column in columns_to_check:  # To check only one at a time for easier interpretation, change check_only_one to 'True' and edit the variable to the col title as needed.
            print("\nValue counts:\n", merged[column].value_counts())
            print("Uniques:\n", merged[column].unique())
    return columns_to_check

# View columns with no data and NA headers
def check_empty_cols(merged, dataset, status):
    print(f"Number of columns in {dataset} {status} removing empty columns:")
    print(len(merged.columns))
    dupe_cols = merged.columns[merged.columns.isnull() | (merged.columns == "")]
    print(f"Number of duplicated columns in {dataset} {status} removing empty columns:")
    print(len(dupe_cols))
    if len(dupe_cols) > 0:
        print("Duplicate cols:", dupe_cols.tolist())

def plot_row_missingness(merged, meta_cols, sample_inves_7, graphs_dir, dataset):
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
        print(f"Missing values distribution in the {dataset} metadata:")
        print(missing_distribution.to_string(index=False))

    ### Plot missingness
    plt.figure(figsize=(10, 6))
    sns.barplot(x='NA_Count', y='Rows', data=missing_distribution, palette='Blues_d', edgecolor='black', hue='NA_Count',
                legend=False, saturation=1.0)
    plt.title('Missing Values in the Metadata')
    plt.xlabel('Number of Missing values per samples')
    plt.ylabel('Number of samples')
    # Add percentage labels
    for index, row in missing_distribution.iterrows():
        plt.text(row.name, row.Rows + 1,  # Offset above bar
                 f'{row.Percentage}%', ha='center')
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}/missing_distribution_{dataset}.png', dpi=300)
    return merged_null_before

    # IMPROVE plot for quant data as well?

# Plot missingness in columns
def plot_missingness_msno(merged, dataset, meta_cols, graphs_dir):
    fig = msno.matrix(merged)
    fig_copy = fig.get_figure()
    fig_copy.set_tight_layout(False)
    with plt.style.context({'figure.autolayout': False}):
        fig_copy.savefig(f'{graphs_dir}/Missingness_All-data_before_filtering_{dataset}.png')
        # Plot missingness of the metadata
        fig = msno.matrix(merged.iloc[:, 0:meta_cols])
        fig_copy = fig.get_figure()
        fig_copy.savefig(f'{graphs_dir}/Missingness_All-metadata_before_filtering_{dataset}.png')
        # Plot missingness of the quant data
        fig = msno.matrix(merged.iloc[:, meta_cols:])
        fig_copy = fig.get_figure()
        fig_copy.savefig(f'{graphs_dir}/Missingness_All-quantdata_before_filtering_{dataset}.png')

# Investigate null values in each column
def investigate_null(merged, dataset, merged_null_before, sample_inves_7, output_dir):
    if sample_inves_7:
        print(f"{dataset} missing count per column:")
        print(merged.isnull().sum(), "\n")
        print("Column count before filtering for missing data:")
        print(len(merged.columns))

        # Extract missingness of below 30% and keep those columns


    merged_low_missing = merged_null_before[merged_null_before['Missing_Percentage_Before'] < 30].index.tolist()
    merged = merged[merged_low_missing]
    merged.to_csv(f"{output_dir}/Surrey_data_low_missing.csv")

    if sample_inves_7:
        print("\nColumn count after filtering for missing data:")
        print(len(merged.columns))

    # Summarise missing values in columns after filtering
    merged_null_after = merged.isnull().sum().to_frame(name='Missing_Count_After')
    merged_null_after['Missing_Percentage_After'] = (merged_null_after['Missing_Count_After'] / len(merged)) * 100

    # Combine the information on missingness data before and after column filtering
    merged_null_summary = pd.concat([merged_null_before, merged_null_after], axis=1, sort=False).fillna('[Column removed]')

    # Save missingness info before and after filtering to csv
    merged_null_summary.to_csv(f"{output_dir}/Missing_values_comparison_{dataset}.csv",
                               header=["NA Count Before", "Missingness (%) Before", "NA Count After",
                                       "Missingness (%) After"],
                               float_format="%.1f")
    # Report how many columns were removed
    num_cols_before = merged_null_before.shape[0]
    num_cols_after = len(merged.columns)
    num_removed = num_cols_before - num_cols_after

    print(f"{num_removed} columns in the {dataset} dataset were removed due to ≥30% missingness.")
    return merged

# Determine remaining metadata columns and plot with missingno
def remaining_meta(meta_columns, merged, sample_inves_7, graphs_dir):
    # Determine remaining columns
    existing_columns = merged.columns.tolist()
    # Work backwards through the list to find the first one still present
    meta_cols = 0  # Initialise
    for col in reversed(meta_columns):
        if col in existing_columns:
            if sample_inves_7:
                print("\nRemaining metadata columns:")
                print(merged.columns.get_loc(col) + 1)  # +1 for 1-based indexing conversion / allows for splicing where the first number is inclusive and the second exclusive
            meta_cols = merged.columns.get_loc(col) + 1
            break
        else:
            if sample_inves_7:
                print("All metadata columns were removed by the missingness filter.")
            meta_cols = 0

    # Plot missingness after filtering (full dataset)
    if graphs_dir:
        with plt.style.context({'figure.autolayout': False}):
            fig = msno.matrix(merged)
            fig_copy = fig.get_figure()
            fig_copy.savefig(f'{graphs_dir}/Missingness_All-data_after_filtering.png')
            # Plot missingness of the metadata
            fig = msno.matrix(merged.iloc[:, 0:meta_cols])
            fig_copy = fig.get_figure()
            fig_copy.savefig(f'{graphs_dir}/Missingness_All-metadata_after_filtering.png')
            # Plot missingness of the quant data
            fig = msno.matrix(merged.iloc[:, meta_cols:])  # Note: This is currently 546-27=519 MS data columns
            fig_copy = fig.get_figure()
            fig_copy.savefig(f'{graphs_dir}/Missingness_All-quantdata_after_filtering.png')
    return meta_cols

# Categorise as numerical or categorical
def categorise_cols(merged, sample_inves_8):
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
    return numerical_cols, categorical_cols

# Check if any numerical columns had values converted to NaN
def numerical_check_nan(merged, numerical_cols, categorical_cols, sample_inves_8, dataset):
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
        after = post_missing[col]
        if after > before:  # Note this shows regardless of show_inves or not
            print(f"Column {col!r} gained {after - before} new NaN(s) (was {before}, now {after})")
            print("Investigate with .unique/.value_counts or similar to compare.")
            cols_nanned += 1

    if sample_inves_8:
        if cols_nanned == 0:
            print(f"\nNone of the {dataset} columns had any real values converted to NaN.")

# Plot class distribution in the dataset
def plot_class_distribution(merged, graphs_dir, dataset):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='O2 req.', data=merged, palette='Blues_d', edgecolor='black', hue='O2 req.',
                  legend=False)
    plt.title(f'O2 requirement class distribution - {dataset} dataset')
    plt.xlabel('O2 required')
    plt.ylabel('Count')
    plt.savefig(f'{graphs_dir}/class_distribution_{dataset}.png', dpi=300)

# Convert text-based comma-split columns to binary
def text_to_binary(dataset, col_name, col_type, training_data, min_count):
    # Process text into uniform lists
    dataset[col_name] = (
        dataset[col_name].fillna('').str.lower().str.split(',').apply(lambda items: [item.strip() for item in items if item.strip()]))

    # Function to remove the last item in the cell if there are multiple items (i.e. remove '' except if the cell is truly NaN)
    def remove_trailing_comma(contents):
        if isinstance(contents, list) and len(contents) > 2 and contents[-1] == '':
            return contents[:-1]
        return contents

    dataset[col_name] = dataset[col_name].apply(remove_trailing_comma)

    # Create binary columns
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(dataset[col_name])
    binary_df = pd.DataFrame(matrix, columns=[f"{col_type}_ {m}" for m in mlb.classes_],
                             index=dataset.index)

    # Count medication frequencies (also used to check for errors) and save to csv
    counts = pd.DataFrame({col_type: mlb.classes_, 'Count': matrix.sum(axis=0)})
    counts = counts.sort_values('Count', ascending=False)
    counts.to_csv(f'{training_data}/Frequencies_{col_type}.csv', index=False)

    # Drop rare binary columns
    frequent_cols = counts[counts['Count'] >= min_count][col_type].tolist()
    cols_added = len(frequent_cols)
    binary_df = binary_df[[f"{col_type}_ {m}" for m in frequent_cols]]

    # Drop old column and add new ones
    col_position = dataset.columns.get_loc(col_name)
    dataset = (dataset.drop(col_name, axis=1))

    # Add columns at position of old column by splitting and rejoing the dataset
    dataset = pd.concat([dataset.iloc[:, :col_position], binary_df, dataset.iloc[:, col_position:]],
                              axis=1)

    print(f"The '{col_name}' column has been expanded into {len(frequent_cols)} binary columns.")
    return dataset, cols_added

    # Apply cleaned names to DataFrame
    df_clean = df.copy()
    df_clean.columns = new_columns
    return df_clean

# Replace values found within a column to fix typos
def replace_values(df, column_name, original, replacement):
    # Escape special regex characters in the original string
    escaped_original = re.escape(original)
    # Create regex pattern with word boundaries
    pattern = r'\b' + escaped_original + r'\b'
    # Perform replacement with case insensitivity
    df[column_name] = df[column_name].str.replace(pattern, replacement, case=False, regex=True)
    return df

### DATA_PREPROCESSING.PY FUNCTIONS ####################################################################################
# Convert categories to pandas categorical - ordinal and nominal
def convert_categories(dataset, ordinal_cats):
    # Ordinal categories
    for cat, codes in ordinal_cats.items():
        if cat in dataset.columns: # Check that the column is present - allows same dict to be used for multiple datasets
            # Convert to pandas category (ordered)
            dataset[cat] = pd.Categorical(dataset[cat], categories=codes, ordered=True)

    # Nominal categories - commented out for now as no nominal categories (also check as I realised my encoding above was previously wrong) - function is not updated for this either
    # for cat in nominal_cats.keys():
    #     dataset[cat] = pd.Categorical(dataset[cat], ordered=False)
    return dataset

# Normalise the MS data
def normalise_MS(dataset, meta_cols):
    # Separate out MS data
    dataset_quant = dataset.iloc[:, meta_cols:]
    # Get sample medians per row
    try:
        medians = dataset_quant.median(axis=1)
    except:
        print("WARNING: Could not determine medians. Saving quant dataset to quant.csv in cwd.") #todo testing only
        print("Columns in 'quant' are:", dataset_quant.columns)
        exit(1)
    # Subtract the median (due to being log2-transformed)
    dataset_quant = dataset_quant.sub(medians, axis=0)
    return dataset_quant

# Plot missigness in the MS data by intensity
def plot_missingness_ms(dataset, graphs_dir, name):
    # Plot MS missingness by average intensity to determine if MNAR
    mv_ratio = dataset.isna().mean()  # Proportion of missing values
    avg_intensity = dataset.mean(skipna=True)
    plot_df = pd.DataFrame({
        'Log2_Avg_Intensity': avg_intensity,
        'MV_Ratio': mv_ratio
    })
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=plot_df, x='Log2_Avg_Intensity', y='MV_Ratio', alpha=0.7)
    plt.xlabel('Average Intensity (Log2)')
    plt.ylabel('Proportion of Missing Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{graphs_dir}/Missing_by_intensity_{name}.png')
    # Result: Greater missingness at lower intensities suggests MNAR prevalence due to left censoring (below detection limit)

# Impute with miceforest package (MICE imputation)
def impute_MICE(dataset, filename, datastring, num_datasets, iterations, graphs_dir):
    # Create a dataset to store intermediate columns for missingness handling
    dataset_missing = dataset.copy()

    # Store index values (have to reset for MICE)
    original_index_dataset = dataset.index.copy()
    # Reset index for miceforest use
    dataset_missing = dataset_missing.reset_index(drop=True)
    dataset_missing = dataset_missing.replace([np.inf, -np.inf], np.nan)

    ### MAR Imputation for complete dataset with MICE
    print(f"Starting MICE imputation with {num_datasets} datasets and {iterations} iterations") # WARNING - hoping to increase these after imputation. mean match, iterations, num_datasets, and n_jobs. also graphs are commented.
    try:
        # Initialize kernel (handles categoricals natively)
        kernel = mf.ImputationKernel(
            data=dataset_missing,
            num_datasets=num_datasets,
            mean_match_candidates=5,
            random_state=42
        )

        # Run MICE with explicit single-threading
        print("Running MICE iterations...")
        kernel.mice(
            iterations=iterations,
            min_data_in_leaf=3,
            n_jobs=1,
            random_state=42
        )

        print("MICE iterations completed successfully")

    except Exception as e:
        print(f"Error during MICE imputation: {str(e)}")
        raise

    # # Save feature importance plot #TODO plots commented again to try and run - issues with threading on HPC
    # # todo also tune hyperparameters? would that give better prediction
    # # todo check miceforest usage examples - see github
    # fig1 = kernel.plot_feature_importance(dataset=0) # WARNING - plots are untested - if unnecessary then remove
    # plt.tight_layout()
    # plt.savefig(f'{graphs_dir}/{datastring}_feature_importance_plot.png')
    # plt.close(fig1)

    # Save imputed distributions plot
    # fig2 = kernel.plot_imputed_distributions()
    # plt.tight_layout()
    # plt.savefig(f'{graphs_dir}/{datastring}_imputed_distributions_plot.png')
    # plt.close(fig2)
    #
    # # Save mean convergence plot #todo added from documentation so need to check
    # fig3 = kernel.plot_mean_convergence() # todo check it converges
    # plt.tight_layout()
    # plt.savefig(f'{graphs_dir}/{datastring}_mean_convergence_plot.png')
    # plt.close(fig3)

    # Return dataset with missing values imputed
    dataset_missing = kernel.complete_data()

    # Restore the original index with SIDs
    dataset_missing.index = original_index_dataset

    # Update X_train data with the imputed datasets
    dataset = dataset_missing

    # Save the dataset - not needed for future processing, just to check correct processing
    dataset.to_csv(filename)
    return dataset

# Encode categorical data
def encode_categorical(dataset, ordinal_cats):
    # Encode ordinal/binary data in X # IMPROVE can this be refined? eg sklearn OrdinalEncoder instead
    for cat in ordinal_cats.keys():
        if cat in dataset.columns:  # Check that the column is present - allows same dict to be used for multiple datasets
            # Extract codes from the category dtype
            dataset[cat] = dataset[cat].cat.codes

            # Verify no missing values remain #todo might tweak this, haven't tested because of imputation memory issues
            assert dataset[cat].isna().sum() == 0

# Encode y data
def encode_y(y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)
    # Convert back to df
    y = pd.DataFrame(y_encoded, index=y.index, columns=["O2 req."])
    return y

### MODEL_BUILDING.PY FUNCTIONS ########################################################################################
# # Detect metadata columns in the dataset - No longer in use # TODO remove at end if remains unused
# def count_meta(dataset, name, metadata_features, drop, show_detail):
#     matched = False # Initialise
#     existing_columns = dataset.columns.tolist()
#     col_number = 0 # Initialise
#     for col in reversed(metadata_features):
#         if col in existing_columns:
#             matched = True
#             if show_detail:
#                 print(f"\nMetadata columns in {name}:")
#                 print(dataset.columns.get_loc(col) + 1) # +1 for 1-based indexing conversion / allows for splicing where the first number is inclusive and the second exclusive
#             col_number = dataset.columns.get_loc(col) + 1
#             print(col_number)
#             break
#     if not matched:
#         if show_detail:
#             print("No metadata columns found.")
#     if drop: # Drop the metadata if bool is true
#         dataset = dataset.iloc[:, col_number:]
#         col_number = 0 # Now removed all metadata so count is 0
#         print(f"Metadata was dropped from {name}; if unintended, disable drop_metadata in the script.")
#     return col_number,dataset

# Basic model training function to get some initial scores and decide which model to proceed with
def basic_train(model, X_train, y_train, identifier, scores_dict, feature_selectors, feature_selection, threshold):
    # Iterate over feature selectors
    model_results = {}
    overall_summary = []
    for fs_name, selector_config in feature_selectors.items():
        # Build selector from configuration
        if fs_name == 'NONE':
            selector = 'passthrough'
        else:
            # Instantiate selector with base parameters
            selector = selector_config['class'](**selector_config['base_params'])

        # Create a pipeline with a) the required preprocessing steps and b) the FS and model. This is then applied to each CV fold to avoid data leakage vs applying to all of X_train
        pipe = Pipeline([
            ('preprocessor', Pipeline([
                ('int_to_float', IntToFloatTransformer()),
                ('var_thresh', VarianceThreshold(threshold=threshold)),
                ('scaler', StandardScaler())
            ])),
            ('feature_selector', selector if feature_selection else 'passthrough'),
            ('classifier', model)
        ])
        try:
            # 10-fold cross validation for AUROC score and accuracy
            roc_val = cross_val_score(pipe, X_train, y_train, scoring='roc_auc', cv=StratifiedKFold(10, shuffle=True, random_state=42))
            accuracy_val = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=StratifiedKFold(10, shuffle=True, random_state=42))

            # Fit the pipeline on the training data
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_train)
            y_proba = pipe.predict_proba(X_train)[:, 1]
            # Metrics calculation
            roc_train = roc_auc_score(y_train, y_proba)
            accuracy_train = accuracy_score(y_train, y_pred)

            # List results for each feature selection method
            model_results[fs_name] = [identifier, fs_name, accuracy_train, accuracy_val.mean(), roc_train, roc_val.mean(), ]
            print(f"Training of {identifier} using {fs_name} complete.")
        except Exception as e:
            print(f"Error training {identifier} with {fs_name}: {str(e)}")
            model_results[identifier] = [identifier, None, None, None, None, None]

    print(f"*** Finished tuning {model}. Current date/time is {datetime.now().strftime('%m-%d %H:%M')} ***")  # TODO testing
    # Print results from best feature selection methods
    model_results_df = pd.DataFrame.from_dict(model_results,
                           orient='index',
                           columns=['Model', 'Selector', 'Train Accuracy', 'CV Accuracy', 'Train AUROC',
                                    'CV AUROC']).sort_values(by=['CV AUROC'], ascending=False)
    print(f"Metrics from {identifier} experimentation:")
    print(model_results_df, "\n")

    # Take the top result unless empty
    if model_results_df.empty:
        scores_dict[identifier] = [identifier, None, None, None, None, None]
        print(f"All feature selection methods failed for {identifier}.")
    else:
        scores_dict[identifier] = model_results_df.iloc[0].to_list()
    print(f"Finished training {identifier}")

    # Return the overall results list for the model
    return model_results_df

# Plot the performance of feature selectors per model
def plot_fs_performance(all_results_sorted, graphs_dir):
    # Determine palette based on number of selectors
    fs_count = len(all_results_sorted["Selector"].unique())
    if fs_count <= 10: # IMPROVE Currently only have ten available options for colour with defined palettes - could add more
        palette = get_palette(f"{fs_count}")
    else:
        palette = get_palette("10")

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=all_results_sorted,
                 x='Model',
                 y='CV AUROC',
                 hue='Selector',
                 style='Selector',
                 markers=True,
                 dashes=False,
                 palette=palette
                 )

    plt.title('Feature Selector Performance Across Models')
    plt.xlabel('Model')
    plt.ylabel('CV AUROC Score')
    plt.xticks(rotation=15)
    plt.legend(title='Feature Selectors', title_fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)  # TODO test
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{graphs_dir}/selector_performance.png")

# Convert integers to floats
class IntToFloatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Only convert if DataFrame (preserves column names)
        if isinstance(X, pd.DataFrame):
            int_cols = X.select_dtypes(include=['int', 'int32', 'int64']).columns
            X[int_cols] = X[int_cols].astype(float)
        return X

# Check if port is in use for MLFlow
def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

### MODEL GRAPHS #######################################################################################################
# Plot confusion matrix
def plot_confusion_matrix(cm, graphs_dir):
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest", cmap=get_palette("cmap_rwb")) # IMPROVE: don't love this colour mapping

    # Label axes
    classes = ["O2 not required", "O2 required"]
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="Actual Oxygen Need",
        xlabel="Predicted Oxygen Need"
    )

    # Annotate each cell with the raw count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" #IMPROVE white unless value is within a certain range of zero (can use cmax to determine) in which case black so to be visible
            )

    fig.savefig(f"{graphs_dir}/confusion_matrix.png")
    plt.close(fig)

# Plot PCA on the combined dataset before feature selection
def plot_pca_original(X_train, X_test, y_train, y_test, graphs_dir):
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

        plt.figure()

        # Plot directly from arrays
        plt.scatter(principal_components[y_full == 0, 0],
                    principal_components[y_full == 0, 1],
                    c=get_palette("2")[1], alpha=0.7, edgecolors=get_palette("2")[1], label='Does not require O₂')
        plt.scatter(principal_components[y_full == 1, 0],
                    principal_components[y_full == 1, 1],
                    c=get_palette("2")[0], alpha=0.7, edgecolors=get_palette("2")[0], label='Requires O₂')

        # Add explained variance
        explained_var = pca.explained_variance_ratio_ * 100
        plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
        plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
        plt.title('PCA of Full Dataset - Surrey')
        plt.legend()

        # Save and show
        plt.savefig(f"{graphs_dir}/pca_all_data.png")
        plt.close()

    except Exception as e:
        print(f"Error creating PCA biplot on full dataset: {str(e)}")

# IMPROVE: PCA functions could be consolidated (especially if combining for plot_pca_original etc outside of the function)
#  Note: this is named to plot pre and post FS, but currently just does it post ('after')
# Plot PCA on the combined dataset - i.e. original data after feature selection
def pca_pre_post_fs(X_full, selected_features, y_full, graphs_dir, fs_state):
    # Select chosen features only
    X_selected = X_full[selected_features]

    # Standardize the selected data
    scaler = StandardScaler()  # WARNING Avoiding using the same one as in the pipeline to prevent data leakage - not sure if it's an issue but it will error when called later if used here due to different number of features
    X_scaled = scaler.fit_transform(X_selected)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df = pc_df.reset_index(drop=True)  # Ensure index alignment

    plt.figure()

    # Create boolean masks
    class_0_mask = (y_full == 0)
    class_1_mask = (y_full == 1)

    # Plot using the aligned indices
    plt.scatter(pc_df.loc[class_0_mask, 'PC1'],
                pc_df.loc[class_0_mask, 'PC2'],
                c=get_palette("2")[1], alpha=0.7, edgecolors=get_palette("2")[1], label='O2 not required')
    plt.scatter(pc_df.loc[class_1_mask, 'PC1'],
                pc_df.loc[class_1_mask, 'PC2'],
                c=get_palette("2")[0], alpha=0.7, edgecolors=get_palette("2")[0], label='O2 required')

    # Add explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title(f'PCA {fs_state} Feature Selection')
    plt.legend()

    # Save and show
    plt.savefig(f"{graphs_dir}/pca_full_dataset_{fs_state}_FS.png")
    plt.close()

# Plot learning curve # TODO learning curve, precision recall etc have a legend and title with AP labelled - remove titles before final report
def plot_learning_curve(final_pipeline, X_train, y_train, graphs_dir):
    # Compute scores at varying training sizes
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=final_pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
    )
    # Plot the learning curve
    fig, ax = plt.subplots()
    LearningCurveDisplay(
        train_sizes=train_sizes,
        train_scores=train_scores,
        test_scores=val_scores
    ).plot(ax=ax)
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Learning Curve")

    # Save the plot
    plt.savefig(f"{graphs_dir}/learning_curve.png")

    # Log the figure as an MLflow artifact
    mlflow.log_figure(fig, f"{graphs_dir}/learning_curve.png")
    plt.close(fig)

# Plot ROC_AUC curve
def plot_roc_auc(y_proba, y_test, graphs_dir):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot and log
    fig, ax = plt.subplots()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
    mlflow.log_figure(fig, f"{graphs_dir}/roc_curve.png")

    # Save the plot
    plt.savefig(f"{graphs_dir}/roc_curve.png")
    plt.close(fig)

# Plot feature importance (or permutation)
def plot_feature_importance(classifier_type, final_pipeline, selected_features, graphs_dir, data_dir, best_params,
                            X_test, y_test, meta_cols=None): # todo take out x test and y test from function calls - not used
    set_graph_style()
    ### Extract feature importances based on the classifier type
    if classifier_type in ['rf', 'xgb', 'gb']:
        # Get feature importances if present
        importances = final_pipeline.named_steps['classifier'].feature_importances_
        importance_column = 'Importance'
    elif classifier_type == 'svm' and best_params.get('kernel') == 'linear':
        # For linear SVM, extract coefficients
        importances = np.abs(final_pipeline.named_steps['classifier'].coef_[0])
        importance_column = 'Coefficient'
    elif classifier_type == 'logreg':
        # For logistic regression, extract coefficients
        importances = np.abs(final_pipeline.named_steps['classifier'].coef_[0])
        importance_column = 'Coefficient'
    else:
        print(f"Feature importance not available for {classifier_type}")
        return

    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({'Feature': selected_features, importance_column: importances})

    # Add category information if meta_cols is provided
    if meta_cols is not None:
        # Create category labels based on feature indices
        categories = []
        for i, feature in enumerate(selected_features):
            if i < meta_cols:
                categories.append('Metadata')
            else:
                categories.append('Proteomics')

        importance_df['Category'] = categories

        # Sort by importance within each category
        importance_df = importance_df.sort_values([importance_column], ascending=False)
        importance_df = importance_df[importance_df[importance_column] > 0.001] # Filter out very low values

        # Create grouped plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Feature Importance Analysis - {type_translation[classifier_type]}', fontsize=20)

        # Plot top 20 overall features
        ax1 = axes[0, 0]
        top_features = importance_df.head(20)
        bars = sns.barplot(x=importance_column, y='Feature', hue='Category', data=top_features,
                           palette=get_palette("2"), ax=ax1, saturation=1.0)
        ax1.set_title('Top 20 Features Overall', fontsize=16)
        ax1.set_xlabel(f'{importance_column}', fontsize=14)
        ax1.set_ylabel('Feature', fontsize=14)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.legend(loc='lower right')
        # Add importances as text
        for idx, value in enumerate(importance_df.head(20)[importance_column]):
            text = ax1.text(
                x=0.03*(top_features[importance_column].max()),
                y=idx,
                s=f"{value:.2f}",
                va='center',
                ha='left',
                fontsize=8,
                fontweight = 'bold',
                color='black',
            )

        # Plot top 10 metadata features
        ax2 = axes[0, 1]
        metadata_features = importance_df[importance_df['Category'] == 'Metadata'].head(10)
        if not metadata_features.empty:
            sns.barplot(x=importance_column, y='Feature', data=metadata_features, color=get_palette("2")[1], ax=ax2,
                        saturation=1.0)
            ax2.set_title('Top 10 Metadata Features', fontsize=16)
            ax2.set_xlabel(f'{importance_column}', fontsize=14)
            ax2.set_ylabel('Feature', fontsize=14)
            ax2.tick_params(axis='x', labelsize=10)
            ax2.tick_params(axis='y', labelsize=10)
            # Add importances as text
            for idx, value in enumerate(metadata_features.head(10)[importance_column]):
                text = ax2.text(
                    x= 0.01*(metadata_features[importance_column].max()),
                    y=idx,
                    s=f"{value:.2f}",
                    va='center',
                    ha='left',
                    fontsize=10,
                    fontweight='bold',
                    color='black',
                )

        else:
            ax2.text(0.5, 0.5, 'No Metadata Features', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Top 10 Metadata Features', fontsize=16)

        # Plot top 10 Proteomics features
        ax3 = axes[1, 0]
        proteomics_features = importance_df[importance_df['Category'] == 'Proteomics'].head(10)
        if not proteomics_features.empty:
            sns.barplot(x=importance_column, y='Feature', data=proteomics_features,
                        color=get_palette("2")[0], ax=ax3, saturation=1.0)
            ax3.set_title('Top 10 Proteomics Features', fontsize=16)
            ax3.set_xlabel(f'{importance_column}', fontsize=14)
            ax3.set_ylabel('Feature', fontsize=14)
            ax3.set_ylabel('Feature', fontsize=14)
            ax3.tick_params(axis='x', labelsize=10)
            ax3.tick_params(axis='y', labelsize=10)
            # Add importances as text
            for idx, value in enumerate(proteomics_features.head(10)[importance_column]):
                text = ax3.text(
                    x=0.04*(proteomics_features[importance_column].max()),
                    y=idx,
                    s=f"{value:.2f}",
                    va='center',
                    ha='left',
                    fontsize=10,
                    fontweight='bold',
                    color='black',
                )
        else:
            ax3.text(0.5, 0.5, 'No Proteomics Features', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Top 10 Proteomics Features', fontsize=16)

        # Plot category summary (total importance by category)
        ax4 = axes[1, 1]
        category_summary = importance_df.groupby('Category')[importance_column].agg(['sum', 'mean', 'count'])
        category_summary = category_summary.reset_index()

        # Create a summary bar plot
        x_pos = range(len(category_summary))
        bars = ax4.bar(x_pos, category_summary['sum'], color=[get_palette("2")[1],get_palette("2")[0]])
        ax4.set_title('Total Importance by Category', fontsize=16)
        ax4.set_xlabel('Category', fontsize=14)
        ax4.set_ylabel(f'Total {importance_column}', fontsize=14)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(category_summary['Category'])
        ax4.tick_params(axis='x', labelsize=10)
        ax4.tick_params(axis='y', labelsize=10)

        # Custom font (for some reason default styles aren't applied) #IMPROVE
        font_path = "Merriweather_24pt-Medium.ttf"
        fm.fontManager.addfont(font_path)
        custom_font = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = custom_font

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}\n({category_summary.iloc[i]["count"]} features)',
                     ha='center', va='bottom', fontproperties=custom_font)

        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/feature_importance_grouped.png")
        plt.close()

        # # Print category summaries
        # print(f"\nFeature importance summary by category:\n")
        # for category in ['Metadata', 'Proteomics']:
        #     cat_data = importance_df[importance_df['Category'] == category]
        #     if not cat_data.empty:
        #         print(f"\n{category} Features:")
        #         print(f"  Total Features: {len(cat_data)}")
        #         print(f"  Total {importance_column}: {cat_data[importance_column].sum():.4f}")
        #         print(f"  Top 5 {category} Features:")
        #         print(cat_data.head(5)[['Feature', importance_column]].to_string(index=False))

    else:
        # Original single plot if no grouping
        importance_df = importance_df.sort_values(importance_column, ascending=False)

        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x=importance_column, y='Feature', data=importance_df.head(20),
                         palette='viridis', saturation=1.0)

        if classifier_type in ['rf', 'xgb', 'gb']:
            ax.set_title(f'Top 20 Feature Importances - {classifier_type.upper()}', fontsize=16)
        else:
            ax.set_title(f'Top 20 Feature Coefficients - {classifier_type.upper()}', fontsize=16)

        ax.set_xlabel(f'{importance_column}', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Save the full feature importance DataFrame as CSV
    if classifier_type in ['rf', 'xgb', 'gb']:
        csv_filename = f"{data_dir}/feature_importances.csv"
    else:
        csv_filename = f"{data_dir}/feature_coefficients.csv"
    importance_df.to_csv(csv_filename, index=False)

    print(f"\nTop 10 most important features overall:") #TODO think metadata colum is including extra features, but might be a data misconfiguration issue
    if meta_cols is not None:
        print(importance_df.head(10)[['Feature', importance_column, 'Category']].to_string(index=False))
    else:
        print(importance_df.head(10)[['Feature', importance_column]].to_string(index=False))

# Plot calibration curve
def plot_calibration_curve(y_proba, y_test, classifier_type, graphs_dir):
    try:
        set_graph_style()

        # Compute calibration curve and Brier score
        brier_score = brier_score_loss(y_test, y_proba)


        # Plot calibration curve
        fig, ax = plt.subplots()
        CalibrationDisplay.from_predictions(
            y_test,
            y_proba,
            n_bins=10,
            strategy='quantile',
            ax=ax,
            name=f"{classifier_type} (Brier: {brier_score:.3f})"
        )
        set_graph_style()
        ax.set_title(f"Calibration Curve")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(loc='upper left')
        plt.savefig(f"{graphs_dir}/calibration_curve.png") # WARNING this overwrites TML when running NN! save to unique folders/names
        plt.close(fig)

        # Log Brier score
        mlflow.log_metric("brier_score", brier_score)

    except Exception as e:
        print(f"Error generating calibration curve: {str(e)}")

def plot_decision_distribution(y_proba, y_test, graphs_dir):
    fig, ax = plt.subplots()
    sns.histplot(
        data=pd.DataFrame({
            'Probability': y_proba,
            'Actual': y_test.replace({0: 'O2 not required', 1: 'O2 required'})
        }),
        x='Probability',
        hue='Actual',
        element='step',
        stat='density',
        common_norm=False,
        bins=10,
        palette = get_palette("2r"),
        ax=ax
    )
    ax.set_title('Predicted Probability Distribution')
    ax.set_xlabel('Predicted Probability of Requiring O2')
    ax.set_ylabel('Density')
    legend = ax.get_legend()
    legend.set_title("Ground truth", prop={'size': 10})
    for text in legend.get_texts():
        text.set_fontsize(8)
    plt.savefig(f'{graphs_dir}/prediction_distribution.png')

# Plot decision tree for tree-based models
def plot_decision_tree(classifier_type, final_pipeline, retained_features, class_names, data_dir, graphs_dir): # IMPROVE - use better styling (my set_styles doesn't seem to apply either). Will tweak if publishing the trees
    ### Plot decision tree for tree-based models #TODO: Basic version - when a final best model is obtained this graph can be fine tuned if useful for the final report
    if classifier_type in ['rf', 'gb', 'xgb']: #TODO untested for gb - and could use refinement of outputs
        try:
            # Extract classifier from pipeline
            clf = final_pipeline.named_steps['classifier']

            # Extract decision tree according to each model type
            if classifier_type == 'rf':
                estimator = clf.estimators_[0]
                plt.figure()
                plot_tree(estimator,
                          feature_names=retained_features,
                          class_names=class_names,
                          filled=True,
                          rounded=True,
                          max_depth=4, #  Limit depth for readability - but ideally expand this for the final graph
                          precision=2,)
                plt.title("Random Forest - First Tree")
                plt.savefig(f"{graphs_dir}/first_decision_tree.png")
                plt.close()

                # Also export text representation
                tree_rules = export_text(estimator,
                                         feature_names=list(retained_features),
                                         max_depth=4)
                with open(f"{data_dir}/first_tree_rules.txt", "w") as f:
                    f.write(tree_rules)

            elif classifier_type == 'gb': #todo check if same sanitising/list format is required as in xgb; if so group in an elif before handling each separately
                plt.figure(figsize=(25, 15))
                estimator = clf.estimators_[0, 0]
                plot_tree(estimator,
                          feature_names=retained_features,
                          class_names=class_names,
                          filled=True,
                          rounded=True,
                          max_depth=4,
                          precision=2,
                          )
                plt.title("Gradient Boosting - First Tree")
                plt.savefig(f"{graphs_dir}/first_decision_tree.png", dpi=300, bbox_inches='tight')
                plt.close()

            elif classifier_type == 'xgb':
                # Sanitise feature names/convert to list for XGB - Replace non-alphanumeric with underscores
                feature_names = list(retained_features.astype(str))
                sanitised_feature_names = [re.sub(r'[^a-zA-Z0-9_]', '_', str(f)) for f in feature_names]
                feature_names = sanitised_feature_names

                # Ensure names are unique after sanitization #todo make cleaner
                seen = {}
                for i, name in enumerate(feature_names):
                    if feature_names.count(name) > 1:
                        feature_names[i] = f"{name}_1"
                        print(f"{name} occurs multiple times; appended _{i}.")

                # Set sanitised feature names on the booster
                booster = clf.get_booster()
                booster.feature_names = feature_names
                # Plot
                plt.figure(figsize=(25, 15))
                xgb_plot_tree(clf, tree_idx=0, rankdir='LR')
                plt.title("XGBoost - First Tree")
                plt.savefig(f"{graphs_dir}/first_decision_tree.png", dpi=300, bbox_inches='tight')
                plt.close()

                # Check size of trees
                show_size = False
                if show_size:
                    for i in range(50):  # Adjust range as needed
                        dot = to_graphviz(clf, tree_idx=i)
                        tree_str = dot.source
                        print(f"Tree {i}: {'leaf=' in tree_str} | size: {len(tree_str)}")

                # Save interactive version # TODO look into supertree
                dot = to_graphviz(clf, with_stats=True, feature_names=feature_names)
                dot.render(filename='xgb_tree_graph_1', format='svg') # Saves to svg - zoomable in browser

            else:
                print(f"Decision tree plotting not supported for {classifier_type}")

        except Exception as e:
            print(f"Error plotting decision tree: {str(e)}")

# Plot precision-recall curve
def plot_precision_recall(y_proba, y_test, graphs_dir):
    try:
        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        average_precision = average_precision_score(y_test, y_proba)

        # Plot the curve
        fig, ax = plt.subplots()
        PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision).plot(ax=ax)
        ax.set_title(f"Precision-Recall Curve (AP = {average_precision:.2f})")

        # Save and log
        plt.savefig(f"{graphs_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        mlflow.log_figure(fig, "precision_recall_curve.png")
        plt.close(fig)

        # Log the average precision metric
        mlflow.log_metric("average_precision", average_precision)
        print(f"Precision-Recall curve saved (AP: {average_precision:.3f})")

    except Exception as e:
        print(f"Error generating Precision-Recall curve: {str(e)}")

# Plot PCA on the final predictions derived from test data
def plot_pca_test_unprocessed(X_test, y_test, graphs_dir):
    # Reset y index to avoid errors # IMPROVE for this and other PCA, check index resetting or rewrite to avoid
    y_test = y_test.reset_index(drop=True)

    # Standardize the selected data
    scaler = StandardScaler()  # WARNING Avoiding using the same one as in the pipeline to prevent data leakage - not sure if it's an issue but an error occurs when using the pipeline ver in the second PCA above
    X_scaled = scaler.fit_transform(X_test)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df = pc_df.reset_index(drop=True)  # Ensure index alignment

    plt.figure()

    # Create boolean masks
    class_0_mask = (y_test == 0)
    class_1_mask = (y_test == 1)

    # Plot using the aligned indices
    plt.scatter(pc_df.loc[class_0_mask, 'PC1'],
                pc_df.loc[class_0_mask, 'PC2'],
                c=get_palette("2")[1], alpha=0.7, edgecolors=get_palette("2")[1],  label='O2 not required')
    plt.scatter(pc_df.loc[class_1_mask, 'PC1'],
                pc_df.loc[class_1_mask, 'PC2'],
                c=get_palette("2")[0], alpha=0.7, edgecolors=get_palette("2")[0], label='O2 required')

    # Add explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title('PCA of test data - ground truth')
    plt.legend()

    # Save and show
    plt.savefig(f"{graphs_dir}/pca_test_before_processing.png")
    plt.close()

# Plot PCA on the final predictions derived from test data
def plot_pca_predicted(X_test, selected_features, y_test, graphs_dir, y_pred):
    # Select the features determined by feature selection
    X_selected = X_test[selected_features]

    # Reset y index to avoid errors # IMPROVE for this and other PCA, check index resetting or rewrite to avoid
    y_test = y_test.reset_index(drop=True)

    # Standardize the selected data
    scaler = StandardScaler()  # WARNING Avoiding using the same one as in the pipeline to prevent data leakage - not sure if it's an issue but an error occurs when using the pipeline ver in the second PCA above
    X_scaled = scaler.fit_transform(X_selected)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df = pc_df.reset_index(drop=True)  # Ensure index alignment

    plt.figure()

    # Create boolean masks
    class_0_mask = (y_test == 0)
    class_1_mask = (y_test == 1)

    # Plot using the aligned indices
    plt.scatter(pc_df.loc[class_0_mask, 'PC1'],
                pc_df.loc[class_0_mask, 'PC2'],
                c=get_palette("2")[1], alpha=0.7, edgecolors=get_palette("2")[1], label='O2 not required')
    plt.scatter(pc_df.loc[class_1_mask, 'PC1'],
                pc_df.loc[class_1_mask, 'PC2'],
                c=get_palette("2")[0], alpha=0.7, edgecolors=get_palette("2")[0], label='O2 required')

    # Add explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title('PCA of test data - ground truth')
    plt.legend()

    # Save and show
    plt.savefig(f"{graphs_dir}/pca_test_before_prediction.png")
    plt.close()

    ### Create a second PCA colour coded by TP/FP/TN/FN
    # Create masks for each outcome type
    mask_tn = (y_test == 0) & (y_pred == 0)  # True Negative
    mask_fp = (y_test == 0) & (y_pred == 1)  # False Positive
    mask_fn = (y_test == 1) & (y_pred == 0)  # False Negative
    mask_tp = (y_test == 1) & (y_pred == 1)  # True Positive

    # Create plot with distinct colors # TODO: Could switch colours for FN FP if more legible that way
    plt.figure()
    plt.scatter(pc_df.loc[mask_tn, 'PC1'], pc_df.loc[mask_tn, 'PC2'],
                c=get_palette("ld")[2], alpha=0.8, edgecolors=get_palette("ld")[2], label='True Negative')  # Blue = negative (doesn't need O2)
    plt.scatter(pc_df.loc[mask_fp, 'PC1'], pc_df.loc[mask_fp, 'PC2'],
                c=get_palette("ld")[3], alpha=0.8, edgecolors=get_palette("ld")[3], label='False Positive')  # Dark blue = Predicted positive but should be negative
    plt.scatter(pc_df.loc[mask_fn, 'PC1'], pc_df.loc[mask_fn, 'PC2'],
                c=get_palette("ld")[1], alpha=0.8, edgecolors=get_palette("ld")[1], label='False Negative')  # Dark red = Predicted negative but should be positive
    plt.scatter(pc_df.loc[mask_tp, 'PC1'], pc_df.loc[mask_tp, 'PC2'],
                c=get_palette("ld")[0], alpha=0.8, edgecolors=get_palette("ld")[0], label='True Positive')  # Red = Postive (needs O2)

    # Add labels and title
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title('PCA of test data (Prediction outcomes)')
    plt.legend()

    # Add count annotations
    counts = {
        'TN': mask_tn.sum(),
        'FP': mask_fp.sum(),
        'FN': mask_fn.sum(),
        'TP': mask_tp.sum()
    }
    plt.figtext(0.15, 0.01,
                f"TN: {counts['TN']} | FP: {counts['FP']} | FN: {counts['FN']} | TP: {counts['TP']}",
                ha="center", fontsize=12,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    # Save and log
    plt.savefig(f"{graphs_dir}/pca_test_prediction_outcomes.png")
    plt.close()

### NEURAL NETWORKS ####################################################################################################
# Create function for grouped shap
def grouped_shap(shap_vals, features, groups):
    revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped

### OTHER ##############################################################################################################
def plot_metrics_heatmap(metrics, output_dir, string):
    plt.figure(figsize=(9, (9/6)*len(metrics)))
    ax = sns.heatmap(
        metrics,
        annot=True,
        fmt=".2f",
        cmap='Greens',
        linewidths=.5,
        cbar=False
    )

    # Customize labels
    ax.set_title(f'{string} Evaluation Metrics', fontsize=14)
    ax.set_xlabel('Evaluation metrics', fontsize=12)
    ax.set_ylabel('Model pipeline', fontsize=12)
    # Save figure
    plt.savefig(f"{output_dir}/{string}_heatmap.png")

########################################################################################################################
plt.close("all")