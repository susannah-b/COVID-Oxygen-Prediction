import socket
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree, export_text
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from xgboost import to_graphviz
from xgboost import plot_tree as xgb_plot_tree
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import re
import missingno as msno

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
                legend=False)
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
    fig_copy.savefig(f'{graphs_dir}/Missingness_All-data_before_filtering_{dataset}.png', bbox_inches='tight')
    # Plot missingness of the metadata
    fig = msno.matrix(merged.iloc[:, 0:meta_cols])
    fig_copy = fig.get_figure()
    fig_copy.savefig(f'{graphs_dir}/Missingness_All-metadata_before_filtering_{dataset}.png', bbox_inches='tight')
    # Plot missingness of the quant data
    fig = msno.matrix(merged.iloc[:, meta_cols:])
    fig_copy = fig.get_figure()
    fig_copy.savefig(f'{graphs_dir}/Missingness_All-quantdata_before_filtering_{dataset}.png', bbox_inches='tight')

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
    return merged

# Determine remaining metadata columns and plot with missingno # IMPROVE - think same logic is used elsewhere - could apply function
def remaining_meta(meta_columns, merged, sample_inves_7, graphs_dir):
    # Determine remaining columns
    existing_columns = merged.columns.tolist()
    # Work backwards through the list to find the first one still present
    meta_cols = 0  # Initialise
    for col in reversed(meta_columns):
        if col in existing_columns:
            if sample_inves_7:
                print("\nRemaining metadata columns:")
                print(merged.columns.get_loc(
                    col) + 1)  # +1 for 1-based indexing conversion / allows for splicing where the first number is inclusive and the second exclusive
            meta_cols = merged.columns.get_loc(col) + 1
            break
    else:
        if sample_inves_7:
            print("All metadata columns were removed by the missingness filter.")
        meta_cols = 0

    # Plot missingness after filtering (full dataset)
    fig = msno.matrix(merged)
    fig_copy = fig.get_figure()
    fig_copy.savefig(f'{graphs_dir}/Missingness_All-data_after_filtering.png', bbox_inches='tight')
    # Plot missingness of the metadata
    fig = msno.matrix(merged.iloc[:, 0:meta_cols])
    fig_copy = fig.get_figure()
    fig_copy.savefig(f'{graphs_dir}/Missingness_All-metadata_after_filtering.png', bbox_inches='tight')
    # Plot missingness of the quant data
    fig = msno.matrix(merged.iloc[:, meta_cols:])  # Note: This is currently 546-27=519 MS data columns
    fig_copy = fig.get_figure()
    fig_copy.savefig(f'{graphs_dir}/Missingness_All-quantdata_after_filtering.png', bbox_inches='tight')

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
    plt.savefig(f'{graphs_dir}/class_distribution_{dataset}.png', dpi=200)

### MODEL_BUILDING.PY FUNCTIONS ########################################################################################
# Detect metadata columns in the dataset
def count_meta(dataset, name, metadata_features, drop, show_detail):
    matched = False # Initialise
    existing_columns = dataset.columns.tolist()
    col_number = 0 # Initialise
    for col in reversed(metadata_features):
        if col in existing_columns:
            matched = True
            if show_detail:
                print(f"\nMetadata columns in {name}:")
                print(dataset.columns.get_loc(col) + 1) # +1 for 1-based indexing conversion / allows for splicing where the first number is inclusive and the second exclusive
            col_number = dataset.columns.get_loc(col) + 1
            break
    if not matched:
        if show_detail:
            print("No metadata columns found.")
    if drop: # Drop the metadata if bool is true
        dataset = dataset.iloc[:, col_number:]
        col_number = 0 # Now removed all metadata so count is 0
        print(f"Metadata was dropped from {name}; if unintended, disable drop_metadata in the script.")
    return col_number,dataset

# Basic model training function to get some initial scores and decide which model to proceed with
def basic_train(model, X_train, y_train, identifier, scores_dict, feature_selectors, feature_selection, threshold):
    # Iterate over feature selectors
    model_results = {}
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
            # 10-fold cross validation for F1 score and accuracy
            f1_val = cross_val_score(pipe, X_train, y_train, scoring='f1', cv=StratifiedKFold(10, shuffle=True, random_state=42))
            accuracy_val = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=StratifiedKFold(10, shuffle=True, random_state=42))

            # Fit the pipeline on the training data
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_train)
            f1_train = f1_score(y_train, y_pred)
            accuracy_train = accuracy_score(y_train, y_pred)

            #model_results[fs_name] = [identifier, fs_name, f1_train, f1_val.mean(), accuracy_train, accuracy_val.mean()]
            model_results[fs_name] = [identifier, fs_name, accuracy_train, accuracy_val.mean(), f1_train, f1_val.mean(), ]
            print(f"Training of {identifier} using {fs_name} complete.")
        except Exception as e:
            print(f"Error training {identifier} with {fs_name}: {str(e)}")
            model_results[identifier] = [identifier, None, None, None, None, None]
    # Print results from best feature selection methods
    model_results_df = pd.DataFrame.from_dict(model_results,
                           orient='index',
                           columns=['Model', 'Selector', 'Train Accuracy', 'CV Accuracy', 'Train F1',
                                    'Test F1']).sort_values(by=['Test F1'], ascending=False)
    print(f"Metrics from {identifier} experimentation:")
    print(model_results_df, "\n")
    # Take the top result unless empty
    if model_results_df.empty:
        scores_dict[identifier] = [identifier, None, None, None, None, None]
        print(f"All feature selection methods failed for {identifier}.")
    else:
        scores_dict[identifier] = model_results_df.iloc[0].to_list()
    print(f"Finished training {identifier}")

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
# Plot PCA on the combined dataset - i.e. original data after feature selection
def pca_original(X_train, X_test, selected_features, y_train, y_test, graphs_dir):
    # Combine X/y train and test for full dataset visualization
    X_full = pd.concat([X_train, X_test])
    X_selected = X_full[selected_features]
    y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

    # Standardize the selected data
    scaler = StandardScaler()  # WARNING Avoiding using the same one as in the pipeline to prevent data leakage - not sure if it's an issue but it will error when called later if used here due to different number of features
    X_scaled = scaler.fit_transform(X_selected)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df = pc_df.reset_index(drop=True)  # Ensure index alignment

    plt.figure(figsize=(14, 10))

    # Create boolean masks
    class_0_mask = (y_full == 0)
    class_1_mask = (y_full == 1)

    # Plot using the aligned indices
    plt.scatter(pc_df.loc[class_0_mask, 'PC1'],
                pc_df.loc[class_0_mask, 'PC2'],
                c='#088BDD', alpha=0.7, label='Does not require O2')
    plt.scatter(pc_df.loc[class_1_mask, 'PC1'],
                pc_df.loc[class_1_mask, 'PC2'],
                c='red', alpha=0.7, label='Requires O2')

    # Add explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title('PCA After Feature Selection')
    plt.grid(alpha=0.3)
    plt.legend()

    # Save and show
    plt.savefig(f"{graphs_dir}/pca_full_dataset_after_FS.png", dpi=150, bbox_inches='tight')
    plt.close()

# Plot learning curve
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
    ax.set_ylabel("F1 Score")
    ax.set_title("Learning Curve")

    # Save the plot
    plt.savefig(f"{graphs_dir}/learning_curve.png", dpi=200, bbox_inches='tight')

    # Log the figure as an MLflow artifact
    mlflow.log_figure(fig, f"{graphs_dir}/learning_curve.png")
    plt.close(fig)

# Plot ROC_AUC curve
def plot_roc_auc(final_pipeline, X_test, y_test, graphs_dir):
    # Get prediction probabilities
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot and log
    fig, ax = plt.subplots()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
    mlflow.log_figure(fig, f"{graphs_dir}/roc_curve.png")

    # Save the plot
    plt.savefig(f"{graphs_dir}/roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log the AUC metric explicitly
    mlflow.log_metric("test_auc", roc_auc)

# Plot feature importance
def plot_feature_importance(classifier_type, final_pipeline, selected_features, graphs_dir, data_dir, best_params,
                            X_test, y_test):
    # Extract feature importances based on the classifier type
    if classifier_type in ['rf', 'xgb', 'gb']:
        # These classifiers have feature_importances_ attribute
        importances = final_pipeline.named_steps['classifier'].feature_importances_

        # Create DataFrame for easier plotting with seaborn
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Importance', y='Feature', hue='Feature', legend=False, data=importance_df.head(20),
                         palette='viridis')
        ax.set_title(f'Top 20 Feature Importances - {classifier_type.upper()}', fontsize=16)
        ax.set_xlabel('Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), f"{graphs_dir}/feature_importance.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv(f"{data_dir}/feature_importances.csv", index=False)

        print(f"\nTop 10 most important features:")
        print(importance_df.head(10))

    elif classifier_type == 'svm' and best_params.get('kernel') == 'linear':
        # For linear SVM, we can extract coefficients
        coefficients = np.abs(final_pipeline.named_steps['classifier'].coef_[0])

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Coefficient', y='Feature', data=importance_df.head(20), palette='viridis')
        ax.set_title('Top 20 Feature Coefficients - Linear SVM', fontsize=16)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), f"{graphs_dir}/feature_coefficients.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv(f"{data_dir}/feature_coefficients.csv", index=False)

        print(f"\nTop 10 most important features (by coefficient magnitude):")
        print(importance_df.head(10))

    elif classifier_type == 'logreg':
        # For logistic regression, extract coefficients
        coefficients = np.abs(final_pipeline.named_steps['classifier'].coef_[0])

        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Coefficient', y='Feature', data=importance_df.head(20), palette='viridis')
        ax.set_title('Top 20 Feature Coefficients - Logistic Regression', fontsize=16)
        ax.set_xlabel('Absolute Coefficient Value', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{graphs_dir}/feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), f"{graphs_dir}/feature_coefficients.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv(f"{data_dir}/feature_coefficients.csv", index=False)

        print(f"\nTop 10 most important features (by coefficient magnitude):")
        print(importance_df.head(10))

    else:
        # For other models where direct feature importance is not available use permutation importance as an alternative
        print("\nCalculating permutation importance for features as an alternative to feature importance.")
        try:  # WARNING - currently fails (for SVC, Ada, KNN) - if any do perform highly then rectify
            # Calculate permutation importance
            perm_importance = permutation_importance(
                final_pipeline,
                X_test,
                y_test,
                n_repeats=30,
                random_state=42,
                scoring='f1',
            )

            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=False)

            # Plot with seaborn
            plt.figure(figsize=(12, 8))
            sns.set_style("whitegrid")
            ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
            ax.set_title(f'Top 20 Permutation Feature Importances - {classifier_type.upper()}', fontsize=16)
            ax.set_xlabel('Mean Importance', fontsize=14)
            ax.set_ylabel('Feature', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{graphs_dir}/permutation_importance.png", dpi=300, bbox_inches='tight')

            # Log the figure to MLflow
            mlflow.log_figure(plt.gcf(), f"{graphs_dir}/permutation_importance.png")
            plt.close()

            # Also save the full feature importance DataFrame as CSV
            importance_df.to_csv(f"{data_dir}/permutation_importances.csv", index=False)

            print(f"\nTop 10 most important features (by permutation importance):")
            print(importance_df.head(10))
        except Exception as e:
            print(f"Unable to calculate feature importance.\n{e}")

# Plot calibration curve
def plot_calibration_curve(final_pipeline, X_test, y_test, classifier_type, graphs_dir):
    try:
        # Check if model supports probability estimates
        if hasattr(final_pipeline, 'predict_proba'):
            # Get predicted probabilities for the positive class
            prob_pos = final_pipeline.predict_proba(X_test)[:, 1]

            # Compute calibration curve and Brier score
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, prob_pos, n_bins=10, strategy='quantile'
            )
            brier_score = brier_score_loss(y_test, prob_pos)

            # Plot calibration curve
            fig, ax = plt.subplots(figsize=(10, 8))
            CalibrationDisplay.from_predictions(
                y_test,
                prob_pos,
                n_bins=10,
                strategy='quantile',
                ax=ax,
                name=f"{classifier_type} (Brier: {brier_score:.3f})"
            )
            ax.set_title(f"Calibration Curve")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.grid(True)
            plt.savefig(f"{graphs_dir}/calibration_curve.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Log Brier score
            mlflow.log_metric("brier_score", brier_score)

        # For models without predict_proba but with decision_function (like SVM)
        elif hasattr(final_pipeline, 'decision_function'):
            # Get decision scores and scale to [0,1]
            decision_scores = final_pipeline.decision_function(X_test)
            prob_pos = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())

            # Compute calibration curve and Brier score
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, prob_pos, n_bins=10, strategy='quantile'
            )
            brier_score = brier_score_loss(y_test, prob_pos)

            # Plot calibration curve
            fig, ax = plt.subplots(figsize=(10, 8))
            CalibrationDisplay.from_predictions(
                y_test,
                prob_pos,
                n_bins=10,
                strategy='quantile',
                ax=ax,
                name=f"{classifier_type} (scaled scores, Brier: {brier_score:.3f})"
            )
            ax.set_title(f"Calibration Curve (Scaled Decision Scores)")
            ax.set_xlabel("Mean Scaled Decision Score")
            ax.set_ylabel("Fraction of Positives")
            ax.grid(True)
            plt.savefig(f"{graphs_dir}/calibration_curve.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Log Brier score
            mlflow.log_metric("brier_score", brier_score)
            print(f"Calibration curve (scaled scores) and Brier score ({brier_score:.3f}) saved successfully.")

        else:
            print("Model doesn't support probability estimates or decision scores - skipping calibration curve")

    except Exception as e:
        print(f"Error generating calibration curve: {str(e)}")

# Plot decision tree for tree-based models
def plot_decision_tree(classifier_type, final_pipeline, X_train, class_names, data_dir, graphs_dir):
    ### Plot decision tree for tree-based models #TODO: Basic version - when a final best model is obtained this graph can be fine tuned if useful for the final report
    if classifier_type in ['rf', 'gb', 'xgb']: #TODO untested for gb - and could use refinement of outputs
        try:
            # Extract classifier from pipeline
            clf = final_pipeline.named_steps['classifier']

            # Get feature names after feature selection
            selector = final_pipeline.named_steps['feature_selector']
            if selector != 'passthrough':
                if hasattr(selector, 'get_support'):
                    mask = selector.get_support()
                    feature_names = X_train.columns[mask]
                elif hasattr(selector, 'support_'):
                    feature_names = X_train.columns[selector.support_]
            else:
                feature_names = X_train.columns

            # Extract decision tree according to each model type
            if classifier_type == 'rf':
                estimator = clf.estimators_[0]
                plt.figure(figsize=(25, 15))
                plot_tree(estimator,
                          feature_names=feature_names,
                          class_names=class_names,
                          filled=True,
                          rounded=True,
                          max_depth=4) # Limit depth for readability - but ideally expand this for the final graph
                plt.title("Random Forest - First Tree")
                plt.savefig(f"{graphs_dir}/decision_tree_1.png", dpi=200, bbox_inches='tight')
                plt.close()

                # Also export text representation
                tree_rules = export_text(estimator,
                                         feature_names=list(feature_names),
                                         max_depth=4)
                with open(f"{data_dir}/tree_rules_1.txt", "w") as f:
                    f.write(tree_rules)

            elif classifier_type == 'gb': #todo check if same sanitising/list format is required as in xgb; if so group in an elif before handling each separately
                plt.figure(figsize=(25, 15))
                estimator = clf.estimators_[0, 0]
                plot_tree(estimator,
                          feature_names=feature_names,
                          class_names=class_names,
                          filled=True,
                          rounded=True,
                          max_depth=4)
                plt.title("Gradient Boosting - First Tree")
                plt.savefig(f"{graphs_dir}/decision_tree_1.png", dpi=200, bbox_inches='tight')
                plt.close()

            elif classifier_type == 'xgb':
                # Sanitise feature names/convert to list for XGB - Replace non-alphanumeric with underscores
                feature_names = list(feature_names.astype(str))
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
                plt.savefig(f"{graphs_dir}/decision_tree_1.png", dpi=200, bbox_inches='tight')
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
def plot_precision_recall(final_pipeline, X_test, y_test, graphs_dir):
    try:
        # Get prediction probabilities
        if hasattr(final_pipeline, 'predict_proba'):
            y_proba = final_pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(final_pipeline, 'decision_function'): # For models that use decision_function instead of predict_proba
            decision_scores = final_pipeline.decision_function(X_test)
            y_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
        else:
            raise RuntimeError("Model doesn't support probability estimates")

        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        average_precision = average_precision_score(y_test, y_proba)

        # Plot the curve
        fig, ax = plt.subplots()
        PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision).plot(ax=ax)
        ax.set_title(f"Precision-Recall Curve (AP = {average_precision:.2f})")

        # Save and log
        plt.savefig(f"{graphs_dir}/precision_recall_curve.png", dpi=150, bbox_inches='tight')
        mlflow.log_figure(fig, "precision_recall_curve.png")
        plt.close(fig)

        # Log the average precision metric
        mlflow.log_metric("average_precision", average_precision)
        print(f"Precision-Recall curve saved (AP: {average_precision:.3f})")

    except Exception as e:
        print(f"Error generating Precision-Recall curve: {str(e)}")

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

    plt.figure(figsize=(7, 5))

    # Create boolean masks
    class_0_mask = (y_test == 0)
    class_1_mask = (y_test == 1)

    # Plot using the aligned indices
    plt.scatter(pc_df.loc[class_0_mask, 'PC1'],
                pc_df.loc[class_0_mask, 'PC2'],
                c='#088BDD', alpha=0.7, label='Does not require O2')
    plt.scatter(pc_df.loc[class_1_mask, 'PC1'],
                pc_df.loc[class_1_mask, 'PC2'],
                c='red', alpha=0.7, label='Requires O2')

    # Add explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title('PCA of test data - ground truth')
    plt.grid(alpha=0.3)
    plt.legend()

    # Save and show
    plt.savefig(f"{graphs_dir}/pca_test_before_prediction.png", dpi=200, bbox_inches='tight')
    plt.close()

    ### Create a second PCA colour coded by TP/FP/TN/FN
    # Create masks for each outcome type
    mask_tn = (y_test == 0) & (y_pred == 0)  # True Negative
    mask_fp = (y_test == 0) & (y_pred == 1)  # False Positive
    mask_fn = (y_test == 1) & (y_pred == 0)  # False Negative
    mask_tp = (y_test == 1) & (y_pred == 1)  # True Positive

    # Create plot with distinct colors # TODO: Could switch colours for FN FP if more legible that way
    plt.figure(figsize=(7, 5))
    plt.scatter(pc_df.loc[mask_tn, 'PC1'], pc_df.loc[mask_tn, 'PC2'],
                c='#088BDD', alpha=0.7, label='True Negative')  # Blue = negative (doesn't need O2)
    plt.scatter(pc_df.loc[mask_fp, 'PC1'], pc_df.loc[mask_fp, 'PC2'],
                c='#084769', alpha=0.7, label='False Positive')  # Dark blue = Predicted positive but should be negative
    plt.scatter(pc_df.loc[mask_fn, 'PC1'], pc_df.loc[mask_fn, 'PC2'],
                c='#780000', alpha=0.7, label='False Negative')  # Dark red = Predicted negative but should be positive
    plt.scatter(pc_df.loc[mask_tp, 'PC1'], pc_df.loc[mask_tp, 'PC2'],
                c='red', alpha=0.7, label='True Positive')  # Red = Postive (needs O2)

    # Add labels and title
    plt.xlabel(f'PC1 ({explained_var[0]:.1f}%)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1f}%)')
    plt.title('PCA of test data (Prediction outcomes)')
    plt.grid(alpha=0.3)
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
    plt.savefig(f"{graphs_dir}/pca_test_prediction_outcomes.png", dpi=200, bbox_inches='tight')
    plt.close()




