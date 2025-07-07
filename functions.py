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

### MODEL BUILDING FUNCTIONS ###########################################################################################
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




