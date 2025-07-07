import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import socket
import shutil
import os

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

# Copy folder contents
def copy_contents(source, destination):
    # Make folder for the outputs
    os.makedirs(destination, exist_ok=True)
    for item in source.iterdir():
        dest_path = destination / item.name
        if item.is_dir():
            shutil.copytree(item, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_path)