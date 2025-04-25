import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import learning_curve, LearningCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
# todo clean up at end

# TODO Any other useful metrics to generate? Or adjust what I asses by

# todo do i need to do anything with balancing classes?

### Read in data
# Train
X_path = Path(__file__).parent / "Surrey_X_train.csv"
y_path = Path(__file__).parent / "Surrey_y_train.csv"
X_train = pd.read_csv(X_path, index_col=0)
y_train = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Test
X_path = Path(__file__).parent / "Surrey_X_test.csv"
y_path = Path(__file__).parent / "Surrey_y_test.csv"
X_test = pd.read_csv(X_path, index_col=0)
y_test = pd.read_csv(y_path, index_col=0).squeeze()  # Convert to 1D array
# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# TODO Saw a mention of VIF analysis, do I use that in conjunction with thresholding? Currently only do 0 variance thresholding though

### VARIANCE THRESHOLDING ##############################################################################################
print("Feature count before thresholding:")
print(len(X_train.columns))
#todo rushed through for speed - do properly later
 # todo also just set to 0 variance (which removes none of my data) and got a better result so leaving it at that for now
# Calculate median variance of all features
variances = X_train.var(axis=0)
#threshold = np.median(variances)
#threshold = np.quantile(variances, 0.75) #todo experiment/research good threshold for mass spec
threshold = 0.0

# Creating a preprocessing pipeline to scale and feature select through variance thresholding
preprocessor = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=threshold)),
    ('scaler', StandardScaler())
])
#todo i'm transforming data with a preprocessor outside of the main modeling pipeline, which could lead to data leakage. implement as one pipeline

# Fit on training data and transform test data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Track retained features
scaler = preprocessor.named_steps['scaler']
var_thresh = preprocessor.named_steps['var_thresh']

# Get feature mask after both scaling and thresholding
retained_mask = var_thresh.get_support()
retained_features = X_train.columns[retained_mask]

print("Features after thresholding:", len(retained_features))
print("Retained features:", list(retained_features))

# Step 5: Convert back to DataFrames
X_train = pd.DataFrame(X_train_transformed,
                      columns=retained_features,
                      index=X_train.index)

X_test = pd.DataFrame(X_test_transformed,
                     columns=retained_features,
                     index=X_test.index)

print("Feature count after thresholding:")
print(len(X_train.columns))
print("Retained features:", list(X_train.columns))
# TODO Might be worth exerpimenting with keeping all metadata/doing different thresholds for proteins. or doing smarter methods than variance to pick up on subtle patterns

### DEFINE FEATURE SELECTION PER MODEL #################################################################################
# Define a dictionary mapping classifier types to their optimal feature selectors
# TODO commenting out this version and replacing with another simpler one - needs full research and overhaul anyway
# feature_selectors_dict = { # TODO: AI-gen-ed for quick options. Do research into actual best selection ethods for each model - knn/[and another] had to be replaced with sklearn native
#     'svm': SelectFromModel(LinearSVC(C=0.1, random_state=42, max_iter=1000)), # todo had to change linear svc to regular svc, removed l1 penalty. but will research better FS later and redo all
#     'rf': SelectFromModel(RandomForestClassifier(n_estimators=100,
#                                                  max_depth=5,
#                                                  random_state=42),
#                                                  threshold="median"
#                           ),
#     'logreg': SelectFromModel(LogisticRegression(penalty="elasticnet",
#                                                     solver="saga",
#                                                     l1_ratio=0.5,  # Mix of L1/L2
#                                                     C=0.1,
#                                                     random_state=42
#                                                  )
#                               ),
#     'xgb': RFECV(estimator=XGBClassifier(n_estimators=100, max_depth=3),
#                                         step=1,
#                                         cv=StratifiedKFold(3),
#                                         scoring="f1"
#                                         ),
#     'gb': SelectFromModel(GradientBoostingClassifier(random_state=42)),
#     'knn': RFECV(estimator=RandomForestClassifier(random_state=42),
#                  step=1, cv=StratifiedKFold(5, shuffle=True, random_state=42),
#                  scoring='f1', min_features_to_select=1
#                  ),
#     'ada': RFECV(estimator=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
#         step=1,
#         cv=StratifiedKFold(3),
#         scoring="f1"
#     )
#
# }

feature_selectors_dict = {
    'svm': SelectFromModel(LinearSVC(C=0.1, random_state=42, max_iter=2000)),
    'rf': SelectFromModel(RandomForestClassifier(n_estimators=100,
                                                max_depth=5,
                                                random_state=42),
                          threshold="median"),
    'logreg': SelectFromModel(LogisticRegression(penalty="l1",
                                                solver="saga",
                                                C=0.1,
                                                max_iter=2000,
                                                random_state=42)),
    'xgb': SelectFromModel(XGBClassifier(n_estimators=100,
                                         max_depth=3,
                                         random_state=42)),
    'gb': SelectFromModel(GradientBoostingClassifier(random_state=42)),
    'knn': SelectKBest(f_classif, k=20),  # Using statistical test instead of RFECV
    'ada': SelectFromModel(AdaBoostClassifier(n_estimators=50, random_state=42))
}
# WARNING get error in xgboost and it fails - investigate

### ESTIMATE BEST MODELS WITH BASIC SETTINGS ###########################################################################
# Bool to decide whether to run the basic model training - if already know best candidates from results can skip
basic_training = True #todo might remove and just always do depending on time to generate
# Initialise dict
model_scores = {}

# Basic model training function to get some initial scores and decide which model to proceed with
# IMPROVE add more model types?

if basic_training:
    def basic_train(model, model_type, X_train, y_train, identifier, dict):
        # Create a pipeline with feature selection and classifier - ensures same CV folds/feature selection
        selector = feature_selectors_dict[model_type]
        pipe = Pipeline([
            ('feature_selector', selector),
            ('classifier', model)
        ])
        try:
            # 10-fold cross validation for F1 score and accuracy
            f1_val = cross_val_score(pipe, X_train, y_train, scoring='f1', cv=StratifiedKFold(5, shuffle=True, random_state=42)) # todo is this meant to be 5
            accuracy_val = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=StratifiedKFold(10, shuffle=True, random_state=42))

            # Fit the pipeline on the training data
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_train)
            f1_train = f1_score(y_train, y_pred)
            accuracy_train = accuracy_score(y_train, y_pred)

            dict[identifier] = [identifier, f1_train, f1_val.mean(), accuracy_train, accuracy_val.mean()]
            print(f"Successfully trained {identifier}")
        except Exception as e:
            print(f"Error training {identifier}: {str(e)}")
            dict[identifier] = [identifier, None, None, None, None]

    # Logistic Regression
    log_reg = LogisticRegression(penalty='l1', solver='saga', tol=1e-3) # TODO inrcreased tol from 1e-4, maybe remove this term to reset to default later when I can increase max_iter
    basic_train(log_reg, 'logreg', X_train, y_train, 'Logistic Regression', model_scores)

    # SVM
    svc_clf = SVC()
    basic_train(svc_clf, 'svm', X_train, y_train, 'Support Vector Classifier', model_scores)

    # Random Forest
    rnd_clf = RandomForestClassifier(random_state=42)
    basic_train(rnd_clf, 'rf', X_train, y_train, 'RandomForestClassifier', model_scores)

    # AdaBoost
    dt_clf_ada = DecisionTreeClassifier()
    ada_clf = AdaBoostClassifier(estimator=dt_clf_ada, random_state=42)
    basic_train(ada_clf, 'ada', X_train, y_train, "AdaBoost Classifier", model_scores)

    # GradientBoosting
    gdb_clf = GradientBoostingClassifier(random_state=42, subsample=0.8)
    basic_train(gdb_clf, 'gb', X_train, y_train, "GradientBoosting Classifier", model_scores)

    # XGBoost
    xgb_clf = XGBClassifier(verbosity=0)
    basic_train(xgb_clf, 'xgb', X_train, y_train, "XGBoost Classifier", model_scores)

    # KNN
    knn_clf = KNeighborsClassifier()
    basic_train(knn_clf, 'knn', X_train, y_train, 'K-Nearest Neighbors Classifier', model_scores)

    # Make dataframe of model scores and print results
    scores = pd.DataFrame.from_dict(model_scores, orient='index',
                                    columns=['Model', 'Train F1', 'Test F1', 'Train Accuracy',
                                             'Test Accuracy']).reset_index(drop=True).sort_values(by='Test F1',
                                                                                                   ascending=False)
    # TODO '/Users/s.blundell/miniconda3/envs/O2_ML/lib/python3.12/site-packages/sklearn/linear_model/_sag.py:348: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
    #   warnings.warn('
    #  Should set max_iter higher, but also I think i'm supposed to scale here within a pipeline? (w FS and classifier)

    print(scores.head(len(scores)))
    # Example printed result:
    # TODO paste in example output once feature selection is more pinned down

    # TODO which perform best on test vs train F1 write here - currently svm and rf

    # todo: xgboost fails in test, and check if f1 score is best ranking, could test other models
### HYPEROPT PARAMETER TUNING ##########################################################################################
# For selected models, define a parameter params['type'] for the model name. Then evaluates parameters and calculates the cross-validated accuracy.
# Dictionary to store the best model accuracies
best_accuracies = { # warning need to pick the best models somewhere - is it just here?
    'svm': 0.0,
    'rf': 0.0,
    'logreg': 0.0,
    'xgb': 0.0,
    'gb': 0.0
}


# Objective function; which parameter configuation is used
def objective(params):
    classifier_type = params['type']
    del params['type']
    selector = feature_selectors_dict[classifier_type] #todo: also tune FS hyperparams

    # Build the classifier based on provided type and convert parameters that must be integers (hyperopt returns floats) if necessary
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    elif classifier_type == 'xgb':
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        params['n_estimators'] = int(params['n_estimators'])
        clf = XGBClassifier(**params)
    elif classifier_type == 'gb':
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_samples_split'] = int(params['min_samples_split'])
        params['min_samples_leaf'] = int(params['min_samples_leaf'])
        clf = GradientBoostingClassifier(**params)
    else:
        return {'loss': 1, 'status': STATUS_OK}

    # Incorporate feature selection into the pipeline
    pipe = Pipeline([
        ('feature_selector', selector),
        ('classifier', clf)
    ])

    # Use 10-fold cross validation to compute the mean accuracy
    accuracy = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1').mean()  # Reduced to 5-fold for speed

    # Log the best accuracy for each model type if improved
    if accuracy > best_accuracies[classifier_type]:
        best_accuracies[classifier_type] = accuracy
        mlflow.log_metric(f"best_{classifier_type}_accuracy", accuracy)

    # Because fmin() tries to minimize the objective, this function must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

# Define the search space over hyperparameters (for classifier only; feature selection is fixed) #TODO find more practical examples where these are defined; what is worth definining and what ranges?
search_space = hp.choice('classifier_type', [ #todo commented out ones i'm not investigating but colud be done more elegantly - eg extract best X from basic train and automatically pick those. or keep all if doing on hpc
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('svm_kernel', ['linear', 'rbf'])
    },
    {
        'type': 'rf',
        'criterion': hp.choice('rf_criterion', ['gini', 'entropy']),
        'n_estimators': hp.quniform('rf_n_estimators', 50, 500, 50),
        'max_depth': hp.quniform('rf_max_depth', 2, 10, 1),
        'min_samples_split': hp.quniform('rf_min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', 0.8]),
        'class_weight': hp.choice('rf_class_weight', [None, 'balanced'])
    },
    # {
    #     'type': 'logreg',
    #     'C': hp.lognormal('lr_C', 0, 1.0),
    #     # 'solver': hp.choice('lr_solver', ['liblinear', 'lbfgs'])
    #     'solver': 'saga',  # Force solver for elasticnet in feature selection
    #     'penalty': 'elasticnet',
    #     'l1_ratio': hp.uniform('l1_ratio', 0, 1),  # Required for ElasticNet
    # },
    # {
    #     'type': 'xgb',
    #     'max_depth': hp.quniform("xgb_max_depth", 3, 15, 1),
    #     'gamma': hp.uniform('xgb_gamma', 0, 9),
    #     'reg_alpha': hp.quniform('xgb_reg_alpha', 0, 10, 1),
    #     'reg_lambda': hp.uniform('xgb_reg_lambda', 0, 10),
    #     'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.6, 1),
    #     'min_child_weight': hp.quniform('xgb_min_child_weight', 0, 12, 1),
    #     'n_estimators': hp.quniform('xgb_n_estimators', 100, 500, 50),
    #     'seed': 0,
    #     'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
    #     'scale_pos_weight': hp.uniform('xgb_scale_pos_weight', 1, 10)  # Adjust if classes are imbalanced
    # },
    # {
    #     'type': 'gb',
    #     'n_estimators': hp.quniform('gb_n_estimators', 50, 500, 50),
    #     'max_depth': hp.quniform('gb_max_depth', 3, 15, 1),
    #     'min_samples_split': hp.quniform('gb_min_samples_split', 2, 20, 1),
    #     'min_samples_leaf': hp.quniform('gb_min_samples_leaf', 1, 10, 1),
    #     'learning_rate': hp.loguniform('gb_learning_rate', np.log(0.005), np.log(0.2)),
    #     'subsample': hp.uniform('gb_subsample', 0.6, 1.0),
    #     'max_features': hp.choice('gb_max_features', ['sqrt', 'log2', 0.8]),
    #     'loss': hp.choice('gb_loss', ['log_loss', 'exponential'])
    # },
])

print("Now tuning hyperparameters \n")

with mlflow.start_run():
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials()
    )

# Print the best accuracies for each model type
print("\nHighest model accuracies on train data:")
best_accuracy_df = pd.DataFrame(list(best_accuracies.items()), columns=['Models', 'Highest accuracy'])
print(best_accuracy_df)

# Extract and print the best hyperparameter configuration
best_config = space_eval(search_space, best_result)
print("\nBest model configuration:")
best_config_df = pd.DataFrame(list(best_config.items()), columns=['Parameters', 'Values'])
print(best_config_df)

### TRAIN FINAL MODEL ###########################################################################################
# Train final model using the full training data
mlflow.sklearn.autolog()
with mlflow.start_run():  # TODO need to find examples of this being done - unsure on the final training/testing after hyperopt tuning
    classifier_type = best_config['type']
    best_params = {k: v for k, v in best_config.items() if k != 'type'}
    selector = feature_selectors_dict[classifier_type]

    # Log the best hyperparameters
    mlflow.log_params(best_config)

    # Construct the classifier with the best parameters - converting to integers if needed
    if classifier_type == 'svm':
        classifier = SVC(**best_params)
    elif classifier_type == 'rf':
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        classifier = RandomForestClassifier(**best_params)
    elif classifier_type == 'logreg':
        classifier = LogisticRegression(**best_params)
    elif classifier_type == 'xgb':
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        classifier = XGBClassifier(**best_params)
    elif classifier_type == 'gb':
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
        classifier = GradientBoostingClassifier(**best_params)

    # Create the final pipeline with feature selection and classifier # TODO not sure if i need pipeline here since I dont feed into cross_val_score?
    final_pipeline = Pipeline([
        ('feature_selector', selector),
        ('classifier', classifier)
    ])

    # Train on full training data
    final_pipeline.fit(X_train, y_train)

    # Print the selected features
    try: # TODO this was made when RFECV was used for all, so likely no longer works
        selected = final_pipeline.named_steps['feature_selector']
        selected_features = X_train.columns[selected.support_]
        print(f"\nSelected {len(selected_features)} features:")
        print(selected_features.tolist())
    except:
        print("Unable to print features - see note in code.")

    # Log the final pipeline model
    mlflow.sklearn.log_model(final_pipeline, "best_model")

    # Save final model #todo not sure what to do with this yet but worth saving - or does mlflow save?
    joblib.dump(final_pipeline, "Oxygen_Prediction_Model.joblib")
    mlflow.log_artifact("Oxygen_Prediction_Model.joblib")

    # Evaluate the final model on the test set
    y_pred = final_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    print(f"\nTest accuracy with best model ({classifier_type}): {test_accuracy:.4f}")
    print(f"Test F1 score with best model ({classifier_type}): {test_f1:.4f}")

    ### Plot learning curve #todo not sure if this (and auc) is in the right place - check over code too
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
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")

    # Log the figure as an MLflow artifact
    mlflow.log_figure(fig, "learning_curve.png")
    plt.close(fig)

    ### Plot ROC/AUC curves
    # Get prediction probabilities
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot and log
    fig, ax = plt.subplots()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

    # Log the AUC metric explicitly
    mlflow.log_metric("test_auc", roc_auc)


    # Get feature list
    # Grab the fitted selector step
    sel = final_pipeline.named_steps['feature_selector']

    # If it’s a scikit‐learn selector (RFECV, SelectKBest, etc.), you can use .get_support() # todo make compatible with all
    mask = sel.get_support()  # boolean mask of length n_features

    # Apply that mask to the original feature names
    feature_names = X_train.columns[mask]
    print(f"{len(feature_names)} features selected:")
    print(feature_names.tolist())

    ### Plot feature importance #todo test for all
    # Extract feature importances based on the classifier type
    if classifier_type in ['rf', 'xgb', 'gb']:
        # These classifiers have feature_importances_ attribute
        importances = final_pipeline.named_steps['classifier'].feature_importances_

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

        # Create DataFrame for easier plotting with seaborn
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Plot with seaborn
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
        ax.set_title(f'Top 20 Feature Importances - {classifier_type.upper()}', fontsize=16)
        ax.set_xlabel('Importance', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        print(f"\nTop 10 most important features:")
        print(importance_df.head(10))

    elif classifier_type == 'svm' and best_params.get('kernel') == 'linear':
        # For linear SVM, we can extract coefficients
        coefficients = np.abs(final_pipeline.named_steps['classifier'].coef_[0])

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

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
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "feature_coefficients.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("feature_coefficients.csv", index=False)
        mlflow.log_artifact("feature_coefficients.csv")

        print(f"\nTop 10 most important features (by coefficient magnitude):")
        print(importance_df.head(10))

    elif classifier_type == 'logreg':
        # For logistic regression, extract coefficients
        coefficients = np.abs(final_pipeline.named_steps['classifier'].coef_[0])

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

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
        plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "feature_coefficients.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("feature_coefficients.csv", index=False)
        mlflow.log_artifact("feature_coefficients.csv")

        print(f"\nTop 10 most important features (by coefficient magnitude):")
        print(importance_df.head(10))

    else:
        # For other models where direct feature importance is not available
        # Use permutation importance as an alternative
        print("\nCalculating permutation importance for features...")

        # Calculate permutation importance
        perm_importance = permutation_importance(
            final_pipeline, X_test, y_test,
            n_repeats=10,
            random_state=42
        )

        # Get the selected feature names
        selector = final_pipeline.named_steps['feature_selector']
        feature_mask = selector.get_support()
        selected_features = X_train.columns[feature_mask]

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
        plt.savefig("permutation_importance.png", dpi=300, bbox_inches='tight') # todo added - but logs below so not sure if redundant

        # Log the figure to MLflow
        mlflow.log_figure(plt.gcf(), "permutation_importance.png")
        plt.close()

        # Also save the full feature importance DataFrame as CSV
        importance_df.to_csv("permutation_importances.csv", index=False)
        mlflow.log_artifact("permutation_importances.csv")

        print(f"\nTop 10 most important features (by permutation importance):")
        print(importance_df.head(10))


    ### END OF GRAPHS todo check over

# Example output:
# Test accuracy with best model (rf): 0.6000
# Test F1 score with best model (rf): 0.6957


# TODO: Question: If I get different models (LR and GB currently) on different runs, what should I do? Pick one? Use the
#  most common of multiple attempts? Or set seed so it's always consistent


# IMPROVE: Early stopping isn't implemented at all because it would work for some and not others so is more complicated to implement - but could add.
#  Could also do an ensemble model approach for the final training, and stacking/voting
#  More elegant way to handle hyperopt returning floats?

