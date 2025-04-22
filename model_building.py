import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import mlflow

# TODO produce graphs of metrics like AUC, ROC precision recall curve, etc.

### Read in data
# Train
X_path = Path(__file__).parent / "Surrey_X_train.csv"
y_path = Path(__file__).parent / "Surrey_y_train.csv"
X_train = pd.read_csv(X_path)
y_train = pd.read_csv(y_path).squeeze()  # Convert to 1D array
# Test
X_path = Path(__file__).parent / "Surrey_X_test.csv"
y_path = Path(__file__).parent / "Surrey_y_test.csv"
X_test = pd.read_csv(X_path)
y_test = pd.read_csv(y_path).squeeze()  # Convert to 1D array
