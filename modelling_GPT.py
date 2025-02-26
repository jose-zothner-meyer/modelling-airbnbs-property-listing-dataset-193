import itertools
import json
import math
import os
import pandas as pd
import joblib
import numpy as np
from joblib import load
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple, Dict, Any, List

from tabular_data import load_airbnb  # Assuming your load_airbnb function is here.


def import_and_standarised_data(data_file: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_file)
    X, y = load_airbnb(df)
    numeric_columns = X.select_dtypes(include=[np.number])

    std = StandardScaler()
    scaled_features = std.fit_transform(numeric_columns.values)
    X[numeric_columns.columns] = scaled_features
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    np.random.seed(10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=12)
    X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=0.5)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def custom_tune_regression_model_hyperparameters(
    model: Any,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
    hyperparameters_dict: Dict[str, List[Any]]
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    best_model = None
    best_hyperparameters = None
    best_rmse = float('inf')

    for hyperparameter_values in itertools.product(*hyperparameters_dict.values()):
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        current_model = model.set_params(**hyperparameters)
        current_model.fit(X_train, y_train)

        y_val_pred = current_model.predict(X_validation)
        rmse_val = math.sqrt(mean_squared_error(y_validation, y_val_pred))

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = current_model
            best_hyperparameters = hyperparameters

    y_test_pred = best_model.predict(X_test)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)

    performance_metrics = {
        "validation_RMSE": rmse_test,
        "R^2": r2_test
    }
    return best_model, best_hyperparameters, performance_metrics


def custom_tune_regression_model_hyperparameters_log(
    model: Any,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: pd.Series,
    y_validation_log: pd.Series,
    y_test_original: pd.Series,
    hyperparameters_dict: Dict[str, List[Any]]
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    best_model = None
    best_hyperparameters = None
    best_rmse = float('inf')

    for hyperparameter_values in itertools.product(*hyperparameters_dict.values()):
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        current_model = model.set_params(**hyperparameters)
        current_model.fit(X_train, y_train_log)

        y_val_log_pred = current_model.predict(X_validation)
        y_val_pred = np.expm1(y_val_log_pred)
        y_val_true = np.expm1(y_validation_log)

        rmse_val = math.sqrt(mean_squared_error(y_val_true, y_val_pred))

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = current_model
            best_hyperparameters = hyperparameters

    y_test_log_pred = best_model.predict(X_test)
    y_test_pred = np.expm1(y_test_log_pred)

    rmse_test = math.sqrt(mean_squared_error(y_test_original, y_test_pred))
    r2_test = r2_score(y_test_original, y_test_pred)

    performance_metrics = {
        "validation_RMSE": rmse_test,
        "R^2": r2_test
    }
    return best_model, best_hyperparameters, performance_metrics


def save_model(
    folder_name: str,
    best_model: Any,
    best_hyperparameters: Dict[str, Any],
    performance_metric: Dict[str, float]
) -> None:
    models_dir = 'modelling-airbnbs-property-listing-dataset-193/models/'
    current_dir = os.path.dirname(os.getcwd())
    models_path = os.path.join(current_dir, models_dir)
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    regression_dir = 'modelling-airbnbs-property-listing-dataset-193/models/regression/'
    regression_path = os.path.join(current_dir, regression_dir)
    if not os.path.exists(regression_path):
        os.mkdir(regression_path)

    folder_name_dir = os.path.join(regression_path, folder_name)
    folder_name_path = os.path.join(current_dir, folder_name_dir)
    if not os.path.exists(folder_name_path):
        os.mkdir(folder_name_path)

    joblib.dump(best_model, os.path.join(folder_name_path, "model.joblib"))

    hyperparameters_filename = os.path.join(folder_name_path, "hyperparameters.json")
    with open(hyperparameters_filename, 'w') as json_file:
        json.dump(best_hyperparameters, json_file)

    metrics_filename = os.path.join(folder_name_path, "metrics.json")
    with open(metrics_filename, 'w') as json_file:
        json.dump(performance_metric, json_file)


def evaluate_all_models(
    models: List[Any],
    hyperparameters_dict: List[Dict[str, List[Any]]]
) -> None:
    data_file = "clean_tabular_data.csv"
    X, y = import_and_standarised_data(data_file)

    print("\n[SKEW CHECK - Raw Target]")
    print("Original Price_Night skew:", y.skew())
    print("Log(Price_Night + 1) skew:", np.log1p(y).skew())

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    for i, model in enumerate(models):
        best_model, best_hparams, performance_metrics = custom_tune_regression_model_hyperparameters(
            model,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            hyperparameters_dict[i]
        )
        print(f"\nBest model found for {model}:\n", best_model)
        print("Hyperparameters:", best_hparams)
        print("Performance (test):", performance_metrics)

        folder_name = str(model)[:-2]
        save_model(folder_name, best_model, best_hparams, performance_metrics)


def evaluate_all_models_log_target(
    models: List[Any],
    hyperparameters_dict: List[Dict[str, List[Any]]]
) -> None:
    data_file = "clean_tabular_data.csv"
    X, y = import_and_standarised_data(data_file)

    print("\n[SKEW CHECK - Log Target Run]")
    print("Original Price_Night skew:", y.skew())
    print("Log(Price_Night + 1) skew:", np.log1p(y).skew())

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    for i, model in enumerate(models):
        best_model, best_hparams, performance_metrics = custom_tune_regression_model_hyperparameters_log(
            model,
            X_train, X_val, X_test,
            y_train_log, y_val_log, y_test,
            hyperparameters_dict[i]
        )
        print(f"\nBest model (log target) for {model}:\n", best_model)
        print("Hyperparameters:", best_hparams)
        print("Performance (on original scale):", performance_metrics)

        folder_name = str(model)[:-2] + "_log"
        save_model(folder_name, best_model, best_hparams, performance_metrics)


def find_best_model(models: List[Any]) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Finds the best regression model among the subfolders for the RAW (non-log) runs.
    Looks in 'modelling-airbnbs-property-listing-dataset-193/models/regression/<ModelName>/'.
    """
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}

    regression_dir = "modelling-airbnbs-property-listing-dataset-193/models/regression"
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)

    for model_obj in models:
        model_str = str(model_obj)[0:-2]
        model_dir = os.path.join(regression_path, model_str)
        
        if not os.path.exists(model_dir):
            continue

        model_path = os.path.join(model_dir, 'model.joblib')
        if not os.path.isfile(model_path):
            continue
        loaded_model = load(model_path)

        hyper_file = os.path.join(model_dir, 'hyperparameters.json')
        if not os.path.isfile(hyper_file):
            continue
        with open(hyper_file, 'r') as f:
            hyperparams = json.load(f)

        metrics_file = os.path.join(model_dir, 'metrics.json')
        if not os.path.isfile(metrics_file):
            continue
        with open(metrics_file, 'r') as f:
            metric_data = json.load(f)

        if best_regression_model is None or metric_data.get("R^2", -9999) > best_metrics_dict.get("R^2", -9999):
            best_regression_model = loaded_model
            best_hyperparameters_dict = hyperparams
            best_metrics_dict = metric_data

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict


def find_best_model_log(models: List[Any]) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Finds the best regression model among the subfolders for the LOG runs.
    Looks in 'modelling-airbnbs-property-listing-dataset-193/models/regression/<ModelName>_log/'.
    """
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}

    regression_dir = "modelling-airbnbs-property-listing-dataset-193/models/regression"
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)

    for model_obj in models:
        # e.g. "SGDRegressor" => "SGDRegressor_log"
        model_str = str(model_obj)[0:-2] + "_log"
        model_dir = os.path.join(regression_path, model_str)
        
        if not os.path.exists(model_dir):
            continue

        model_path = os.path.join(model_dir, 'model.joblib')
        if not os.path.isfile(model_path):
            continue
        loaded_model = load(model_path)

        hyper_file = os.path.join(model_dir, 'hyperparameters.json')
        if not os.path.isfile(hyper_file):
            continue
        with open(hyper_file, 'r') as f:
            hyperparams = json.load(f)

        metrics_file = os.path.join(model_dir, 'metrics.json')
        if not os.path.isfile(metrics_file):
            continue
        with open(metrics_file, 'r') as f:
            metric_data = json.load(f)

        if best_regression_model is None or metric_data.get("R^2", -9999) > best_metrics_dict.get("R^2", -9999):
            best_regression_model = loaded_model
            best_hyperparameters_dict = hyperparams
            best_metrics_dict = metric_data

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict


# Define models & hyperparameter grids
models = [
    SGDRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor()
]

hyperparameters_dict = [
    {
        'loss': ['squared_error', 'huber', 'squared_epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.00001, 0.0001],
        'l1_ratio': [0.15, 0.2],
        'fit_intercept': [True, False],
        'max_iter': [5000, 10000],
        'tol': [1e-5, 1e-6],
        'shuffle': [True, False],
        'early_stopping': [True, False]
    },
    { 
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_features': [10]
    },
    {
        'n_estimators': [50, 100],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'bootstrap': [True, False],
        'max_features': [10]
    },
    {
        'loss': ['squared_error', 'huber'],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [50, 100],
        'criterion': ['squared_error', 'friedman_mse'],
        'max_features': [10],
    }
]


if __name__ == "__main__":
    # 1. Evaluate all models on raw target
    print(">>> Evaluating models on raw Price_Night <<<")
    evaluate_all_models(models, hyperparameters_dict)

    # 2. Pick the best among the raw-target subfolders
    best_regression_model, best_hyperparams, best_metrics = find_best_model(models)
    print("\nBEST MODEL (raw target) =>", best_regression_model)
    print("Best Hyperparams:", best_hyperparams)
    print("Best Metrics:", best_metrics)

    # 3. Evaluate all models on log target
    print("\n>>> Evaluating models on log(Price_Night + 1) <<<")
    evaluate_all_models_log_target(models, hyperparameters_dict)

    # 4. Pick the best among the log-target subfolders
    best_regression_model_log, best_hyperparams_log, best_metrics_log = find_best_model_log(models)
    print("\nBEST MODEL (log target) =>", best_regression_model_log)
    print("Best Hyperparams (log target):", best_hyperparams_log)
    print("Best Metrics (log target):", best_metrics_log)
