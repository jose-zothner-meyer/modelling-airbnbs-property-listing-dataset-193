import itertools  # Provides functions for creating and working with iterators, e.g. product
import json       # Allows reading/writing data in JSON format
import math       # Provides mathematical functions such as sqrt
import os         # Offers a way of using operating system dependent functionality
import pandas as pd  # Used for data manipulation in DataFrame structures
import joblib         # Enables saving/loading Python objects to disk
import numpy as np    # Fundamental numerical computing package
from joblib import load  # A function from the joblib library to load saved models
from sklearn import metrics  # Provides various machine learning evaluation metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor 
# Two tree-based ensemble models for regression: GradientBoostingRegressor and RandomForestRegressor
from sklearn.linear_model import SGDRegressor  # A linear model with stochastic gradient descent
from sklearn.metrics import mean_squared_error, r2_score  # mean_squared_error, r2_score are metrics for regression
from sklearn.model_selection import GridSearchCV, train_test_split  # GridSearchCV performs exhaustive parameter search, train_test_split splits data
from sklearn.pipeline import Pipeline  # Lets you chain transformers and estimators
from sklearn.preprocessing import StandardScaler  # Scales features to zero mean, unit variance
from sklearn.tree import DecisionTreeRegressor  # A non‐linear regression model based on decision trees
from tabular_data import load_airbnb  # Custom function to load Airbnb data
from typing import Tuple, Dict, Any, List  # For type hinting function parameters and returns


def import_and_standarised_data(data_file: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Imports the Airbnb data from a CSV file and returns standardized numeric features
    plus the target column (Price_Night).

    Parameters
    ----------
    data_file : str
        Path to the CSV file containing the Airbnb data.

    Returns
    -------
    X : pd.DataFrame
        A DataFrame of numeric features (after standardization) and possibly 
        other columns if not numeric but not scaled.
    y : pd.Series
        The target column (Price_Night).
    """
    # 1. Read CSV into a pandas DataFrame
    df = pd.read_csv(data_file)
    
    # 2. Pass the DataFrame to load_airbnb (which returns (features, labels))
    X, y = load_airbnb(df)

    # 3. Select only numeric columns from X for standardization
    numeric_columns = X.select_dtypes(include=[np.number])

    # 4. Fit a StandardScaler on the numeric columns and transform them
    std = StandardScaler()
    scaled_features = std.fit_transform(numeric_columns.values)

    # 5. Overwrite the numeric columns in X with their scaled versions
    X[numeric_columns.columns] = scaled_features

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataset into training, validation, and testing sets (70%-15%-15%).

    Parameters
    ----------
    X : pd.DataFrame
        The features of the dataset.
    y : pd.Series
        The target values corresponding to the features.

    Returns
    -------
    X_train : pd.DataFrame
        The features for training the model (70%).
    X_validation : pd.DataFrame
        The features for validation (15%).
    X_test : pd.DataFrame
        The features for testing the model (15%).
    y_train : pd.Series
        The target values for training (70%).
    y_validation : pd.Series
        The target values for validation (15%).
    y_test : pd.Series
        The target values for testing (15%).
    """
    np.random.seed(10)  # Sets a random seed for reproducible shuffling

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    # Second split: Of the 30% temp, half becomes final test (15%), half validation (15%)
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_temp, y_temp, test_size=0.5
    )

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
    """
    Performs an exhaustive grid search (manual nested loops) over the specified 
    hyperparameter values for a given regression model, using the training and 
    validation sets to determine the best hyperparameters. 
    Then evaluates the best model on the test set.

    Parameters
    ----------
    model : Any
        The base regression model to be tuned (SGDRegressor, RandomForestRegressor, etc.).
    X_train : pd.DataFrame
        The features for training.
    X_validation : pd.DataFrame
        The features for validation.
    X_test : pd.DataFrame
        The features for testing (used for final performance evaluation).
    y_train : pd.Series
        The labels for training.
    y_validation : pd.Series
        The labels for validation.
    y_test : pd.Series
        The labels for testing.
    hyperparameters_dict : Dict[str, List[Any]]
        Dictionary where keys are hyperparameter names, and values are 
        lists of possible parameter values to try.

    Returns
    -------
    best_model : Any
        The model instance trained with the best hyperparameters.
    best_hyperparameters : Dict[str, Any]
        The dictionary of hyperparameters corresponding to the best model.
    performance_metrics : Dict[str, float]
        Performance metrics (e.g., RMSE, R^2) of the best model on the test set.
    """
    best_model = None
    best_hyperparameters = None
    best_rmse = float('inf')

    # Iterate over all possible hyperparameter combinations using itertools.product
    for hyperparameter_values in itertools.product(*hyperparameters_dict.values()):
        # Pair up the keys and the current combination of values
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        
        # Clone and set the hyperparameters on the model
        current_model = model.set_params(**hyperparameters)
        
        # Fit on the training set
        current_model.fit(X_train, y_train)
        
        # Predict on the validation set and compute RMSE
        y_val_pred = current_model.predict(X_validation)
        rmse_val = math.sqrt(mean_squared_error(y_validation, y_val_pred))

        # If the validation RMSE is the best so far, update best_model
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = current_model
            best_hyperparameters = hyperparameters

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)

    performance_metrics = {
        "validation_RMSE": rmse_test,  # naming it "validation_RMSE" but it's actually your test RMSE
        "R^2": r2_test
    }

    return best_model, best_hyperparameters, performance_metrics


# --------------------------------------------------------------------------
# LOG-TRANSFORMED VERSION
# --------------------------------------------------------------------------
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
    """
    Similar to custom_tune_regression_model_hyperparameters but for log-transformed targets.

    - We train on log(Price_Night + 1) for y_train and y_validation.
    - We measure final performance by exponentiating predictions and comparing 
      to the original y_test.

    Parameters
    ----------
    model : Any
        Regression model to tune.
    X_train, X_validation, X_test : pd.DataFrame
        Features for training, validation, and testing.
    y_train_log, y_validation_log : pd.Series
        The log-transformed targets for training and validation.
    y_test_original : pd.Series
        The original (non-log) y values for final test evaluation.
    hyperparameters_dict : Dict[str, List[Any]]
        Hyperparameter grid.

    Returns
    -------
    best_model : Any
        Model with best hyperparameters (under log transformation).
    best_hyperparameters : Dict[str, Any]
        Dictionary of best hyperparams.
    performance_metrics : Dict[str, float]
        Performance metrics (RMSE and R^2) on the original scale (Price_Night).
    """
    best_model = None
    best_hyperparameters = None
    best_rmse = float('inf')

    for hyperparameter_values in itertools.product(*hyperparameters_dict.values()):
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        
        current_model = model.set_params(**hyperparameters)
        current_model.fit(X_train, y_train_log)  # train on the log transform

        # Predict on validation set (log scale), then exponentiate
        y_val_log_pred = current_model.predict(X_validation)
        y_val_pred = np.expm1(y_val_log_pred)  # revert to original Price scale

        y_val_true = np.expm1(y_validation_log)  # revert the true val set as well

        rmse_val = math.sqrt(mean_squared_error(y_val_true, y_val_pred))

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_model = current_model
            best_hyperparameters = hyperparameters

    # Evaluate best model on test set
    y_test_log_pred = best_model.predict(X_test)
    y_test_pred = np.expm1(y_test_log_pred)
    # Compare to original test labels
    rmse_test = math.sqrt(mean_squared_error(y_test_original, y_test_pred))
    r2_test = r2_score(y_test_original, y_test_pred)

    performance_metrics = {
        "validation_RMSE": rmse_test,  # on test set, but naming as "validation_RMSE"
        "R^2": r2_test
    }

    return best_model, best_hyperparameters, performance_metrics


def tune_regression_model_hyperparameters(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Uses GridSearchCV to systematically work through multiple combinations
    of hyperparameters for a RandomForestRegressor, 
    to find the best possible parameters that minimize MSE.

    Parameters
    ----------
    X : pd.DataFrame
        The features of the dataset.
    y : pd.Series
        The corresponding target (Price_Night) for each feature row.

    Returns
    -------
    None
    """
    random_forest_model = RandomForestRegressor()

    parameter_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=random_forest_model,
        param_grid=parameter_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X.values, y)

    print(f"Training and tuning RandomForestRegressor...")
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Score (Negative Mean Squared Error):", grid_search.best_score_)


def save_model(
    folder_name: str, 
    best_model: Any, 
    best_hyperparameters: Dict[str, Any], 
    performance_metric: Dict[str, float]
) -> None:
    """
    Saves a trained regression model, its hyperparameters, and performance metrics to disk
    under 'modelling-airbnbs-property-listing-dataset-193/models/regression/<folder_name>/'.

    Parameters
    ----------
    folder_name : str
        The subfolder name (e.g., "SGDRegressor" or "RandomForestRegressor").
    best_model : Any
        The trained regression model to be saved.
    best_hyperparameters : Dict[str, Any]
        The hyperparameters used for training the best model.
    performance_metric : Dict[str, float]
        A dictionary containing relevant performance metrics (e.g., RMSE, R^2).

    Returns
    -------
    None
    """
    # Base folder for storing models
    models_dir = 'modelling-airbnbs-property-listing-dataset-193/models/'
    current_dir = os.path.dirname(os.getcwd())
    models_path = os.path.join(current_dir, models_dir)
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    # "regression" sub-dir
    regression_dir = 'modelling-airbnbs-property-listing-dataset-193/models/regression/'
    regression_path = os.path.join(current_dir, regression_dir)
    if not os.path.exists(regression_path):
        os.mkdir(regression_path)

    # Subfolder for this particular model
    folder_name_dir = os.path.join(regression_path, folder_name)
    folder_name_path = os.path.join(current_dir, folder_name_dir)
    if not os.path.exists(folder_name_path):
        os.mkdir(folder_name_path)

    # Save the model
    joblib.dump(best_model, os.path.join(folder_name_path, "model.joblib"))

    # Save hyperparams
    hyperparameters_filename = os.path.join(folder_name_path, "hyperparameters.json")
    with open(hyperparameters_filename, 'w') as json_file:
        json.dump(best_hyperparameters, json_file)
    
    # Save metrics
    metrics_filename = os.path.join(folder_name_path, "metrics.json")
    with open(metrics_filename, 'w') as json_file:
        json.dump(performance_metric, json_file)


def evaluate_all_models(
    models: List[Any], 
    hyperparameters_dict: List[Dict[str, List[Any]]]
) -> None:
    """
    Evaluates and tunes multiple regression models (predicting Price_Night directly) 
    with different sets of hyperparameters, saving their best configurations.

    Parameters
    ----------
    models : List[Any]
        A list of regression model instances (e.g., SGDRegressor(), RandomForestRegressor()).
    hyperparameters_dict : List[Dict[str, List[Any]]]
        A list where each element is a dictionary of hyperparameter search spaces
        for the corresponding model in 'models'.

    Returns
    -------
    None
    """
    data_file = "clean_tabular_data.csv"
    X, y = import_and_standarised_data(data_file)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    for i, model in enumerate(models):
        best_model, best_hparams, performance_metrics = custom_tune_regression_model_hyperparameters(
            model,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            hyperparameters_dict[i]
        )
        print(best_model, best_hparams, performance_metrics)
        
        folder_name = str(model)[:-2]  # e.g. "SGDRegressor"
        save_model(folder_name, best_model, best_hparams, performance_metrics)


def evaluate_all_models_log_target(
    models: List[Any],
    hyperparameters_dict: List[Dict[str, List[Any]]]
) -> None:
    """
    Similar to evaluate_all_models, but predicts log(Price_Night + 1) 
    and exponentiates predictions back to the original Price_Night scale.

    Parameters
    ----------
    models : List[Any]
        List of regression model instances.
    hyperparameters_dict : List[Dict[str, List[Any]]]
        List of hyperparameter grids for each model.

    Returns
    -------
    None
    """
    data_file = "clean_tabular_data.csv"
    X, y = import_and_standarised_data(data_file)

    # Transform y => log(y + 1)
    y_log = np.log1p(y)

    # We'll still do a normal split on X, y
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Also keep log versions for train & val
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    # y_test remains un-transformed for final performance check

    for i, model in enumerate(models):
        best_model, best_hparams, performance_metrics = custom_tune_regression_model_hyperparameters_log(
            model,
            X_train, X_val, X_test,
            y_train_log, y_val_log, y_test,
            hyperparameters_dict[i]
        )
        print(best_model, best_hparams, performance_metrics)

        # Save these models to a distinct subfolder so we don't overwrite the raw ones
        folder_name = str(model)[:-2] + "_log"
        save_model(folder_name, best_model, best_hparams, performance_metrics)


def find_best_model(models: List[Any]) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Finds the best regression model by comparing saved models in 
    'modelling-airbnbs-property-listing-dataset-193/models/regression/' subfolders.

    Parameters
    ----------
    models : List[Any]
        A list of model instances used to identify subfolders.

    Returns
    -------
    best_regression_model : Any
        The best‐performing loaded regression model object.
    best_hyperparameters_dict : Dict[str, Any]
        Hyperparameters for the best model.
    best_metrics_dict : Dict[str, float]
        The performance metrics for the best model.
    """
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}

    regression_dir = "modelling-airbnbs-property-listing-dataset-193/models/regression"
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)

    # For each model, read from its subfolder
    for model_obj in models:
        model_str = str(model_obj)[0:-2]  
        model_dir = os.path.join(regression_path, model_str)
        
        if not os.path.exists(model_dir):
            continue

        # Attempt to load
        model_path = os.path.join(model_dir, 'model.joblib')
        if not os.path.isfile(model_path):
            continue
        
        # Load the stored model
        loaded_model = load(model_path)

        # Load hyperparams
        hyper_file = os.path.join(model_dir, 'hyperparameters.json')
        if not os.path.isfile(hyper_file):
            continue
        with open(hyper_file, 'r') as f:
            hyperparams = json.load(f)

        # Load metrics
        metrics_file = os.path.join(model_dir, 'metrics.json')
        if not os.path.isfile(metrics_file):
            continue
        with open(metrics_file, 'r') as f:
            metric_data = json.load(f)

        # Compare by R^2
        if best_regression_model is None or metric_data.get("R^2", -9999) > best_metrics_dict.get("R^2", -9999):
            best_regression_model = loaded_model
            best_hyperparameters_dict = hyperparams
            best_metrics_dict = metric_data

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict


# Define the models to test
models = [
    SGDRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor()
]

# Define hyperparameter dictionaries
hyperparameters_dict = [
    {  # Hyperparameters for SGDRegressor
        'loss': ['squared_error', 'huber', 'squared_epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001],
        'l1_ratio': [0.15, 0.2],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000],
        'tol': [1e-3, 1e-4],
        'shuffle': [True, False],
        'early_stopping': [True, False]
    },
    {  # Hyperparameters for DecisionTreeRegressor
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_features': [10]
    },
    {  # Hyperparameters for RandomForestRegressor
        'n_estimators': [50, 100],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'bootstrap': [True, False],
        'max_features': [10]
    },
    {  # Hyperparameters for GradientBoostingRegressor
        'loss': ['squared_error', 'huber'],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [50, 100],
        'criterion': ['squared_error', 'friedman_mse'],
        'max_features': [10],
    }
]


if __name__ == "__main__":
    # 1) Evaluate all models on the original Price_Night target
    print(">>> Evaluating models on raw Price_Night <<<")
    evaluate_all_models(models, hyperparameters_dict)

    best_regression_model, best_hyperparams, best_metrics = find_best_model(models)
    print("\nBEST MODEL (raw target) =>", best_regression_model)
    print("Best Hyperparams:", best_hyperparams)
    print("Best Metrics:", best_metrics)

    # 2) Evaluate all models on log(Price_Night+1) target
    print("\n>>> Evaluating models on log(Price_Night + 1) <<<")
    evaluate_all_models_log_target(models, hyperparameters_dict)

    print("\n(NOTE) If you want to find the best among the log_ subfolders automatically,")
    print("you could implement a similar 'find_best_model_log()' or adapt find_best_model")
    print("to look for subfolders named '<ModelName>_log'.")
