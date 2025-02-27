from scipy.stats import boxcox

import itertools  # Provides functions for creating and working with iterators, e.g. product
import json       # Allows reading/writing data in JSON format
import math       # Provides mathematical functions such as sqrt
import os         # Offers a way of using operating system dependent functionality
import pandas as pd  # Used for data manipulation in DataFrame structures
import joblib         # Enables saving/loading Python objects to disk
import numpy as np    # Fundamental numerical computing package

from joblib import load  # A function from the joblib library to load saved models
from sklearn import metrics  # Provides various machine learning evaluation metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # Two tree-based ensemble models for regression: GradientBoostingRegressor and RandomForestRegressor
from sklearn.linear_model import SGDRegressor  # A linear model with stochastic gradient descent
from sklearn.metrics import mean_squared_error, r2_score # mean_squared_error, r2_score are metrics for regression
from sklearn.model_selection import GridSearchCV, train_test_split # GridSearchCV performs exhaustive parameter search, train_test_split splits data
from sklearn.pipeline import Pipeline  # Lets you chain transformers and estimators
from sklearn.preprocessing import StandardScaler  # Scales features to zero mean, unit variance
from sklearn.tree import DecisionTreeRegressor  # A non‐linear regression model based on decision trees
from tabular_data import load_airbnb  # Custom function to load Airbnb data
from typing import Tuple, Dict, Any, List  # For type hinting function parameters and returns


def import_and_standarised_data(data_file: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Imports the Airbnb data from a CSV file and returns standardized numeric features
    plus the target column. 
    """

    # 1. Read CSV into a pandas DataFrame
    df = pd.read_csv(data_file)  # Load the data from the CSV file into a DataFrame

    # List of continuous columns to apply Box-Cox transformation
    continuous_columns = [
        "Price_Night", "Cleanliness_rating",
        "Accuracy_rating", "Communication_rating", "Location_rating",
        "Check-in_rating", "Value_rating", "amenities_count"
    ]

    # Apply Box-Cox transformation to each continuous column
    for column in continuous_columns:
        df[column], _ = boxcox(df[column])  # Apply Box-Cox transformation directly to the column
    
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
    X : pandas.core.frame.DataFrame
        The features of the dataset.
    y : pandas.core.series.Series
        The target values corresponding to the features.

    Returns
    -------
    X_train : pandas.core.frame.DataFrame
        The features for training the model.
    X_validation : pandas.core.frame.DataFrame
        The features for validation.
    X_test : pandas.core.frame.DataFrame
        The features for testing the model.
    y_train : pandas.core.series.Series
        The target values for training.
    y_validation : pandas.core.series.Series
        The target values for validation.
    y_test : pandas.core.series.Series
        The target values for testing.
    """
    np.random.seed(10)  # Sets a random seed for reproducible shuffling

    # First split: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    # Second split: Of the 30% test, half becomes final test (15%), half becomes validation (15%)
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def train_linear_regression_model(
    X_train: pd.DataFrame, 
    X_validation: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_validation: pd.Series, 
    y_test: pd.Series
) -> None:
    """
    Trains a linear regression model (SGDRegressor) and evaluates its performance 
    on training and test sets.

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
        The features for training.
    X_validation : pandas.core.frame.DataFrame
        The features for validation (not explicitly used in this function's metric reporting).
    X_test : pandas.core.frame.DataFrame
        The features for testing.
    y_train : pandas.core.series.Series
        The labels for training.
    y_validation : pandas.core.series.Series
        The labels for validation (not explicitly used in this function's metric reporting).
    y_test : pandas.core.series.Series
        The labels for testing.

    Returns
    -------
    None
        This function prints performance metrics but returns nothing.
    """
    linear_regression_model_SDGRegr = SGDRegressor()  # Initializes a linear model with SGD
    model = linear_regression_model_SDGRegr.fit(X_train, y_train)  # Trains the model on the training set

    y_pred = model.predict(X_test)  # Predicts the targets for the test set
    y_pred_train = model.predict(X_train)  # Predicts the targets for the training set

    # Calculate performance metrics on the TEST set
    mse = mean_squared_error(y_test, y_pred)  # Mean squared error for test predictions
    rmse = math.sqrt(mse)                     # Root mean squared error for test predictions
    r2 = r2_score(y_test, y_pred)             # R^2 (coefficient of determination) for test predictions

    # Print the performance metrics for the TEST set
    print(f"RMSE: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared (R2) Score: {r2}")

    # Calculate performance metrics on the TRAIN set
    mse = mean_squared_error(y_train, y_pred_train)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_train, y_pred_train)

    print("\nTraining Metrics")
    print(f"RMSE: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared (R2) Score: {r2}")


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
    Performs an exhaustive grid search over the specified hyperparameter values 
    for a given regression model, using the training and validation sets 
    to determine the best hyperparameters.

    Parameters
    ----------
    model : Any
        The base regression model to be tuned.
    X_train : pandas.core.frame.DataFrame
        The features for training.
    X_validation : pandas.core.frame.DataFrame
        The features for validation.
    X_test : pandas.core.frame.DataFrame
        The features for testing, used to evaluate final performance.
    y_train : pandas.core.series.Series
        The labels for training.
    y_validation : pandas.core.series.Series
        The labels for validation.
    y_test : pandas.core.series.Series
        The labels for testing.
    hyperparameters_dict : Dict[str, List[Any]]
        A dictionary where keys are hyperparameter names, and values are 
        lists of possible parameter values to try.

    Returns
    -------
    best_model : Any
        The model instance trained with the best hyperparameters.
    best_hyperparameters : Dict[str, Any]
        The dictionary of hyperparameters corresponding to the best model.
    performance_metrics : Dict[str, float]
        A dictionary containing performance metrics (e.g., RMSE, R^2) of 
        the best model on the test set.
    """
    best_model = None               # Will hold the best performing model found so far
    best_hyperparameters = None     # Will hold the hyperparams for the best model
    best_rmse = float('inf')        # Initialize best RMSE to infinity for comparison
    validation_R2 = []              # List to store the validation R^2 for all attempts

    # Iterate over all possible hyperparameter combinations
    for hyperparameter_values in itertools.product(*hyperparameters_dict.values()):
        # Create a dict by pairing keys and one combination of values
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        
        # Update the model with these hyperparameters
        regression_model = model.set_params(**hyperparameters)
        
        # Train the updated model on the training set
        model = regression_model.fit(X_train, y_train)
        
        # Predict on the validation set
        y_pred_val = model.predict(X_validation)
        
        # Compute RMSE on the validation predictions
        rmse_val = math.sqrt(mean_squared_error(y_validation, y_pred_val))
        
        # Also compute the R^2 on validation
        validation_R2.append(metrics.r2_score(y_validation, y_pred_val))

        # If this validation RMSE is better (lower) than any found so far, update best model
        if rmse_val < best_rmse:
            best_model = model
            best_hyperparameters = hyperparameters
            best_rmse = rmse_val

    # Evaluate the best model on the test set
    y_pred_test = best_model.predict(X_test)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_pred_test))
    test_R2 = metrics.r2_score(y_test, y_pred_test)

    # Compile final performance metrics into a dictionary
    performance_metrics = {
        "validation_RMSE": rmse_test,  # The test RMSE (labelled as "validation_RMSE" for demonstration)
        "R^2": test_R2                # The R^2 score on the test set
    }

    return best_model, best_hyperparameters, performance_metrics


def tune_regression_model_hyperparameters(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Uses GridSearchCV to systematically work through multiple combinations
    of hyperparameters for a RandomForestRegressor, 
    to find the best possible parameters that minimize MSE.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        The features of the dataset.
    y : pandas.core.series.Series
        The corresponding target (Price_Night) for each feature row.

    Returns
    -------
    None
        The function prints the best hyperparameters and best score, 
        but does not return them explicitly.
    """
    random_forest_model = RandomForestRegressor()  # Instantiate a random forest regressor

    # Define a grid of possible hyperparameter values to try
    parameter_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Set up a GridSearchCV to exhaustively test all parameter combinations
    grid_search = GridSearchCV(
        estimator=random_forest_model,
        param_grid=parameter_grid,
        cv=5,  # Use 5-fold cross-validation
        scoring='neg_mean_squared_error'  # Score by negative MSE
    )
    
    # Fit the grid search on the entire dataset
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
    Saves a trained regression model, its hyperparameters, and performance metrics to disk.

    Parameters
    ----------
    folder_name : str
        The subfolder name (e.g., "SGDRegressor" or "RandomForestRegressor") 
        in which to store the model artifacts.
    best_model : Any
        The trained regression model to be saved.
    best_hyperparameters : Dict[str, Any]
        The hyperparameters used for training the best model.
    performance_metric : Dict[str, float]
        A dictionary containing relevant performance metrics (e.g., RMSE, R^2).

    Returns
    -------
    None
        This function performs I/O operations but does not return a value.
    """
    models_dir = 'modelling-airbnbs-property-listing-dataset-193/models/'  # Main models folder name

    # Get the parent directory of the current working directory
    current_dir = os.path.dirname(os.getcwd())
    
    # Construct the full path for the main models directory
    models_path = os.path.join(current_dir, models_dir)
    
    # Create the models directory if it doesn't exist
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    # Create the "regression" sub-directory within "models"
    regression_dir = 'modelling-airbnbs-property-listing-dataset-193/models/regression/'
    regression_path = os.path.join(current_dir, regression_dir)
    if not os.path.exists(regression_path):
        os.mkdir(regression_path)

    # Create the subfolder within regression for the specific model
    folder_name_dir = os.path.join(regression_path, folder_name)
    folder_name_path = os.path.join(current_dir, folder_name_dir)
    if not os.path.exists(folder_name_path):
        os.mkdir(folder_name_path)

    # Save the model object to a ".joblib" file
    joblib.dump(best_model, os.path.join(folder_name_path, "model.joblib"))

    # Save the hyperparameters to a "hyperparameters.json" file
    hyperparameters_filename = os.path.join(folder_name_path, "hyperparameters.json")
    with open(hyperparameters_filename, 'w') as json_file:
        json.dump(best_hyperparameters, json_file)
    
    # Save the metrics to a "metrics.json" file
    metrics_filename = os.path.join(folder_name_path, "metrics.json")
    with open(metrics_filename, 'w') as json_file:
        json.dump(performance_metric, json_file)


def evaluate_all_models(
    models: List[Any], 
    hyperparameters_dict: List[Dict[str, List[Any]]]
) -> None:
    """
    Evaluates and tunes multiple regression models with different sets of hyperparameters.

    Parameters
    ----------
    models : list
        A list of initialized regression model instances (e.g., SGDRegressor(), RandomForestRegressor()).
    hyperparameters_dict : list of dict
        A list where each element is a dictionary of hyperparameter search spaces 
        corresponding to each model in 'models'.

    Returns
    -------
    None
        This function prints out the best model and saves each model's artifacts 
        (trained model, hyperparameters, metrics) to disk.
    """
    data_file = "clean_tabular_data.csv"  # The CSV file containing the Airbnb dataset

    # Load and standardize the data
    X, y = import_and_standarised_data(data_file)

    # Split the data into training, validation, and testing sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    # Loop through each model, tuning them according to the hyperparameters list
    for i in range(len(models)):
        # Tune the i-th model with the i-th hyperparameter dictionary
        best_regression_model, best_hyperparameters_dict, performance_metrics = custom_tune_regression_model_hyperparameters(
            models[i], X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict[i]
        )

        # Print out which model is best and what hyperparams gave best performance
        print(best_regression_model, best_hyperparameters_dict, performance_metrics)
        
        # Convert the model name to a clean folder name, e.g. "SGDRegressor"
        folder_name = str(models[i])[0:-2]

        # Save the best model and related info to disk
        save_model(folder_name, best_regression_model, best_hyperparameters_dict, performance_metrics)


def find_best_model(models: List[Any]) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Finds the best regression model by comparing saved models in the local folders.

    Parameters
    ----------
    models : list
        A list of regression model instances used to identify subfolders for each model's artifacts.

    Returns
    -------
    best_regression_model : Any
        The best‐performing loaded regression model object.
    best_hyperparameters_dict : dict
        The hyperparameters that yielded the best model.
    best_metrics_dict : dict
        The performance metrics of the best model.
    """
    best_regression_model = None      # To store the best model found so far
    best_hyperparameters_dict = {}    # To store hyperparameters of the best model
    best_metrics_dict = {}            # To store performance metrics of the best model
    
    # Path to the "regression" folder containing each model subfolder
    regression_dir = "modelling-airbnbs-property-listing-dataset-193/models/regression"
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)

    # Iterate through each model in the provided list
    for i in range(len(models)):
        # Convert the model object into a string for subfolder naming consistency
        model_str = str(models[i])[0:-2]
        
        # Construct the path to that model’s folder
        model_dir = os.path.join(regression_path, model_str)
        
        # Load the stored model from "model.joblib"
        model = load(os.path.join(model_dir, 'model.joblib'))
        
        # Load its hyperparameters from "hyperparameters.json"
        with open(os.path.join(model_dir, 'hyperparameters.json'), 'r') as hyperparameters_path:
            hyperparameters = json.load(hyperparameters_path)
        
        # Load the metrics from "metrics.json"
        with open(os.path.join(model_dir, 'metrics.json'), 'r') as metrics_path:
            metrics_data = json.load(metrics_path)

        # Compare the R^2 of this model to the best model found so far
        if best_regression_model is None or metrics_data.get("R^2") > best_metrics_dict.get("R^2", -float('inf')):
            best_regression_model = model
            best_hyperparameters_dict = hyperparameters
            best_metrics_dict = metrics_data

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict


# Define a list of different regression model instances to evaluate
models = [
    SGDRegressor(),           # Linear model with stochastic gradient descent
    DecisionTreeRegressor(),  # Single decision tree regressor
    RandomForestRegressor(),  # Ensemble of decision trees
    GradientBoostingRegressor()  # Gradient boosting ensemble of decision trees
]

# Define a corresponding list of hyperparameter dictionaries
hyperparameters_dict = [
    {  # Hyperparameters for SGDRegressor
        'loss': ['squared_error', 'huber', 'squared_epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.00001, 0.0001, 0.000001],
        'l1_ratio': [0.15, 0.2],
        'fit_intercept': [True, False],
        'max_iter': [10000, 40000, 70000, 1000000, 10000000],
        'tol': [1e-3, 1e-4, 1e-5],
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
        'learning_rate': [0.1, 0.2, 0.5],
        'n_estimators': [50, 100, 200],
        'criterion': ['squared_error', 'friedman_mse'],
        'max_features': [10],
    }
]

if __name__ == "__main__":
    # Evaluate all models by tuning them on the data and saving their best versions
    evaluate_all_models(models, hyperparameters_dict)

    # Identify which model among the saved ones is truly the best based on R^2
    best_regression_model, best_hyperparameters_dict, best_metrics_dict = find_best_model(models)
    
    # Print out information about the best model
    print("Best Regression Model:")
    print(best_regression_model)              # The actual model instance
    print("Hyperparameters:")
    print(best_hyperparameters_dict)          # The hyperparams dictionary for this model
    print("Metrics:")
    print(best_metrics_dict)                  # The performance metrics (e.g., R^2, RMSE)