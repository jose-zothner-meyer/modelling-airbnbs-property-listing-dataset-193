import pandas as pd
from typing import Tuple

def remove_rows_with_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows from the DataFrame that have missing values in any of the 
    following rating columns:

    - Cleanliness_rating
    - Accuracy_rating
    - Communication_rating
    - Location_rating
    - Check-in_rating
    - Value_rating

    Args:
        df (pd.DataFrame): The original DataFrame containing Airbnb data.

    Returns:
        pd.DataFrame: A copy of the DataFrame with rows that are missing any of 
        the rating columns removed.
    """
    # Drop rows with missing values in any of the six rating columns.
    df = df.dropna(
        subset=[
            "Cleanliness_rating",
            "Accuracy_rating",
            "Communication_rating",
            "Location_rating",
            "Check-in_rating",
            "Value_rating"
        ]
    )
    # Create a copy to avoid potential chained assignment warnings.
    df = df.copy()
    return df


def combine_description_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes the 'Description' column by:
      1) Dropping any rows where 'Description' is missing (NaN).
      2) Removing or replacing unwanted substrings (e.g. '[' or '\\n').
      3) Splitting the resulting string on spaces and rejoining into a single 
         normalized string.

    Args:
        df (pd.DataFrame): The DataFrame containing a 'Description' column.

    Returns:
        pd.DataFrame: A copy of the DataFrame with a cleaned 'Description' 
        column.
    """
    # Drop rows missing 'Description'.
    df = df.dropna(subset=["Description"])
    # Copy to avoid SettingWithCopyWarning when modifying the DataFrame further.
    df = df.copy()

    # Apply a lambda to remove unwanted characters / patterns:
    # "'About this space', ", "'', ", "[", "]", and replace '\\n' with '.      '
    # Then split the string on spaces.
    df["Description"] = df["Description"].apply(
        lambda x: x.replace("'About this space', ", "")
                  .replace("'', ", "")
                  .replace("[", "")
                  .replace("]", "")
                  .replace("\\n", ".      ")
                  .replace("''", "")
                  .split(" ")
    )
    # Rejoin the splitted list back into a single normalized string.
    df["Description"] = df["Description"].apply(lambda tokens: " ".join(tokens))

    return df


def set_default_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets default values for certain columns if they are missing or contain 
    inconsistent values:
      - 'guests' and 'bedrooms' will be filled with the string '1'.
      - 'beds' and 'bathrooms' will be filled with the integer 1.
      - Corrects specific string anomalies in 'guests' and 'bedrooms'.

    Args:
        df (pd.DataFrame): The DataFrame that potentially has missing or 
                           inconsistent values.

    Returns:
        pd.DataFrame: A copy of the DataFrame with default values set and 
        anomalies corrected.
    """
    # Copy the DataFrame to avoid chained assignment issues.
    df = df.copy()

    # Fill 'guests' and 'bedrooms' with '1' where missing, 
    # because these columns are stored as strings in this dataset.
    df.loc[:, ["guests", "bedrooms"]] = df.loc[:, ["guests", "bedrooms"]].fillna("1")

    # Fill 'beds' and 'bathrooms' with integer 1 where missing.
    df.loc[:, ["beds", "bathrooms"]] = df.loc[:, ["beds", "bathrooms"]].fillna(1)

    # Fix specific known anomalies in 'guests' and 'bedrooms'.
    df.loc[
        df["guests"] == "Somerford Keynes England United Kingdom", 
        "guests"
    ] = "1"

    df.loc[
        df["bedrooms"] == ("https://www.airbnb.co.uk/rooms/49009981?adults=1&category_tag=Tag%3A677&children=0&infants=0&search_mode=flex_destinations_search&check_in=2022-04-18&check_out=2022-04-25&previous_page_section_name=1000&federated_search_id=0b044c1c-8d17-4b03-bffb-5de13ff710bc"), "bedrooms"] = "1"
    return df


def clean_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the cleaning process by calling the helper functions in sequence:
      1. remove_rows_with_missing_ratings
      2. combine_description_strings
      3. set_default_feature_values

    Args:
        df (pd.DataFrame): The initial raw DataFrame.

    Returns:
        pd.DataFrame: The fully cleaned DataFrame.
    """
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df


def load_airbnb(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drops unwanted columns, then splits the dataset into features and labels 
    for modeling. The 'Price_Night' column is used as the label.

    The columns dropped (if they exist) include:
      - 'Unnamed: 0'
      - 'Unnamed: 19'
      - 'ID'
      - 'Category'
      - 'Title'
      - 'Description'
      - 'Amenities'
      - 'Location'
      - 'url'

    Args:
        df (pd.DataFrame): The cleaned DataFrame from which columns will be 
                           dropped.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple of (features, labels) where 
        features is the remaining DataFrame after dropping columns and labels 
        is the 'Price_Night' column.
    """
    # Copy the DataFrame to avoid unexpected side effects.
    df = df.copy()

    # Safely drop columns that are not needed for modeling. 
    # 'errors="ignore"' ensures the code won't fail if the column doesn't exist.
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 19"], errors="ignore")

    # Extract feature columns by dropping label and other unnecessary columns.
    features = df.drop(
        columns=["Price_Night", "ID", "Category", "Title", "Description", 
                 "Amenities", "Location", "url"],
        errors="ignore"
    )
    
    # The label we want to predict is 'Price_Night'.
    labels = df["Price_Night"]

    return features, labels


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # 1. Read the CSV file containing the listings data.
    # --------------------------------------------------------------------------
    df: pd.DataFrame = pd.read_csv('tabular_data/listing.csv')
    
    # --------------------------------------------------------------------------
    # 2. Clean the raw tabular data by calling the orchestrating function.
    # --------------------------------------------------------------------------
    df = clean_tabular_data(df)
    
    # --------------------------------------------------------------------------
    # 3. (Optional) Save the cleaned data to a new CSV for future use.
    #    Index is set to False to avoid writing an extra index column.
    # --------------------------------------------------------------------------
    df.to_csv('clean_tabular_data.csv', index=False)
    
    # --------------------------------------------------------------------------
    # 4. Reload the cleaned data for any follow-up analysis or modeling.
    # --------------------------------------------------------------------------
    df = pd.read_csv('clean_tabular_data.csv')
    
    # --------------------------------------------------------------------------
    # 5. Print out the unique categories for inspection (if 'Category' column 
    #    exists in the data).
    # --------------------------------------------------------------------------
    if "Category" in df.columns:
        print(df["Category"].unique())
    
    # --------------------------------------------------------------------------
    # 6. Split the cleaned data into features and labels for modeling.
    # --------------------------------------------------------------------------
    features, labels = load_airbnb(df)
    
    # --------------------------------------------------------------------------
    # 7. Print feature DataFrame, shapes, and label shapes for confirmation.
    # --------------------------------------------------------------------------
    print(features)
    print(features.shape)
    print(labels.shape)
    print(labels)
    # --------------------------------------------------------------------------
    # 8. (Optional) Perform additional data processing or modeling steps.
    # --------------------------------------------------------------------------