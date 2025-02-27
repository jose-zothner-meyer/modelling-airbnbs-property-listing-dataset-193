import pandas as pd
from typing import Tuple

def remove_rows_with_missing_ratings(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    return df


def combine_description_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Description"])
    df = df.copy()

    df["Description"] = df["Description"].apply(
        lambda x: x.replace("'About this space', ", "")
                  .replace("'', ", "")
                  .replace("[", "")
                  .replace("]", "")
                  .replace("\\n", ".      ")
                  .replace("''", "")
                  .split(" ")
    )
    df["Description"] = df["Description"].apply(lambda tokens: " ".join(tokens))
    return df


def combine_amenities_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Amenities"])
    df = df.copy()

    df["Amenities"] = df["Amenities"].apply(
        lambda x: x.replace("'What this place offers', ", "")
                  .replace("[", "")
                  .replace("]", "")
                  .replace("\\n", ".      ")
                  .replace("''", "")
                  .split(" ")
    )
    df["Amenities"] = df["Amenities"].apply(lambda tokens: " ".join(tokens))
    return df


def set_default_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[:, ["guests", "bedrooms"]] = df.loc[:, ["guests", "bedrooms"]].fillna("1")
    df.loc[:, ["beds", "bathrooms"]] = df.loc[:, ["beds", "bathrooms"]].fillna(1)

    df.loc[
        df["guests"] == "Somerford Keynes England United Kingdom", 
        "guests"
    ] = "1"

    df.loc[
        df["bedrooms"] == ("https://www.airbnb.co.uk/rooms/49009981?adults=1&category_tag=Tag%3A677&children=0&infants=0"
                           "&search_mode=flex_destinations_search&check_in=2022-04-18&check_out=2022-04-25&previous_"
                           "page_section_name=1000&federated_search_id=0b044c1c-8d17-4b03-bffb-5de13ff710bc"), 
        "bedrooms"
    ] = "1"
    return df


def clean_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = combine_amenities_strings(df)
    df = set_default_feature_values(df)
    return df


def load_airbnb(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Copy the DataFrame to avoid unexpected side effects.
    df = df.copy()

    # Drop any unwanted columns if they exist
    df = df.drop(columns=["Unnamed: 0", "Unnamed: 19"], errors="ignore")

    # Separate features from label
    features = df.drop(
        columns=[
            "Price_Night", 
            "ID", 
            "Category", 
            "Title", 
            "Description", 
            "Amenities", 
            "Location", 
            "url"
        ],
        errors="ignore"
    )

    labels = df["Price_Night"]
    return features, labels


if __name__ == '__main__':
    # 1. Read the CSV
    df: pd.DataFrame = pd.read_csv('tabular_data/listing.csv')
    
    # 2. Strip out leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # 3. Drop any 'Unnamed: 19' or other "Unnamed" columns right away
    #    (You can also drop 'Unnamed: 0' here if you like)
    df.drop(columns=['Unnamed: 0', 'Unnamed: 19'], errors='ignore', inplace=True)

    # 4. Clean the raw tabular data
    df = clean_tabular_data(df)

    # 5. (Optional) Save the cleaned data to a new CSV
    df.to_csv('clean_tabular_data.csv', index=False)

    # 6. Reload the cleaned data for any follow-up analysis or modeling
    df = pd.read_csv('clean_tabular_data.csv')

    # 7. Check unique categories if "Category" is present
    if "Category" in df.columns:
        print(df["Category"].unique())

    # 8. Split into features and labels
    features, labels = load_airbnb(df)

    # 9. Print shape checks
    print(features.head())
    print(features.shape)
    print(labels.shape)
