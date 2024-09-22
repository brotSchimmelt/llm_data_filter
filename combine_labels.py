import os
from typing import List

import pandas as pd


def load_labeled_data(file_path: str = "data/output/labeled_dataset.parquet") -> pd.DataFrame:
    """
    Loads labeled data from a Parquet file.

    Args:
        file_path (str, optional): The file path to the labeled dataset.
            Defaults to "data/output/labeled_dataset.parquet".

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No dataset found at {file_path}")

    return pd.read_parquet(file_path)


def find_label_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies the columns in the DataFrame that contain labels ("good" or "bad") in the first row.

    Args:
        df (pd.DataFrame): The input DataFrame with data.

    Raises:
        ValueError: If the DataFrame is empty.

    Returns:
        List[str]: A list of column names where the first row contains "good" or "bad".
    """
    if len(df) == 0:
        raise ValueError("No data found in the dataset.")

    labels = ["good", "bad"]
    first_row = df.iloc[0]

    columns = []
    for col in df.columns:
        if first_row[col] in labels:
            columns.append(col)

    return columns


def main() -> None:
    dataset = load_labeled_data()

    label_columns = find_label_columns(dataset)

    # create a mask where any column in label_columns contains 'bad'
    bad_mask = dataset[label_columns].eq("bad").any(axis=1)
    final_label = ["bad" if is_bad else "good" for is_bad in bad_mask]

    dataset["quality_label"] = final_label
    dataset.drop(columns=label_columns, inplace=True)

    dataset.to_parquet("data/output/combined_dataset.parquet")


if __name__ == "__main__":
    main()
    print("Done.")
