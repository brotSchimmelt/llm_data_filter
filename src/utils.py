import os

import pandas as pd


def save_output(
    df: pd.DataFrame, model_name: str, path: str = "./data/labeled_data_{}.parquet"
) -> None:
    """Save the labeled data to a parquet file.

    Args:
        df (pd.DataFrame): DataFrame containing the data to save.
        model_name (str): Model name to use in the filename.
        path (str, optional): File path template for saving. Defaults to
            "./data/labeled_data_{}.parquet".
    """
    print("Saving labeled data ...")
    output_path = path.format(model_name)

    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df)} labeled examples to {output_path}")


def clean_up(model_name: str) -> None:
    """Clean up generated files for the given model.

    Args:
        model_name (str): Name of the model for which files should be deleted.
    """
    try:
        os.remove(f"./data/labeled_data_{model_name}.parquet")
        os.remove(f"./data/labeled_data_{model_name}.csv")
    except FileNotFoundError:
        pass


def read_data() -> pd.DataFrame:
    """Reads a single dataset from a hardcoded directory. The function ensures there is only one
    dataset (CSV or Parquet) and raises an error if multiple dataset files are found.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        ValueError: If more than one dataset file is found.
    """
    dir_path = "./data/input"

    all_files = os.listdir(dir_path)

    dataset_files = [f for f in all_files if f.endswith(".csv") or f.endswith(".parquet")]

    if len(dataset_files) > 1:
        raise ValueError(
            "Multiple dataset files found. "
            f"Only one CSV or Parquet file should be present in {dir_path}."
        )

    if len(dataset_files) == 0:
        raise ValueError(
            f"No dataset file found. Ensure there is one CSV or Parquet file in {dir_path}."
        )

    dataset_file = os.path.join(dir_path, dataset_files[0])

    print(f"Reading data from {dataset_file} ...")

    if dataset_file.endswith(".csv"):
        df = pd.read_csv(dataset_file)
    elif dataset_file.endswith(".parquet"):
        df = pd.read_parquet(dataset_file)

    print(f"Data loaded successfully, number of rows: {len(df)}")
    return df


def read_model_predictions(model_name: str, prediction_dir: str = "./data/") -> pd.DataFrame:
    """
    Reads the model predictions from a parquet file.

    Args:
        model_name (str): Name of the model.
        prediction_dir (str, optional): Path of the prediction files. Defaults to "./data/".

    Raises:
        FileNotFoundError: If the predictions file is not found.

    Returns:
        pd.DataFrame: DataFrame containing the model predictions.
    """
    file_name = f"labeled_data_{model_name}.parquet"
    path = os.path.join(prediction_dir, file_name)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file for {model_name} not found at {path}")

    return pd.read_parquet(path)


def parquet_exists(path: str = "./data/output") -> bool:
    """Checks if a Parquet file exists in the specified directory.

    Args:
        path (str): The directory to check for Parquet files. Defaults to "./data/output".

    Returns:
        bool: True if a Parquet file exists, False otherwise.
    """
    files = os.listdir(path)

    for file in files:
        if file.endswith(".parquet"):
            return True

    return False
