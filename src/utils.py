import os

import pandas as pd
from flex_infer import GenerationParams


def get_generation_params(
    temp: int, seed: int = 42, max_tokens: int = 32
) -> GenerationParams:
    """Get the generation parameters for the model.

    Args:
        temp (int): Temperature for sampling.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 32.

    Returns:
        GenerationParams: Generation parameters object.
    """
    return GenerationParams(
        temperature=temp,
        seed=seed,
        max_tokens=max_tokens,
    )


def save_output(
    df: pd.DataFrame, model_name: str, path: str = "./data/labeled_data_{}.parquet"
) -> None:
    """Save the labeled data to a parquet file.

    Args:
        df (pd.DataFrame): DataFrame containing the data to save.
        model_name (str): Model name to use in the filename.
        path (str, optional): File path template for saving. Defaults to "./data/labeled_data_{}.parquet".
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
    """Reads a single dataset from a hardcoded directory. The function ensures there is only one dataset
    (CSV or Parquet) and raises an error if multiple dataset files are found.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        ValueError: If more than one dataset file is found.
    """
    dir_path = "./data/input"

    all_files = os.listdir(dir_path)

    dataset_files = [
        f for f in all_files if f.endswith(".csv") or f.endswith(".parquet")
    ]

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
