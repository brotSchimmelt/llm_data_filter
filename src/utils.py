import os
from typing import List

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


def get_prompts(df: pd.DataFrame, template: str) -> List[str]:
    """Generate prompts based on the DataFrame and a template.

    Args:
        df (pd.DataFrame): Input DataFrame with revision data.
        template (str): Template string for generating prompts.

    Returns:
        List[str]: List of formatted prompts.
    """
    before_revisions = df["before_revision"].tolist()
    after_revisions = df["after_revision"].tolist()

    prompts = []
    for before, after in zip(before_revisions, after_revisions):
        prompt = template.format(before_revision=before, after_revision=after)
        prompts.append(prompt)

    return prompts


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


def read_data(path: str = "./data/new_dataset.parquet") -> pd.DataFrame:
    """Read dataset from a given file path.

    Args:
        path (str, optional): Path to the dataset file. Defaults to "./data/new_dataset.parquet".

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    print(f"Reading data from {path} ...")

    df = pd.read_parquet(path)
    print("len(df)", len(df))
    return df
