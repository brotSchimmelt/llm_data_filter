import argparse
import os
from typing import List

import pandas as pd
from flex_infer import VLLM, GenerationParams
from icecream import ic

from prompt_components import CLASSIFY_PROMPT, SYSTEM_PROMPT


def parse_arguments() -> argparse.Namespace:
    """Simple argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="This script runs an experiment using the specified model and saves the data."
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        required=True,
        help="The model name. It is required to run the experiment and to save the data.",
    )

    return parser.parse_args()


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


def load_model(model_name: str, seed: int = 42) -> VLLM:
    """Load the specified model with optional seed.

    Args:
        model_name (str): Name of the model to load.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Raises:
        ValueError: If the model name is not supported.

    Returns:
        VLLM: Loaded model instance.
    """
    print(f"Loading model {model_name} ...")
    model_settings = {
        "seed": seed,
        "quant": None,
        "max_logprobs": 4,
        "num_gpus": 1,
    }

    if model_name == "mistral":
        model_settings["name"] = "mistral"
        model_settings["model_path"] = "../models/mistral-7b-instruct-v02"
        model_settings["prompt_format"] = "llama2"
    elif model_name == "llama-3.1":
        model_settings["name"] = "llama-3.1"
        model_settings["model_path"] = "../models/llama3_1-8b-instruct"
        model_settings["prompt_format"] = "llama3"
    elif model_name == "gemma":
        model_settings["name"] = "gemma-2"
        model_settings["model_path"] = "../models/gemma-2-9b-it"
        model_settings["prompt_format"] = "gemma"
    elif model_name == "gemma-27":
        model_settings["name"] = "gemma-2-27b"
        model_settings["model_path"] = "../models/gemma-2-27b-it"
        model_settings["prompt_format"] = "gemma"
    elif model_name == "nemo":
        model_settings["name"] = "mistral-nemo"
        model_settings["model_path"] = "../models/mistral-nemo-instruct-12b"
        model_settings["prompt_format"] = "llama2"
        model_settings["max_model_len"] = 8192

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return VLLM(**model_settings)


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


def generate_output(model: VLLM, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Generate model predictions and append them to the DataFrame.

    Args:
        model (VLLM): Loaded model instance.
        df (pd.DataFrame): DataFrame containing input data.
        model_name (str): Name of the model used for generating output.

    Returns:
        pd.DataFrame: DataFrame with generated predictions.
    """
    generation_params = get_generation_params(temp=0.0)

    prompts = get_prompts(df, template=CLASSIFY_PROMPT)

    answer_choices = ["good", "bad"]

    print(f"Generating predictions for {len(prompts)} examples ...")

    model_prediction = model.generate(
        prompts,
        generation_params,
        choices=answer_choices,
        system_prompt=SYSTEM_PROMPT,
        use_tqdm=True,
        return_type="str",
    )

    print("len(pred), len(df)", len(model_prediction), len(df))

    df[f"{model_name}_prediction"] = model_prediction

    return df


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


def main(args: argparse.Namespace) -> None:
    clean_up(args.model_name)

    df = read_data()

    model = load_model(args.model_name)

    output = generate_output(model, df, args.model_name)

    save_output(output, args.model_name)

    print("Value_counts", output[f"{args.model_name}_prediction"].value_counts())


if __name__ == "__main__":
    ic.enable() if os.getenv("IC_DEBUG") == "True" else ic.disable()

    args = parse_arguments()
    main(args)
