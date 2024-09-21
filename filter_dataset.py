import argparse
import os
from typing import List

import pandas as pd
from flex_infer import VLLM, GenerationParams
from icecream import ic

from src.prompt_components import CLASSIFY_PROMPT, SYSTEM_PROMPT
from src.settings import MAX_TOKENS, RANDOM_SEED
from src.utils import (
    clean_up,
    read_data,
    save_output,
)


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

    if model_name == "mistral-7b-v2":
        model_settings["name"] = "mistral-7b-v2"
        model_settings["model_path"] = "../models/mistral-7b-instruct-v02"
        model_settings["prompt_format"] = "llama2"
    elif model_name == "llama-3.1-8b":
        model_settings["name"] = "llama-3.1-8b"
        model_settings["model_path"] = "../models/llama3_1-8b-instruct"
        model_settings["prompt_format"] = "llama3"
    elif model_name == "gemma-2-9b":
        model_settings["name"] = "gemma-2-9b"
        model_settings["model_path"] = "../models/gemma-2-9b-it"
        model_settings["prompt_format"] = "gemma"
    elif model_name == "gemma-2-27b":
        model_settings["name"] = "gemma-2-27b"
        model_settings["model_path"] = "../models/gemma-2-27b-it"
        model_settings["prompt_format"] = "gemma"
    elif model_name == "gemma-2b":
        model_settings["name"] = "gemma-2b"
        model_settings["model_path"] = "../models/gemma-2b-it"
        model_settings["prompt_format"] = "gemma"
    elif model_name == "mistral-nemo-12b":
        model_settings["name"] = "mistral-nemo-12b"
        model_settings["model_path"] = "../models/mistral-nemo-instruct-12b"
        model_settings["prompt_format"] = "llama2"
        model_settings["max_model_len"] = 8_192  # decrease context length to fit on 1 A100 80GB
    elif model_name == "phi3-mini-4k":
        model_settings["name"] = "phi3-mini-4k"
        model_settings["model_path"] = "../models/phi3-mini-4k-instruct"
        model_settings["prompt_format"] = "phi"

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return VLLM(**model_settings)


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


def generate_output(
    model_name: str,
    df: pd.DataFrame,
    columns_to_use: List[str],
    temp: float = 0.0,
    template: str = CLASSIFY_PROMPT,
) -> pd.DataFrame:
    """Generate model predictions and append them to the DataFrame.

    Args:
        model_name (str): Name of the model used for generating output.
        df (pd.DataFrame): DataFrame with data.
        columns_to_use (List[str]): Columns to use for generating prompts.
        temp (float, optional): Temperature for generation. Defaults to 0.0.
        template (str, optional): Template for generating prompts. Defaults to CLASSIFY_PROMPT.

    Returns:
        pd.DataFrame: DataFrame with generated predictions.
    """
    model = load_model(model_name)

    generation_params = GenerationParams(
        temperature=temp,
        seed=RANDOM_SEED,
        max_tokens=MAX_TOKENS,
    )

    prompts = get_prompts(df, columns_to_use, template=template)

    answer_choices = ["good", "bad"]

    print(f"Generating predictions for {len(prompts)} examples ...")

    model_prediction = model.generate(
        prompts,
        generation_params,
        choices=answer_choices,
        system_prompt=SYSTEM_PROMPT,
        use_tqdm=USE_TQDM,
        return_type="str",
    )

    print("len(pred), len(df)", len(model_prediction), len(df))

    df[f"{model_name}_prediction"] = model_prediction

    return df


def get_prompts(df: pd.DataFrame, columns_to_use: List[str], template: str) -> List[str]:
    """Generate prompts based on the DataFrame and a template.

    Args:
        df (pd.DataFrame): Input DataFrame with revision data.
        columns_to_use (List[str]): Columns to use for generating prompts.
        template (str): Template string for generating prompts.

    Returns:
        List[str]: List of formatted prompts.
    """
    if len(columns_to_use) != 2:
        raise ValueError("columns_to_use should contain exactly 2 columns")

    text_1 = df[columns_to_use[0]].tolist()
    text_2 = df[columns_to_use[1]].tolist()

    prompts = []
    for before, after in zip(text_1, text_2):
        prompt = template.format(before, after)
        prompts.append(prompt)

    return prompts


def main(args: argparse.Namespace) -> None:
    clean_up(args.model_name)

    # reads the dataset from ./data/input
    df = read_data()

    output = generate_output(args.model_name, df, COLUMNS, temp=0.0, template=PROMPT_TEMPLATE)

    save_output(output, args.model_name)

    print("Value_counts", output[f"{args.model_name}_prediction"].value_counts())


if __name__ == "__main__":
    ic.enable() if os.getenv("IC_DEBUG") == "True" else ic.disable()
    USE_TQDM = os.getenv("USE_TQDM", "False") == "True"

    # set the columns to use for generating prompts
    COLUMNS = ["before_revision", "after_revision"]

    # set the prompt template
    PROMPT_TEMPLATE = CLASSIFY_PROMPT

    args = parse_arguments()
    main(args)
