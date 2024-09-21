import argparse
import os

import pandas as pd
from flex_infer import VLLM
from icecream import ic

from src.prompt_components import CLASSIFY_PROMPT, SYSTEM_PROMPT
from src.utils import (
    clean_up,
    get_generation_params,
    get_prompts,
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
        model_settings["max_model_len"] = (
            8_192  # decrease context length to fit on 1 A100 80GB
        )

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
    model: VLLM, df: pd.DataFrame, model_name: str, temp: float = 0.0
) -> pd.DataFrame:
    """Generate model predictions and append them to the DataFrame.

    Args:
        model (VLLM): Loaded model instance.
        df (pd.DataFrame): DataFrame containing input data.
        model_name (str): Name of the model used for generating output.

    Returns:
        pd.DataFrame: DataFrame with generated predictions.
    """
    generation_params = get_generation_params(temp=temp)

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


def main(args: argparse.Namespace) -> None:
    clean_up(args.model_name)

    df = read_data("./data/input/revision_dataset.parquet")

    model = load_model(args.model_name)

    output = generate_output(model, df, args.model_name, temp=0.0)

    save_output(output, args.model_name)

    print("Value_counts", output[f"{args.model_name}_prediction"].value_counts())


if __name__ == "__main__":
    ic.enable() if os.getenv("IC_DEBUG") == "True" else ic.disable()

    args = parse_arguments()
    main(args)
