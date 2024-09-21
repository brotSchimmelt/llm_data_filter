import argparse
import os

import pandas as pd
from icecream import ic

from src.utils import clean_up, parquet_exists, read_data, read_model_predictions


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


def main(args: argparse.Namespace) -> None:
    if parquet_exists("./data/output/"):
        df = pd.read_parquet("./data/output/labeled_dataset.parquet")
    else:
        df = read_data()

    predictions_df = read_model_predictions(args.model_name)
    df[f"{args.model_name}_prediction"] = predictions_df[f"{args.model_name}_prediction"]
    ic(df.columns)

    df.to_parquet("./data/output/labeled_dataset.parquet", index=False)

    clean_up(args.model_name)

    print("Predictions have been added to the dataset.")
    print("Merged dataset saved at ./data/output/labeled_dataset.parquet")


if __name__ == "__main__":
    ic.enable() if os.getenv("IC_DEBUG") == "True" else ic.disable()

    args = parse_arguments()
    main(args)
