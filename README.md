# LLM Data Filter

A simple script to filter datasets with LLMs based on a single prompt.

For instance, the script can be used to remove nonsensical texts from large datasets, ensuring that only well-formed and correctly spelled examples are retained.

This tool can be particularly valuable in preprocessing text datasets for machine learning tasks, where maintaining high-quality input data is critical for model performance. It ensures that irrelevant, incoherent, or poorly structured texts are filtered out automatically, reducing the need for manual data cleaning.

## Dependencies

Python Version `>= 3.10`

`flex-infer` can be downloaded [here](https://github.com/brotSchimmelt/flex-infer).

```bash
pip install -r requirements.txt

pip install -e path/to/flex-infer
```

## Usage

Set the model paths in `filter_dataset.py` in the `load_model` function.

```bash
chmod +x run_script.sh

./run_script.sh <model_name>
```
