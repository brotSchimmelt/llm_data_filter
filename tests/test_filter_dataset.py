import pandas as pd
import pytest

from filter_dataset import get_prompts


def test_get_prompts_normal_case():
    data = {
        "col_before": ["This is the old text", "Here is the previous version"],
        "col_after": ["This is the new text", "Here is the updated version"],
    }
    df = pd.DataFrame(data)

    columns_to_use = ["col_before", "col_after"]

    template = "Before: '{}'. After: '{}'."

    expected_prompts = [
        "Before: 'This is the old text'. After: 'This is the new text'.",
        "Before: 'Here is the previous version'. After: 'Here is the updated version'.",
    ]

    prompts = get_prompts(df, columns_to_use, template)

    assert prompts == expected_prompts


def test_get_prompts_error_case():
    data = {"col_before": ["This is the old text"], "col_after": ["This is the new text"]}
    df = pd.DataFrame(data)

    columns_to_use = ["col_before"]

    template = "Before: '{}'. After: '{}'."

    with pytest.raises(ValueError, match="columns_to_use should contain exactly 2 columns"):
        get_prompts(df, columns_to_use, template)


def test_get_prompts_empty_dataframe():
    df = pd.DataFrame(columns=["col_before", "col_after"])

    columns_to_use = ["col_before", "col_after"]

    template = "Before: '{}'. After: '{}'."

    prompts = get_prompts(df, columns_to_use, template)

    assert prompts == []
