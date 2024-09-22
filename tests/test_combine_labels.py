from unittest import mock

import pandas as pd
import pytest

from combine_labels import find_label_columns, load_labeled_data


@mock.patch("combine_labels.pd.read_parquet")
@mock.patch("combine_labels.os.path.exists", return_value=True)
def test_load_labeled_data_file_exists(mock_exists, mock_read_parquet):
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_read_parquet.return_value = mock_df

    df = load_labeled_data("data/output/labeled_dataset.parquet")

    mock_exists.assert_called_once_with("data/output/labeled_dataset.parquet")
    mock_read_parquet.assert_called_once_with("data/output/labeled_dataset.parquet")

    assert df.equals(mock_df)


@mock.patch("combine_labels.os.path.exists", return_value=False)
def test_load_labeled_data_file_not_found(mock_exists):
    with pytest.raises(
        FileNotFoundError, match="No dataset found at data/output/labeled_dataset.parquet"
    ):
        load_labeled_data("data/output/labeled_dataset.parquet")


def test_find_label_columns_valid():
    data = {
        "col1": ["good", "some_value1", "some_value2"],
        "col2": ["bad", "some_value3", "some_value4"],
        "col3": ["neutral", "some_value5", "some_value6"],
    }
    df = pd.DataFrame(data)

    label_columns = find_label_columns(df)

    assert label_columns == ["col1", "col2"]


def test_find_label_columns_no_labels():
    data = {
        "col1": ["neutral", "some_value1", "some_value2"],
        "col2": ["neutral", "some_value3", "some_value4"],
    }
    df = pd.DataFrame(data)

    label_columns = find_label_columns(df)

    assert label_columns == []


def test_find_label_columns_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="No data found in the dataset."):
        find_label_columns(df)
