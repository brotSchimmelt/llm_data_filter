from unittest import mock

import pandas as pd
import pytest

from src.utils import (
    clean_up,
    parquet_exists,
    read_data,
    read_model_predictions,
    save_output,
)


@mock.patch("src.utils.os.listdir", return_value=["file1.parquet", "file2.csv"])
def test_parquet_exists_with_parquet(mock_listdir):
    result = parquet_exists("./data/output")
    assert result is True


@mock.patch("src.utils.os.listdir", return_value=["file1.csv", "file2.txt"])
def test_parquet_exists_without_parquet(mock_listdir):
    result = parquet_exists("./data/output")
    assert result is False


@mock.patch("src.utils.os.listdir", return_value=[])
def test_parquet_exists_empty_dir(mock_listdir):
    result = parquet_exists("./data/output")
    assert result is False


@mock.patch("src.utils.os.listdir", return_value=["file1.parquet", "file2.parquet"])
def test_parquet_exists_multiple_parquet(mock_listdir):
    result = parquet_exists("./data/output")
    assert result is True


@mock.patch("src.utils.pd.DataFrame.to_parquet")
def test_save_output(mock_to_parquet):
    df = pd.DataFrame({"col1": [1, 2, 3]})
    model_name = "test_model"

    save_output(df, model_name)

    mock_to_parquet.assert_called_once_with("./data/labeled_data_test_model.parquet", index=False)


@mock.patch("src.utils.os.remove")
def test_clean_up(mock_remove):
    model_name = "test_model"

    clean_up(model_name)

    mock_remove.assert_any_call("./data/labeled_data_test_model.parquet")
    mock_remove.assert_any_call("./data/labeled_data_test_model.csv")


@mock.patch("src.utils.os.listdir", return_value=[])
def test_read_data_no_files(mock_listdir):
    with pytest.raises(ValueError, match="No dataset file found"):
        read_data()


@mock.patch("src.utils.os.listdir", return_value=["file1.csv", "file2.parquet"])
def test_read_data_multiple_files(mock_listdir):
    with pytest.raises(ValueError, match="Multiple dataset files found"):
        read_data()


@mock.patch("src.utils.pd.read_csv")
@mock.patch("src.utils.os.listdir", return_value=["file.csv"])
def test_read_data_csv(mock_listdir, mock_read_csv):
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_read_csv.return_value = mock_df

    df = read_data()

    mock_read_csv.assert_called_once_with("./data/input/file.csv")
    assert len(df) == 3


@mock.patch("src.utils.pd.read_parquet")
@mock.patch("src.utils.os.listdir", return_value=["file.parquet"])
def test_read_data_parquet(mock_listdir, mock_read_parquet):
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_read_parquet.return_value = mock_df

    df = read_data()

    mock_read_parquet.assert_called_once_with("./data/input/file.parquet")
    assert len(df) == 3


@mock.patch("src.utils.pd.read_parquet")
@mock.patch("src.utils.os.path.exists", return_value=True)
def test_read_model_predictions_file_exists(mock_exists, mock_read_parquet):
    mock_df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_read_parquet.return_value = mock_df

    model_name = "test_model"
    result = read_model_predictions(model_name)

    mock_read_parquet.assert_called_once_with("./data/labeled_data_test_model.parquet")

    assert result.equals(mock_df)


@mock.patch("src.utils.os.path.exists", return_value=False)
def test_read_model_predictions_file_not_found(mock_exists):
    model_name = "non_existent_model"

    with pytest.raises(
        FileNotFoundError, match="Predictions file for non_existent_model not found"
    ):
        read_model_predictions(model_name)
