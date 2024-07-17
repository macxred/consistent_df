"""Unit tests for enforcing DataFrame column schema with enforce_dtypes()."""

from zoneinfo import ZoneInfo
from consistent_df import enforce_dtypes
import pandas as pd
import pytest


def test_empty_input() -> None:
    """Test input is None."""
    result = enforce_dtypes(None)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == []


def test_required_columns_added() -> None:
    """Test adding required columns."""
    required_columns = {
        "Column1": "int64",
        "Column2": "float64",
    }
    result = enforce_dtypes(None, required=required_columns)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["Column1", "Column2"]
    assert result["Column1"].dtype == "int64"
    assert result["Column2"].dtype == "float64"


def test_optional_columns_added() -> None:
    """Test adding optional columns."""
    required_columns = {"Column1": "int64"}
    optional_columns = {"Column2": "float64"}
    df = pd.DataFrame({"Column1": [1]})
    result = enforce_dtypes(
        data=df,
        required=required_columns,
        optional=optional_columns,
    )
    assert list(result.columns) == ["Column1", "Column2"]
    assert result["Column2"].dtype == "float64"


def test_missing_required_columns() -> None:
    """Test exception when missing required columns."""
    df = pd.DataFrame({"Column3": ["data"]})
    required_columns = {"Column1": "int64"}
    with pytest.raises(ValueError):
        enforce_dtypes(data=df, required=required_columns)


def test_keep_extra_columns() -> None:
    """Test dropping columns not in the required or optional lists."""
    df = pd.DataFrame({"Column1": [1], "Column3": ["extra"]})
    required_columns = {"Column1": "int64"}
    result = enforce_dtypes(
        data=df,
        required=required_columns,
        keep_extra_columns=False,
    )
    assert "Column3" not in result.columns
    assert list(result.columns) == ["Column1"]
    result = enforce_dtypes(
        data=df,
        required=required_columns,
        keep_extra_columns=True,
    )
    assert "Column3" in result.columns
    assert list(result.columns) == ["Column1", "Column3"]


def test_dtype_conversion() -> None:
    """Test dtype conversion where original vector has string elements
    interpretable as floats or integers.
    """
    required_columns = {"Column1": "int64", "Column2": "float64"}
    optional_columns = {"Column3": "object"}
    df = pd.DataFrame({"Column1": ["1", "2", "3"], "Column2": ["1.1", "2.2", "3.3"]})
    result = enforce_dtypes(
        data=df,
        required=required_columns,
        optional=optional_columns,
    )
    assert result["Column1"].dtype == "int64"
    assert result["Column2"].dtype == "float64"


def test_invalid_dtype_conversion() -> None:
    """Test exception when dtype conversion is invalid."""
    df = pd.DataFrame({"Column1": ["invalid"], "Column2": ["data"]})
    required_columns = {"Column1": "int64", "Column2": "float64"}
    with pytest.raises(
        ValueError, match="^Failed to convert 'Column1' to int64:"
    ):
        enforce_dtypes(data=df, required=required_columns)


def test_datetime_without_timezone() -> None:
    """Test adding a datetime column without timezone."""
    df = pd.DataFrame({"Column1": [1, 2], "Date": ["2021-01-01", "2022-02-02"]})
    required_columns = {"Column1": "int64", "Date": "datetime64[ns]"}
    result = enforce_dtypes(data=df, required=required_columns)
    assert pd.to_datetime("2021-01-01") in result["Date"].values
    assert result["Date"].dtype == "datetime64[ns]"


def test_datetime_with_timezone() -> None:
    """Test adding a datetime column with timezone using zoneinfo."""
    df = pd.DataFrame({"Column1": [1], "Date": ["2021-01-01T12:00:00"]})
    required_columns = {"Column1": "int64", "Date": "datetime64[ns, US/Eastern]"}
    result = enforce_dtypes(data=df, required=required_columns)
    expected_date = pd.to_datetime("2021-01-01T12:00:00").tz_localize(
        ZoneInfo("US/Eastern")
    )
    assert expected_date == result["Date"].dt.tz_convert("US/Eastern").item()
    assert result["Date"].dtype == "datetime64[ns, US/Eastern]"


def test_datetime_conversion_fail() -> None:
    """Test failure in converting invalid datetime format."""
    df = pd.DataFrame({"Column1": [1], "Date": ["not a date"]})
    required_columns = {"Column1": "int64", "Date": "datetime64[ns]"}
    with pytest.raises(
        pd._libs.tslibs.parsing.DateParseError,
        match="^Failed to convert 'Date' to datetime64\\[ns\\]:",
    ):
        enforce_dtypes(data=df, required=required_columns)
