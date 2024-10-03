"""Unit tests for enforcing DataFrame column schema with enforce_schema()."""

from zoneinfo import ZoneInfo
from consistent_df import enforce_schema
import pandas as pd
import pytest
from io import StringIO


def test_schema_is_not_dataframe():
    df = pd.DataFrame({"Column1": [1, 2]})
    schema = None
    with pytest.raises(TypeError, match="Schema must be a pandas DataFrame."):
        enforce_schema(df, schema)

    # Test with an invalid schema type
    schema = {"column": "Column1", "dtype": "int64", "mandatory": True}
    with pytest.raises(TypeError, match="Schema must be a pandas DataFrame."):
        enforce_schema(df, schema)


def test_schema_missing_required_columns():
    df = pd.DataFrame({"Column1": [1, 2]})
    schema_csv = """
        column,
        Column1,
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    with pytest.raises(ValueError, match="Schema is missing required columns"):
        enforce_schema(df, schema)


def test_schema_invalid_dtype():
    df = pd.DataFrame({"Column1": [1, 2]})
    schema_csv = """
        column,     dtype,    mandatory
        Column1,         ,    True
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    with pytest.raises(AttributeError):
        enforce_schema(df, schema)


def test_schema_without_mandatory_column():
    schema_csv = """
        column,        dtype
        ticker,        string[python]
        price,         Float64
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"ticker": ["AAPL", "GOOGL"]})

    with pytest.raises(ValueError, match="Input DataFrame is missing required columns: {'price'}"):
        enforce_schema(df, schema)

    df_with_all_columns = pd.DataFrame({
        "ticker": ["AAPL", "GOOGL"],
        "price": [150.0, 2800.0]
    })
    enforce_schema(df_with_all_columns, schema)


def test_invalid_data_type():
    schema_csv = """
        column,    dtype,                mandatory
        Column1,   int64,                True
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)

    invalid_data = {"Column1": [1, 2]}  # Using a dictionary instead of DataFrame
    with pytest.raises(TypeError, match="Data must be a pandas DataFrame or None."):
        enforce_schema(invalid_data, schema)


def test_empty_input():
    schema_csv = """
        column,             dtype,                mandatory
        Column1,            int64,                True
        Column2,            float64,              False
        Date,               datetime64[ns],       False
        Column3,            object,               False
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    result = enforce_schema(None, SAMPLE_SCHEMA)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["Column1", "Column2", "Date", "Column3"]


def test_empty_dataframe():
    schema_csv = """
        column,      dtype,    mandatory
        Column1,     int64,    True
        Column2,   float64,    False
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)

    empty_df = pd.DataFrame()
    result = enforce_schema(empty_df, schema)

    assert list(result.columns) == ["Column1", "Column2"]
    assert result["Column1"].dtype == "int64"
    assert result["Column2"].dtype == "float64"


def test_required_columns_added():
    schema_csv = """
        column,     dtype,     mandatory
        Column1,    int64,     True
        Column2,  float64,     True
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    result = enforce_schema(None, SAMPLE_SCHEMA)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["Column1", "Column2"]
    assert result["Column1"].dtype == "int64"
    assert result["Column2"].dtype == "float64"


def test_optional_columns_added():
    schema_csv = """
        column,      dtype,   mandatory
        Column1,     int64,   True
        Column2,    float64,  False
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column1": [1]})
    result = enforce_schema(data=df, schema=SAMPLE_SCHEMA)
    assert list(result.columns) == ["Column1", "Column2"]
    assert result["Column2"].dtype == "float64"


def test_missing_required_columns():
    schema_csv = """
        column,    dtype,     mandatory
        Column1,   int64,     True
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column3": ["data"]})
    with pytest.raises(ValueError):
        enforce_schema(data=df, schema=SAMPLE_SCHEMA)


def test_keep_extra_columns():
    schema_csv = """
        column,    dtype,     mandatory
        Column1,   int64,     True
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column1": [1], "Column3": ["extra"]})
    result = enforce_schema(data=df, schema=SAMPLE_SCHEMA, keep_extra_columns=False)
    assert "Column3" not in result.columns
    assert list(result.columns) == ["Column1"]

    result = enforce_schema(data=df, schema=SAMPLE_SCHEMA, keep_extra_columns=True)
    assert "Column3" in result.columns
    assert list(result.columns) == ["Column1", "Column3"]


def test_sort_columns_behavior():
    schema_csv = """
        column,             dtype,                mandatory
        ticker,             string[python],       True
        price,              float64,              True
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({
        'price': [150.0, 2800.0],
        'ticker': ['AAPL', 'GOOGL']
    })

    result_sorted = enforce_schema(df, schema, sort_columns=True)
    assert list(result_sorted.columns) == ['ticker', 'price'], (
        "Columns are not sorted correctly with sort_columns=True."
    )

    result_unsorted = enforce_schema(df, schema, sort_columns=False)
    assert list(result_unsorted.columns) == ['price', 'ticker'], (
        "Columns should not be sorted with sort_columns=False."
    )


def test_dtype_conversion():
    schema_csv = """
        column,    dtype,     mandatory
        Column1,   int64,     True
        Column2,   float64,     False
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column1": ["1", "2", "3"], "Column2": ["1.1", "2.2", "3.3"]})
    result = enforce_schema(data=df, schema=SAMPLE_SCHEMA)
    assert result["Column1"].dtype == "int64"
    assert result["Column2"].dtype == "float64"


def test_invalid_dtype_conversion():
    schema_csv = """
        column,    dtype,     mandatory
        Column1,   int64,     True
        Column2,   float64,   False
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column1": ["invalid"], "Column2": ["data"]})
    with pytest.raises(ValueError, match="^Failed to convert 'Column1' to int64:"):
        enforce_schema(data=df, schema=SAMPLE_SCHEMA)


def test_datetime_without_timezone():
    schema_csv = """
        column,    dtype,           mandatory
        Column1,   int64,           True
        Date,      datetime64[ns],  True
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column1": [1, 2], "Date": ["2021-01-01", "2022-02-02"]})
    result = enforce_schema(data=df, schema=SAMPLE_SCHEMA)
    assert pd.to_datetime("2021-01-01") in result["Date"].values
    assert result["Date"].dtype == "datetime64[ns]"


def test_datetime_with_timezone():
    schema_csv = """
        column,    dtype,                         mandatory
        Column1,   int64,                         True
        Date,      "datetime64[ns, US/Eastern]",  True
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column1": [1], "Date": ["2021-01-01T12:00:00"]})
    result = enforce_schema(data=df, schema=SAMPLE_SCHEMA)
    expected_date = pd.to_datetime("2021-01-01T12:00:00").tz_localize(ZoneInfo("US/Eastern"))
    assert expected_date == result["Date"].dt.tz_convert("US/Eastern").item()
    assert result["Date"].dtype == "datetime64[ns, US/Eastern]"


def test_unknown_datetime_dtype():
    df = pd.DataFrame({"Column1": [1], "Date": ["2021-01-01"]})
    schema_csv = """
        column,    dtype,                  mandatory
        Column1,   int64,                  True
        Date,      datetime64[unknown],    True
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)

    with pytest.raises(ValueError, match="Unknown datetime dtype: 'datetime64\\[unknown\\]'."):
        enforce_schema(df, schema)


def test_datetime_tz_convert():
    df = pd.DataFrame(
        {"Column1": [1], "Date": pd.to_datetime(["2021-01-01T12:00:00"]).tz_localize("UTC")}
    )
    schema_csv = """
        column,    dtype,                          mandatory
        Column1,   int64,                          True
        Date,      "datetime64[ns, US/Eastern]",   True
    """
    schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    result = enforce_schema(df, schema)
    expected_date = (
        pd.to_datetime("2021-01-01T12:00:00")
        .tz_localize("UTC")
        .tz_convert("US/Eastern")
    )

    assert result["Date"].dt.tz_convert("US/Eastern").item() == expected_date
    assert result["Date"].dtype == "datetime64[ns, US/Eastern]"


def test_datetime_conversion_fail():
    schema_csv = """
        column,    dtype,           mandatory
        Column1,   int64,           True
        Date,      datetime64[ns],  True
    """
    SAMPLE_SCHEMA = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
    df = pd.DataFrame({"Column1": [1], "Date": ["not a date"]})
    with pytest.raises(
        pd._libs.tslibs.parsing.DateParseError,
        match="^Failed to convert 'Date' to datetime64\\[ns\\]:",
    ):
        enforce_schema(data=df, schema=SAMPLE_SCHEMA)
