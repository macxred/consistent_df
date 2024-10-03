"""Module to enforce column schemas in pandas DataFrames."""

import re
import pandas as pd
from io import StringIO
from zoneinfo import ZoneInfo


SCHEMA_CSV = """
    column,         dtype,     mandatory
    column,         str,       True
    dtype,          str,       True
    mandatory,      bool,      False
"""
SCHEMA = pd.read_csv(StringIO(SCHEMA_CSV), skipinitialspace=True)


def enforce_schema(
    data: pd.DataFrame,
    schema: pd.DataFrame,
    sort_columns: bool = False,
    keep_extra_columns: bool = False,
) -> pd.DataFrame:
    """Enforce schema and type consistency in a pandas DataFrame.

    This function ensures that a DataFrame adheres to a specified schema by
    verifying the presence of required columns, adding missing optional columns,
    and converting columns to the specified data types.

    Behavior with empty input:
    - If the input data is None, the function returns an empty DataFrame with all
    required and optional columns from the schema.
    - If the input data is an empty DataFrame, missing required columns are
    added without raising an error.

    Args:
        data (pd.DataFrame | None): The DataFrame to validate and adjust.
        schema (pd.DataFrame): A DataFrame defining the schema, with the following columns:
            - "column": Name of the column.
            - "dtype": Expected data type (e.g., "int64", "float64", "datetime64[ns]").
            - "mandatory" (optional): Boolean flag indicating whether the column
            is required (True) or optional (False). If not provided, all columns
            are considered mandatory by default.
        keep_extra_columns (bool): If True, columns that are not listed in the schema are retained.
        sort_columns (bool): If True, columns are sorted in the order of appearance in the schema.

    Returns:
        pd.DataFrame: A DataFrame that conforms to the schema.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
        TypeError: If the data or schema is not a pandas DataFrame, or if a
                data type conversion fails.

    Example:
        >>> schema_csv = \"\"\"
        ... column,         dtype,     mandatory
        ... ticker,         string,    True
        ... price,          Float64,   True
        ... \"\"\"
        >>> schema = pd.read_csv(StringIO(schema_csv), skipinitialspace=True)
        >>> data = pd.DataFrame({'ticker': ['AAPL', 'GOOGL'], 'price': [150.0, 2800.0]})
        >>> enforce_schema(data, schema)
        ticker  price
        0    AAPL  150.0
        1   GOOGL  2800.0
    """
    if not isinstance(schema, pd.DataFrame):
        raise TypeError("Schema must be a pandas DataFrame.")
    if not isinstance(data, pd.DataFrame) and data is not None:
        raise TypeError("Data must be a pandas DataFrame or None.")
    if "mandatory" not in schema.columns:
        schema["mandatory"] = True
    schema = _enforce_schema(schema, SCHEMA)
    if data is None:
        data = pd.DataFrame(columns=schema["column"])
    result = _enforce_schema(data=data, schema=schema)
    if not keep_extra_columns:
        result = result[result.columns[result.columns.isin(schema["column"])]]
    if sort_columns:
        cols = (schema["column"].to_list()
                + data.columns[~data.columns.isin(schema["column"])].to_list())
        result = result[cols]
    return result


def _enforce_schema(data: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    if not data.empty:
        missing_cols = set(schema.loc[schema["mandatory"], "column"]).difference(data.columns)
        if missing_cols:
            raise ValueError(f"Data frame is missing required columns: {missing_cols}")

    for col, dtype in zip(schema["column"], schema["dtype"]):
        if col not in data:
            data[col] = pd.Series(dtype=dtype)
        else:
            if dtype.startswith("datetime64"):
                if re.fullmatch("datetime64\\[ns\\]", dtype):
                    timezone = None
                elif re.fullmatch("datetime64\\[ns, .+\\]", dtype):
                    timezone_str = dtype.split(",")[1][:-1].strip()
                    timezone = ZoneInfo(timezone_str)
                else:
                    raise ValueError(f"Unknown datetime dtype: '{dtype}'.")
                try:
                    data[col] = pd.to_datetime(data[col])
                except pd._libs.tslibs.parsing.DateParseError as e:
                    raise type(e)(f"Failed to convert '{col}' to {dtype}: {e}")
                if data[col].dt.tz is None:
                    data[col] = data[col].dt.tz_localize(timezone)
                else:
                    data[col] = data[col].dt.tz_convert(timezone)
            else:
                try:
                    data[col] = data[col].astype(dtype)
                except (TypeError, ValueError) as e:
                    raise type(e)(f"Failed to convert '{col}' to {dtype}: {e}")
    return data
