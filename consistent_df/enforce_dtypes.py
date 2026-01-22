"""Module to enforce column schemas in pandas DataFrames."""

import re
from typing import Dict
from zoneinfo import ZoneInfo
import pandas as pd


def enforce_dtypes(
    data: pd.DataFrame | None = None,
    required: Dict[str, str] | None = None,
    optional: Dict[str, str] | None = None,
    keep_extra_columns: bool = False
) -> pd.DataFrame:
    """Ensures that a DataFrame adheres to a specified column schema, enforcing
    the presence and data types of specified columns.

    If `data` is:
    - None: Returns an empty DataFrame with required and optional columns.
    - an empty DataFrame: Adds missing required and optional columns.
    - a DataFrame with one or more columns: Adds missing optional columns, and
      raises an exception if any required columns are missing.

    Existing required and optional columns are converted to specified dtypes,
    leading to an error if type conversion fails.

    Args:
        data (pd.DataFrame | None): The input DataFrame.
        required (Dict[str, str] | None): Required columns and their dtypes.
        optional (Dict[str, str] | None): Optional columns and their dtypes.
        keep_extra_columns (bool): If True, columns not listed as required or
            optional are left unchanged in the resulting DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with type-consistent columns.

    Raises:
        ValueError: If the input DataFrame is not empty and is missing any
            required columns.
        TypeError: If the input data is not a pandas DataFrame or None, or
            if dtype conversion fails due to incompatible types.

    Example:
        >>> required_columns = {'Column1': 'int64', 'Column2': 'float64'}
        >>> optional_columns = {'Column3': 'object'}
        >>> df = pd.DataFrame({'Column1': [1, 2], 'Column3': ['A', 'B']})
        >>> enforce_dtypes(df, required_columns, optional_columns)
           Column1  Column2 Column3
        0        1      NaN       A
        1        2      NaN       B
    """
    if required is None:
        required = {}
    if optional is None:
        optional = {}

    if not isinstance(data, pd.DataFrame) and data is not None:
        raise TypeError("Data must be a pandas DataFrame or None.")

    all_cols = {**required, **optional}

    if data is None:
        data = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in all_cols.items()})
    else:
        if data.empty:
            for col, dtype in all_cols.items():
                data[col] = pd.Series(dtype=dtype)

        missing_cols = [col for col in required if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Data frame is missing required columns: {missing_cols}")

        for col, dtype in all_cols.items():
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
                        data[col] = pd.to_datetime(data[col], format="mixed")
                    except pd._libs.tslibs.parsing.DateParseError as e:
                        raise type(e)(f"Failed to convert '{col}' to {dtype}: {e}")
                    if timezone is None:
                        data[col] = data[col].astype(dtype)
                    elif data[col].dt.tz is None:
                        data[col] = data[col].dt.tz_localize(timezone).astype(dtype)
                    else:
                        data[col] = data[col].dt.tz_convert(timezone).astype(dtype)
                else:
                    try:
                        data[col] = data[col].astype(dtype)
                    except (TypeError, ValueError) as e:
                        raise type(e)(f"Failed to convert '{col}' to {dtype}: {e}")

        if not keep_extra_columns:
            data = data[[col for col in data.columns if col in all_cols]]

    return data
