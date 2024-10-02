"""Module to enforce column schemas in pandas DataFrames."""

import re
from zoneinfo import ZoneInfo
import pandas as pd


def enforce_schema(
    data: pd.DataFrame,
    schema: pd.DataFrame,
    keep_extra_columns: bool = False
) -> pd.DataFrame:
    """Enforce presence and data types of DataFrame columns.

    This function ensures that required columns are present, adds missing optional columns,
    enforces the specified data types, and orders the columns as defined by the schema.

    Existing required and optional columns are converted to specified dtypes,
    leading to an error if type conversion fails.

    Args:
        data (pd.DataFrame | None): The DataFrame to validate and transform.
        schema (pd.DataFrame): A DataFrame defining the schema. It must include:
            - "column": Name of the column.
            - "dtype": Expected data type (e.g., "int64", "float64", "datetime64[ns]").
            - "mandatory" (optional): Boolean flag indicating if the column is required (True)
              or optional (False). If missing, all columns are considered mandatory.
        keep_extra_columns (bool): If True, columns not listed in the schema are retained.
                                   If False, only columns defined in the schema will be kept.

    Returns:
        pd.DataFrame: Transformed DataFrame adhering to the schema.

    Raises:
        ValueError: If the input DataFrame is missing required columns, or the schema has missing
                    or invalid dtype values.
        TypeError: If the input data is not a pandas DataFrame or None,
                   or if dtype conversion fails.

    Example:
        >>> PRICE_SCHEMA_CSV = \"\"\"
        ... column,             dtype,                mandatory
        ... ticker,             string[python],       True
        ... price,              Float64,              True
        ... \"\"\"
        >>> PRICE_SCHEMA = pd.read_csv(StringIO(PRICE_SCHEMA_CSV), skipinitialspace=True)
        >>> df = pd.DataFrame({'ticker': ['AAPL', 'GOOGL'], 'price': [150.0, 2800.0]})
        >>> enforce_schema(df, PRICE_SCHEMA)
           ticker  price
        0    AAPL  150.0
        1   GOOGL  2800.0
    """

    required_schema_columns = {"column", "dtype"}

    if not isinstance(schema, pd.DataFrame):
        raise TypeError("Schema must be a pandas DataFrame.")
    if not required_schema_columns.issubset(schema.columns):
        raise ValueError(f"Schema is missing required columns: {required_schema_columns}")
    if not isinstance(data, pd.DataFrame) and data is not None:
        raise TypeError("Data must be a pandas DataFrame or None.")

    # Default all columns to mandatory if 'mandatory' column is not present
    if "mandatory" not in schema.columns:
        schema["mandatory"] = True

    if data is None:
        data = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in zip(schema["column"], schema["dtype"])}
        )
    else:
        if data.empty:
            for col, dtype in zip(schema["column"], schema["dtype"]):
                data[col] = pd.Series(dtype=dtype)

        missing_cols = set(schema.loc[schema["mandatory"], "column"]).difference(data.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

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

        if not keep_extra_columns:
            data = data[[col for col in data.columns if col in schema["column"].to_list()]]
        data = data[
            schema["column"].to_list()
            + data.columns[~data.columns.isin(schema["column"].to_list())].tolist()
        ]

    return data
