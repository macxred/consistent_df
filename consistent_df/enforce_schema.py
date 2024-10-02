import re
from zoneinfo import ZoneInfo
import pandas as pd


def enforce_schema(
    data: pd.DataFrame,
    schema: pd.DataFrame,
    keep_extra_columns: bool = False
) -> pd.DataFrame:
    """Ensures that a DataFrame adheres to a specified column schema, enforcing
    the presence and data types of specified columns.

    This function validates and adjusts a DataFrame to follow a predefined schema.
    The schema is provided as a separate DataFrame, where each row specifies a column name,
    its required data type, and whether it's mandatory. It guarantees that required columns
    are present, adds optional columns if missing, ensures the correct data types, and orders
    the columns as defined in the schema.

    If `data` is:
    - None: Returns an empty DataFrame with required and optional columns.
    - an empty DataFrame: Adds missing required and optional columns.
    - a DataFrame with one or more columns: Adds missing optional columns, and
      raises an exception if any required columns are missing.

    Existing required and optional columns are converted to specified dtypes,
    leading to an error if type conversion fails.

    Args:
        data (pd.DataFrame | None): The DataFrame to validate and transform.
        schema (pd.DataFrame): A DataFrame defining the schema. It must include:
            - 'column_name': Name of the column.
            - 'dtype': Expected data type (e.g., 'int64', 'float64', 'datetime64[ns]').
            - 'mandatory': Boolean flag indicating if the column
                           is required (True) or optional (False).
        keep_extra_columns (bool): If True, columns not listed in the schema are retained.
                                   If False, only columns defined in the schema will be kept.

    Returns:
        pd.DataFrame: A DataFrame with type-consistent columns and order.

    Raises:
        ValueError: If the input DataFrame is not empty and is missing any
                    required columns.
        TypeError: If the input data is not a pandas DataFrame or None, or
                   if dtype conversion fails due to incompatible types.
        ValueError: If the schema has missing or invalid dtype values.

    Example:
        >>> PRICE_SCHEMA_CSV = \"\"\"
        ... column_name,        dtype,                mandatory
        ... ticker,             string[python],       True
        ... date,               datetime64[ns],       True
        ... currency,           string[python],       True
        ... price,              Float64,              True
        ... \"\"\"
        >>> PRICE_SCHEMA = pd.read_csv(StringIO(PRICE_SCHEMA_CSV), skipinitialspace=True)
        >>> df = pd.DataFrame({'ticker': ['AAPL', 'GOOGL'], 'price': [150.0, 2800.0]})
        >>> enforce_schema(df, PRICE_SCHEMA)
           ticker  date currency  price
        0   AAPL   NaT      NaN   150.0
        1  GOOGL   NaT      NaN  2800.0
    """

    required_schema_columns = {"column_name", "dtype", "mandatory"}
    if not isinstance(schema, pd.DataFrame):
        raise TypeError("Schema must be a pandas DataFrame.")
    if not required_schema_columns.issubset(schema.columns):
        raise ValueError(f"Schema is missing required columns: {required_schema_columns}")

    schema = schema.set_index("column_name")
    required = schema.loc[schema["mandatory"], 'dtype'].to_dict()
    optional = schema.loc[~schema["mandatory"], 'dtype'].to_dict()
    all_cols = {**required, **optional}

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
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

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
            data = data[[col for col in data.columns if col in all_cols]]
        data = data[schema.index.tolist() + data.columns[~data.columns.isin(schema.index)].tolist()]

    return data
