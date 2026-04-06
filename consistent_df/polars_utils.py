"""Polars utilities for schema enforcement and DataFrame conversion.

Schemas are polars DataFrames with columns: column, dtype, mandatory.
The dtype column contains string names mapped to polars types via DTYPE_MAP.
"""

import re
import warnings
import polars as pl


# Map dtype strings (from CSV schema definitions) to polars types.
# Includes pandas-era names for backward compatibility with existing schemas.
DTYPE_MAP = {
    # Pandas-era names (used in existing schema CSVs)
    "string[python]": pl.String,
    "string": pl.String,
    "str": pl.String,
    "datetime64[ns]": pl.Date,
    "boolean": pl.Boolean,
    "bool": pl.Boolean,
    "object": pl.Object,
    "int": pl.Int64,
    "int64": pl.Int64,
    "Int64": pl.Int64,
    "float": pl.Float64,
    "float64": pl.Float64,
    "Float64": pl.Float64,
    # Polars-native names
    "String": pl.String,
    "Date": pl.Date,
    "Boolean": pl.Boolean,
}

_DATETIME_TZ_RE = re.compile(r"datetime64\[ns,\s*(.+)\]")


def resolve_dtype(dtype_str: str) -> pl.DataType:
    """Resolve a dtype string to a polars DataType.

    Checks DTYPE_MAP for static mappings, then handles timezone-aware
    datetime strings like ``datetime64[ns, Europe/Berlin]``.

    Args:
        dtype_str: Dtype string from a schema definition.

    Returns:
        Corresponding polars DataType.

    Raises:
        ValueError: If the dtype string is not recognized.
    """
    if dtype_str in DTYPE_MAP:
        return DTYPE_MAP[dtype_str]

    m = _DATETIME_TZ_RE.fullmatch(dtype_str)
    if m:
        return pl.Datetime("ns", m.group(1))

    raise ValueError(f"Unknown dtype string: '{dtype_str}'")


def enforce_schema(
    df: pl.DataFrame | None,
    schema: pl.DataFrame,
    keep_extra_columns: bool = False,
    sort_columns: bool = False,
) -> pl.DataFrame:
    """Validate and conform a polars DataFrame to a schema.

    Checks mandatory columns, adds missing optional columns as typed nulls,
    and casts dtypes via polars ``cast(strict=False)``. Incompatible values
    become null.

    Args:
        df: DataFrame to validate. None returns an empty DataFrame.
        schema: Schema with columns: column, dtype, mandatory.
        keep_extra_columns: If True, retain columns not in the schema.
        sort_columns: If True, reorder columns to match schema order.

    Returns:
        DataFrame conforming to the schema.

    Raises:
        TypeError: If df is not a polars DataFrame or None.
        ValueError: If mandatory columns are missing.
    """
    columns = schema["column"].to_list()
    dtypes = {col: resolve_dtype(ds) for col, ds in schema.select("column", "dtype").iter_rows()}
    mandatory = schema.filter(pl.col("mandatory"))["column"].to_list()

    if df is None or (len(df) == 0 and len(df.columns) == 0):
        return pl.DataFrame(schema=dtypes)

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame or None, got {type(df).__name__}")

    # Validate mandatory columns present
    if len(df) > 0:
        missing = set(mandatory) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Add missing optional columns as typed nulls
    df = df.with_columns(**{
        col: pl.lit(None).cast(dtypes[col])
        for col in columns if col not in df.columns
    })

    # Cast types: batch simple casts via df.cast(), handle String→Boolean
    # and String→Date specially (polars can't cast these directly)
    special_casts = {}
    cast_map = {}
    for col, dtype in dtypes.items():
        if col not in df.columns or df[col].dtype == dtype:
            continue
        if dtype == pl.Object or isinstance(dtype, pl.List):
            continue

        if dtype == pl.Boolean and df[col].dtype == pl.String:
            stripped = pl.col(col).str.strip_chars().str.to_lowercase()
            special_casts[col] = (
                pl.when(stripped == "true").then(True)
                .when(stripped == "false").then(False)
                .otherwise(None)
            )
        elif dtype == pl.Date and df[col].dtype == pl.String:
            c = pl.col(col)
            special_casts[col] = pl.coalesce(
                c.str.to_date("%Y-%m-%d", strict=False),
                c.str.to_date("%Y-%m", strict=False),
                c.str.to_date("%Y", strict=False),
            )
        else:
            cast_map[col] = dtype

    if special_casts:
        df = df.with_columns(**special_casts)
    if cast_map:
        df = df.cast(cast_map, strict=False)

    # Column selection: drop extras and/or reorder to schema order
    schema_col_set = set(columns)

    if not keep_extra_columns:
        df = df.drop([c for c in df.columns if c not in schema_col_set])
    if sort_columns:
        ordered = [c for c in columns if c in df.columns]
        extra = [c for c in df.columns if c not in schema_col_set]
        df = df.select(ordered + extra)

    return df


def strip_whitespace(df: pl.DataFrame) -> pl.DataFrame:
    """Strip whitespace from column names and string values.

    Converts empty strings to null. Useful at CSV read boundaries
    where fixed-width formatting adds padding.

    Args:
        df: DataFrame to clean.

    Returns:
        DataFrame with stripped names and values.
    """
    rename_map = {c: c.strip() for c in df.columns if c != c.strip()}

    if rename_map:
        df = df.rename(rename_map)

    str_cols = [c for c in df.columns if df[c].dtype == pl.String]

    if str_cols:
        df = df.with_columns(**{
            c: pl.col(c).str.strip_chars().replace("", None)
            for c in str_cols
        })

    return df


# Migration flag: set to True to emit DeprecationWarning when callers pass
# pandas DataFrames to public methods. Flip before changing pandas=True
# defaults, so external consumers have time to migrate their input data.
WARN_PANDAS_INPUT = False


def ensure_polars(data, func_name: str = ""):
    """Convert input to polars DataFrame, warning on pandas input if enabled.

    Use at the entry of public methods that accept DataFrames from external
    callers. When ``WARN_PANDAS_INPUT`` is True, emits DeprecationWarning
    for pandas input so callers can migrate.

    Args:
        data: Input data in any supported format.
        func_name: Function name for the deprecation warning message.

    Returns:
        Normalized polars DataFrame, or None if data is None.
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame) and WARN_PANDAS_INPUT:
        warnings.warn(
            f"{func_name}: pandas DataFrame input is deprecated, use polars",
            DeprecationWarning, stacklevel=3
        )

    return to_polars(data)


def to_polars(data) -> pl.DataFrame | None:
    """Normalize input to a polars DataFrame.

    Accepts None, pl.DataFrame, pd.DataFrame, pd.Series, dict, or list
    of dicts. Returns None for None or empty list.

    Args:
        data: Input data in any supported format.

    Returns:
        Normalized polars DataFrame, or None.

    Raises:
        TypeError: If the input type is not supported.
    """
    if data is None:
        return None
    if isinstance(data, pl.DataFrame):
        return data

    import pandas as pd
    import numpy as np

    def _sanitize(d):
        return {
            k: (None if v is pd.NaT or v is pd.NA
                or (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in d.items()
        }

    if isinstance(data, pd.DataFrame):
        try:
            return pl.from_pandas(data)
        except OverflowError:
            # Object columns with values that overflow Int64
            data = data.copy()
            for col in data.columns:
                if data[col].dtype == object:
                    data[col] = data[col].astype(pd.StringDtype())

            return pl.from_pandas(data)

    if isinstance(data, pd.Series):
        if data.name is not None:
            return pl.from_pandas(data.to_frame().reset_index(drop=True))

        return pl.DataFrame([_sanitize(data.to_dict())])

    if isinstance(data, dict):
        data = _sanitize(data)
        if any(isinstance(v, (list, tuple)) for v in data.values()):
            return pl.DataFrame(data)

        return pl.DataFrame([data])

    if isinstance(data, list):
        if len(data) == 0:
            return None
        data = [
            _sanitize(item.to_dict()) if isinstance(item, pd.Series)
            else (_sanitize(item) if isinstance(item, dict) else item)
            for item in data
        ]

        return pl.DataFrame(data)

    raise TypeError(f"Expected DataFrame or dict, got {type(data).__name__}")


def to_pandas(df: pl.DataFrame, schema: pl.DataFrame = None):
    """Convert a polars DataFrame to pandas with correct nullable dtypes.

    Reads dtype strings from the schema to restore exact pandas types
    (e.g., ``string[python]`` → ``pd.StringDtype("python")``).

    Args:
        df: Polars DataFrame to convert.
        schema: Schema with column and dtype columns.
            If None, uses default polars-to-pandas conversion.

    Returns:
        Pandas DataFrame with correct nullable dtypes.
    """
    result = df.to_pandas()

    if schema is not None:
        dtype_map = dict(schema.select("column", "dtype").iter_rows())
        for col in result.columns:
            dtype_str = dtype_map.get(col)
            if dtype_str:
                try:
                    result[col] = result[col].astype(dtype_str)
                except (ValueError, TypeError):
                    warnings.warn(
                        f"to_pandas: failed to cast column '{col}' to {dtype_str}"
                    )

    return result
