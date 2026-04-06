# flake8: noqa: F401

"""This package designed to provide shared DataFrame operations that can be
used across multiple modules, promoting code reuse and consistency across multiple modules."""

from .enforce_dtypes import enforce_dtypes
from .enforce_schema import enforce_schema
from .nest import nest, unnest
from .string import df_to_consistent_str
from .testing import assert_frame_equal
from .polars_utils import (  # noqa: F401
    DTYPE_MAP, WARN_PANDAS_INPUT, enforce_schema as pl_enforce_schema,
    ensure_polars, resolve_dtype, strip_whitespace, to_pandas, to_polars,
)