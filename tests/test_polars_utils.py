"""Unit tests for polars utilities in consistent_df.polars_utils."""

import datetime
import pandas as pd
import polars as pl
import pytest
from io import StringIO
from consistent_df.polars_utils import (
    enforce_schema, ensure_polars, resolve_dtype,
    strip_whitespace, to_pandas, to_polars,
)


# --- resolve_dtype ---

class TestResolveDtype:

    def test_static_mappings(self):
        assert resolve_dtype("string[python]") == pl.String
        assert resolve_dtype("datetime64[ns]") == pl.Date
        assert resolve_dtype("Int64") == pl.Int64
        assert resolve_dtype("Float64") == pl.Float64
        assert resolve_dtype("boolean") == pl.Boolean
        assert resolve_dtype("bool") == pl.Boolean
        assert resolve_dtype("Date") == pl.Date
        assert resolve_dtype("String") == pl.String

    def test_timezone_aware_datetime(self):
        result = resolve_dtype("datetime64[ns, Europe/Berlin]")
        assert result == pl.Datetime("ns", "Europe/Berlin")

    def test_timezone_aware_utc(self):
        result = resolve_dtype("datetime64[ns, UTC]")
        assert result == pl.Datetime("ns", "UTC")

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype string"):
            resolve_dtype("unknown_type")


# --- enforce_schema ---

class TestEnforceSchema:

    @staticmethod
    def _make_schema(csv_text):
        return pl.read_csv(StringIO(csv_text))

    def test_none_input(self):
        schema = self._make_schema(
            "column,dtype,mandatory\naccount,String,true\namount,Float64,true\n"
        )
        result = enforce_schema(None, schema)
        assert result.columns == ["account", "amount"]
        assert len(result) == 0
        assert result["account"].dtype == pl.String
        assert result["amount"].dtype == pl.Float64

    def test_empty_dataframe(self):
        schema = self._make_schema(
            "column,dtype,mandatory\nname,String,true\nvalue,Int64,false\n"
        )
        result = enforce_schema(pl.DataFrame(), schema)
        assert result.columns == ["name", "value"]
        assert len(result) == 0

    def test_missing_mandatory_column(self):
        schema = self._make_schema(
            "column,dtype,mandatory\naccount,String,true\namount,Float64,true\n"
        )
        df = pl.DataFrame({"account": ["A"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            enforce_schema(df, schema)

    def test_missing_optional_column_added(self):
        schema = self._make_schema(
            "column,dtype,mandatory\naccount,String,true\nnotes,String,false\n"
        )
        df = pl.DataFrame({"account": ["A", "B"]})
        result = enforce_schema(df, schema)
        assert "notes" in result.columns
        assert result["notes"].null_count() == 2

    def test_type_cast(self):
        schema = self._make_schema(
            "column,dtype,mandatory\nid,Int64,true\nvalue,Float64,true\n"
        )
        df = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
        result = enforce_schema(df, schema)
        assert result["value"].dtype == pl.Float64

    def test_string_to_boolean_cast(self):
        schema = self._make_schema(
            "column,dtype,mandatory\nflag,boolean,true\n"
        )
        df = pl.DataFrame({"flag": ["true", "false", "True", "FALSE", None]})
        result = enforce_schema(df, schema)
        assert result["flag"].dtype == pl.Boolean
        assert result["flag"].to_list() == [True, False, True, False, None]

    def test_string_to_date_cast(self):
        schema = self._make_schema(
            "column,dtype,mandatory\ndate,datetime64[ns],true\n"
        )
        df = pl.DataFrame({"date": ["2024-01-15", "2024-06-30"]})
        result = enforce_schema(df, schema)
        assert result["date"].dtype == pl.Date
        assert result["date"].to_list() == [
            datetime.date(2024, 1, 15), datetime.date(2024, 6, 30)
        ]

    def test_drop_extra_columns(self):
        schema = self._make_schema(
            "column,dtype,mandatory\naccount,String,true\n"
        )
        df = pl.DataFrame({"account": ["A"], "extra": [1]})
        result = enforce_schema(df, schema)
        assert "extra" not in result.columns

    def test_keep_extra_columns(self):
        schema = self._make_schema(
            "column,dtype,mandatory\naccount,String,true\n"
        )
        df = pl.DataFrame({"account": ["A"], "extra": [1]})
        result = enforce_schema(df, schema, keep_extra_columns=True)
        assert "extra" in result.columns

    def test_sort_columns(self):
        schema = self._make_schema(
            "column,dtype,mandatory\nfirst,String,true\nsecond,Int64,true\n"
        )
        df = pl.DataFrame({"second": [1], "first": ["A"]})
        result = enforce_schema(df, schema, sort_columns=True)
        assert result.columns == ["first", "second"]

    def test_invalid_input_type(self):
        schema = self._make_schema(
            "column,dtype,mandatory\naccount,String,true\n"
        )
        with pytest.raises(TypeError, match="Expected pl.DataFrame or None"):
            enforce_schema({"account": ["A"]}, schema)


# --- strip_whitespace ---

class TestStripWhitespace:

    def test_strip_column_names(self):
        df = pl.DataFrame({"  name  ": ["A"], " value ": [1]})
        result = strip_whitespace(df)
        assert result.columns == ["name", "value"]

    def test_strip_string_values(self):
        df = pl.DataFrame({"name": ["  Alice  ", "  Bob  "]})
        result = strip_whitespace(df)
        assert result["name"].to_list() == ["Alice", "Bob"]

    def test_empty_string_to_null(self):
        df = pl.DataFrame({"name": ["Alice", "  ", ""]})
        result = strip_whitespace(df)
        assert result["name"].to_list() == ["Alice", None, None]

    def test_non_string_columns_unchanged(self):
        df = pl.DataFrame({"name": ["Alice"], "value": [42]})
        result = strip_whitespace(df)
        assert result["value"].to_list() == [42]


# --- to_polars ---

class TestToPolars:

    def test_none_returns_none(self):
        assert to_polars(None) is None

    def test_polars_passthrough(self):
        df = pl.DataFrame({"a": [1, 2]})
        assert to_polars(df) is df

    def test_pandas_dataframe(self):
        pdf = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = to_polars(pdf)
        assert isinstance(result, pl.DataFrame)
        assert result["a"].to_list() == [1, 2]
        assert result["b"].to_list() == ["x", "y"]

    def test_pandas_preserves_datetime_as_datetime(self):
        """Generic to_polars does NOT cast Datetime→Date (pyledger-specific)."""
        pdf = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-06-15"])})
        result = to_polars(pdf)
        assert str(result["ts"].dtype).startswith("Datetime")

    def test_pandas_series_named(self):
        s = pd.Series([1, 2, 3], name="val")
        result = to_polars(s)
        assert isinstance(result, pl.DataFrame)
        assert "val" in result.columns

    def test_pandas_series_unnamed(self):
        s = pd.Series({"a": 1, "b": 2})
        result = to_polars(s)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_dict_single_row(self):
        result = to_polars({"a": 1, "b": "x"})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_dict_columnar(self):
        result = to_polars({"a": [1, 2], "b": ["x", "y"]})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_list_of_dicts(self):
        result = to_polars([{"a": 1}, {"a": 2}])
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_empty_list_returns_none(self):
        assert to_polars([]) is None

    def test_sanitizes_nat(self):
        result = to_polars({"a": pd.NaT, "b": 1})
        assert result["a"].to_list() == [None]

    def test_sanitizes_nan(self):
        result = to_polars({"a": float("nan"), "b": 1})
        assert result["a"].to_list() == [None]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Expected DataFrame or dict"):
            to_polars(42)


# --- to_pandas ---

class TestToPandas:

    def test_basic_conversion(self):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = to_pandas(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result["a"]) == [1, 2]

    def test_schema_dtype_restoration(self):
        schema = pl.DataFrame({
            "column": ["name", "value"],
            "dtype": ["string[python]", "Int64"],
            "mandatory": [True, True],
        })
        df = pl.DataFrame({"name": ["Alice"], "value": [42]})
        result = to_pandas(df, schema)
        assert result["name"].dtype == pd.StringDtype("python")
        assert result["value"].dtype == pd.Int64Dtype()

    def test_no_schema(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        result = to_pandas(df)
        assert isinstance(result, pd.DataFrame)


# --- ensure_polars ---

class TestEnsurePolars:

    def test_passes_through_polars(self):
        df = pl.DataFrame({"a": [1]})
        assert ensure_polars(df) is df

    def test_converts_pandas(self):
        pdf = pd.DataFrame({"a": [1]})
        result = ensure_polars(pdf)
        assert isinstance(result, pl.DataFrame)

    def test_none_returns_none(self):
        assert ensure_polars(None) is None
