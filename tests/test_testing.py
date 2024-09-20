"""Unit tests for functions that help in testing."""

from consistent_df import assert_frame_equal
import pandas as pd
import pytest


def test_assert_frame_equal_basic_equality():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    assert_frame_equal(df1, df2)


def test_assert_frame_equal_different_indices():
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"A": [1, 2, 3]}, index=[3, 4, 5])
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_row_order=False)
    assert_frame_equal(df1, df2, ignore_index=True)


def test_assert_frame_equal_ignore_columns():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [10, 11, 12]})
    assert_frame_equal(df1, df2, ignore_columns=["C"])


def test_assert_frame_equal_ignore_columns_with_inequality():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [10, 11, 12]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, ignore_columns=["C"])


def test_assert_frame_equal_inequality():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_empty_dataframes():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    assert_frame_equal(df1, df2)


def test_assert_frame_equal_different_shapes():
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_ignore_row_order():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [3, 1, 2], "B": [6, 4, 5]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_row_order=False)
    assert_frame_equal(df1, df2, ignore_row_order=True)


def test_assert_frame_equal_ignore_row_order_with_data_mismatch():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [3, 2, 1], "B": [6, 5, 7]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_row_order=True)


def test_assert_frame_equal_ignore_columns_and_row_order():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [7, 8, 9], "C": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [2, 3, 1], "B": [8, 9, 7], "C": [5, 6, 4]})
    assert_frame_equal(df1, df2, ignore_columns=["C"], ignore_row_order=True)
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_columns=["B"], ignore_row_order=False)


def test_assert_frame_equal_ignore_row_order_with_duplicates():
    df1 = pd.DataFrame({"A": [1, 1, 2], "B": [4, 4, 5]})
    df2 = pd.DataFrame({"A": [2, 1, 1], "B": [5, 4, 4]})
    assert_frame_equal(df1, df2, ignore_row_order=True)


def test_assert_frame_equal_ignore_row_order_different_shapes():
    df1 = pd.DataFrame({"A": [1, 2], "B": [4, 5]})
    df2 = pd.DataFrame({"A": [2, 1, 3], "B": [5, 4, 6]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_row_order=True)


def test_assert_frame_equal_ignore_row_order_different_columns():
    df1 = pd.DataFrame({"A": [1, 2], "B": [4, 5]})
    df2 = pd.DataFrame({"A": [2, 1], "C": [5, 4]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_row_order=True)


def test_assert_frame_equal_ignore_index_and_row_order():
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"A": [3, 2, 1]}, index=[2, 1, 0])
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, ignore_row_order=False)
    assert_frame_equal(df1, df2, ignore_row_order=True)
