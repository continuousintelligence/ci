"""Tests for the group context that a window's other series give each series' forecast."""

import numpy as np
import pytest
from numpy.typing import NDArray

from continuousintelligence.ci_proto._groups import (
    chunk_groups,
    joint_neighbour_rows,
    static_group_rows,
)


def _window(
    count: int, length: int = 64, seed: int = 0
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.int64]]]:
    """``count`` random histories on one shared timestamp axis."""
    rng = np.random.default_rng(seed)
    histories = [rng.normal(size=length) * (i + 1) + i for i in range(count)]
    stamps = [np.arange(length, dtype=np.int64) * 10 for _ in range(count)]
    return histories, stamps


def _z(row: NDArray[np.float64]) -> NDArray[np.float64]:
    return (row - np.nanmean(row)) / np.nanstd(row)


def test_a_static_group_row_is_the_mean_of_the_other_members() -> None:
    """Each series gets the mean of its group's other z-scored histories."""
    histories, stamps = _window(6)
    rows = static_group_rows(histories, stamps, [["a", "a", "a", "b", "b", "b"]], 2)
    assert rows is not None
    assert [row.shape for row in rows] == [(1, 64)] * 6
    expected = (_z(histories[1]) + _z(histories[2])) / 2
    np.testing.assert_allclose(rows[0][0], expected, rtol=1e-5, atol=1e-6)
    expected = (_z(histories[3]) + _z(histories[4])) / 2
    np.testing.assert_allclose(rows[5][0], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "column",
    [
        ["a", "b", "c", "d", "e", "f"],
        ["a"] * 6,
        ["a", "a", "b", "b", "c", "c"],
    ],
)
def test_a_column_that_does_not_group_the_window_adds_nothing(column: list[str]) -> None:
    """One series per value, one group, or a median group under three gives no rows."""
    histories, stamps = _window(6)
    assert static_group_rows(histories, stamps, [column], 2) is None


def test_the_columns_with_the_most_groups_are_kept() -> None:
    """With room for one row, the finer grouping wins."""
    histories, stamps = _window(9)
    coarse = ["a"] * 5 + ["b"] * 4
    fine = ["x", "x", "x", "y", "y", "y", "z", "z", "z"]
    rows = static_group_rows(histories, stamps, [coarse, fine], 1)
    assert rows is not None
    expected = (_z(histories[1]) + _z(histories[2])) / 2
    np.testing.assert_allclose(rows[0][0], expected, rtol=1e-5, atol=1e-6)


def test_a_missing_static_value_is_a_group_of_its_own() -> None:
    """``None`` values pool with each other, not with any named group."""
    histories, stamps = _window(6)
    rows = static_group_rows(histories, stamps, [[None, None, None, "b", "b", "b"]], 1)
    assert rows is not None
    expected = (_z(histories[0]) + _z(histories[1])) / 2
    np.testing.assert_allclose(rows[2][0], expected, rtol=1e-5, atol=1e-6)


def test_a_series_pools_only_with_series_of_its_own_origin() -> None:
    """A group member whose history ends later never contributes."""
    histories, stamps = _window(6)
    stamps[2] = stamps[2] + 10
    rows = static_group_rows(histories, stamps, [["a", "a", "a", "b", "b", "b"]], 1)
    assert rows is not None
    observed = np.isfinite(rows[0][0])
    expected = _z(histories[1])
    np.testing.assert_allclose(rows[0][0][observed], expected[observed], rtol=1e-5, atol=1e-6)
    assert np.isnan(rows[2][0]).all()


def test_static_groups_need_three_series_and_a_column() -> None:
    """Two series get no rows, and a request for no column fails."""
    histories, stamps = _window(2)
    assert static_group_rows(histories, stamps, [["a", "a"]], 1) is None
    with pytest.raises(ValueError, match="at least one column"):
        static_group_rows(histories, stamps, [["a", "a"]], 0)


def test_joint_neighbours_put_the_best_correlated_series_first() -> None:
    """A series' second row is the series that tracks it, on its own timestamps."""
    histories, stamps = _window(3)
    histories[2] = histories[0] * 3.0 + 1.0 + np.random.default_rng(1).normal(size=64) * 0.01
    rows = joint_neighbour_rows(histories, stamps, 2)
    assert rows is not None
    assert [row.shape for row in rows] == [(2, 64)] * 3
    np.testing.assert_array_equal(rows[0][0], histories[0])
    np.testing.assert_array_equal(rows[0][1], histories[2])
    np.testing.assert_array_equal(rows[2][1], histories[0])


def test_a_series_without_neighbours_repeats_its_own_history() -> None:
    """Too few common steps, or another origin, leave the series alone."""
    histories, stamps = _window(3, length=40)
    stamps[1] = stamps[1] + 10
    stamps[2] = stamps[2] + 20
    rows = joint_neighbour_rows(histories, stamps, 3)
    assert rows is not None
    for own, row in zip(histories, rows, strict=True):
        np.testing.assert_array_equal(row, np.stack([own, own, own]))


def test_a_neighbour_is_missing_where_it_has_no_value() -> None:
    """A shorter neighbour is aligned on timestamps, missing before its history starts."""
    histories, stamps = _window(2, length=64)
    histories[1] = histories[0][16:] * 2.0
    stamps[1] = stamps[1][16:]
    rows = joint_neighbour_rows(histories, stamps, 2)
    assert rows is not None
    assert np.isnan(rows[0][1][:16]).all()
    np.testing.assert_array_equal(rows[0][1][16:], histories[1])


def test_joint_neighbours_need_two_series_and_a_group() -> None:
    """One series gets no rows, and a group of one fails."""
    histories, stamps = _window(1)
    assert joint_neighbour_rows(histories, stamps, 2) is None
    with pytest.raises(ValueError, match="at least two series"):
        joint_neighbour_rows(histories, stamps, 1)


@pytest.mark.parametrize(
    ("origins", "size", "groups"),
    [
        ([1, 1, 1, 2, 2, 1], 2, [[0, 1], [2, 5], [3, 4]]),
        ([1, 1, 1], 2, [[0, 1], [2]]),
        ([1, 2, 3], 4, [[0], [1], [2]]),
        ([5, 5, 5, 5], 64, [[0, 1, 2, 3]]),
    ],
)
def test_chunks_keep_window_order_within_one_origin(
    origins: list[int], size: int, groups: list[list[int]]
) -> None:
    """Chunks hold consecutive same-origin series, up to the chunk size."""
    assert chunk_groups(origins, size) == groups


def test_chunks_need_room_for_two_series() -> None:
    """A chunk size below two fails."""
    with pytest.raises(ValueError, match="at least two series"):
        chunk_groups([1, 1], 1)
