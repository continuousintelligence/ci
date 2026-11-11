"""Group context: what the other series of an evaluation window add to each series' forecast.

All three functions take the univariate series of one window, each as its observed history
(``histories``) and that history's epoch-ns timestamps (``stamps``), in the window's order. Every
value they return comes from those histories, and a series only pools with series whose history
ends at the same timestamp, its forecast origin, so no value after the origin is used.

- :func:`static_group_rows` gives each series past covariate rows: the mean of its static
  group's other z-scored histories, as Chronos-2's groups pool series by their metadata.
- :func:`joint_neighbour_rows` gives each series extra target rows: the series whose z-scored
  histories correlate best with its own, forecast jointly with it.
- :func:`chunk_groups` splits the window into chunks that are forecast jointly, every series a
  target, as t0-gamma's ``cross_inference`` and Chronos-2's ``cross_learning`` do on fev-bench.

The z-scores are taken on a grid of the window's last ``span`` distinct timestamps. Each row
these functions return covers its series' own history timestamps and is missing where no value
is available.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

# The timestamps that z-scores and correlations are taken over.
SPAN: Final = 2048

# Two histories correlate only over at least this many common steps.
_MIN_OVERLAP: Final = 32

# A window whose largest origin group holds more series is left without joint neighbours, whose
# correlation matrix would grow with the square of the group.
_MAX_JOINT_GROUP: Final = 8000


def _codes(values: Sequence[object]) -> NDArray[np.int64]:
    """Number each distinct value; a group's members share a number."""
    return np.unique(np.asarray(values), return_inverse=True)[1].astype(np.int64)


def _origins(stamps: Sequence[NDArray[np.int64]]) -> NDArray[np.int64]:
    """Number each series by the last timestamp of its history, its forecast origin."""
    return _codes([int(s[-1]) if s.size else -1 for s in stamps])


def _z_scores(
    histories: Sequence[NDArray[np.float64]], stamps: Sequence[NDArray[np.int64]], span: int
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Return the timestamp grid and each history z-scored onto it.

    A history without spread stays missing.
    """
    grid = np.unique(np.concatenate([s[-span:] for s in stamps]))[-span:]
    z = np.full((len(histories), grid.size), np.nan, dtype=np.float32)
    for i, (row, s) in enumerate(zip(histories, stamps, strict=True)):
        with np.errstate(all="ignore"):
            scale = np.nanstd(row)
        if not np.isfinite(scale) or scale <= 0:
            continue
        at = np.searchsorted(grid, s)
        ok = (at < grid.size) & (grid[np.minimum(at, grid.size - 1)] == s)
        z[i, at[ok]] = ((row[ok] - np.nanmean(row)) / scale).astype(np.float32)
    return grid, z


def _on_own_stamps(
    values: NDArray[np.float64], grid: NDArray[np.int64], stamps: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Read a grid row at a series' own history timestamps; missing off the grid."""
    if (
        values.shape == stamps.shape
        and np.array_equal(grid, stamps)
        and np.all(grid[1:] > grid[:-1])
    ):
        # Duplicate timestamps must retain searchsorted's first-occurrence semantics.
        return values.copy()
    at = np.searchsorted(grid, stamps)
    ok = (at < grid.size) & (grid[np.minimum(at, grid.size - 1)] == stamps)
    row = np.full(stamps.size, np.nan)
    row[ok] = values[at[ok]]
    return row


def static_group_rows(
    histories: Sequence[NDArray[np.float64]],
    stamps: Sequence[NDArray[np.int64]],
    statics: Sequence[Sequence[object]],
    columns: int,
    span: int = SPAN,
) -> list[NDArray[np.float64]] | None:
    """Return each series' static-group rows, one per grouping column, or ``None`` if none groups.

    ``statics`` holds each static column's value for every series. A column groups the window
    when it has at least two distinct values, fewer than the window's series, and a median group
    size of at least three. The ``columns`` of them with the most groups are kept, ties in column
    order. Each gives every series one row: the mean of its same-origin group's other z-scored
    histories. A missing value is a group of its own. A window of fewer than three series gets
    no rows.
    """
    if columns < 1:
        raise ValueError(f"static groups need at least one column, got {columns}")
    count = len(histories)
    if count < 3:
        return None
    groupings = []
    for values in statics:
        codes = _codes([repr(value) for value in values])
        sizes = np.bincount(codes)
        if 2 <= sizes.size < count and np.median(sizes[codes]) >= 3:
            groupings.append((sizes.size, codes))
    chosen = sorted(groupings, key=lambda grouping: -grouping[0])[:columns]
    if not chosen:
        return None
    grid, z = _z_scores(histories, stamps, span)
    seen = np.isfinite(z)
    filled = np.where(seen, z, np.float32(0.0))
    origin = _origins(stamps)
    rows: list[list[NDArray[np.float64]]] = [[] for _ in histories]
    for _, codes in chosen:
        group = _codes((codes * (origin.max() + 1) + origin).tolist())
        sums_by_group = np.zeros((group.max() + 1, grid.size), dtype=np.float32)
        counts_by_group = np.zeros_like(sums_by_group)
        np.add.at(sums_by_group, group, filled)
        np.add.at(counts_by_group, group, seen.astype(np.float32))
        sums = sums_by_group[group].astype(np.float64)
        counts = counts_by_group[group].astype(np.float64) - seen
        with np.errstate(all="ignore"):
            means = (sums - filled) / counts
        means[counts < 1] = np.nan
        for i, s in enumerate(stamps):
            rows[i].append(_on_own_stamps(means[i], grid, s))
    return [np.stack(own) for own in rows]


def joint_neighbour_rows(
    histories: Sequence[NDArray[np.float64]],
    stamps: Sequence[NDArray[np.int64]],
    size: int,
    span: int = SPAN,
) -> list[NDArray[np.float64]] | None:
    """Return each series' history followed by its ``size - 1`` neighbours', or ``None``.

    A series' neighbours are the same-origin series whose z-scored histories correlate best with
    its own over at least 32 common grid steps, each placed on the series' own history
    timestamps. A series with fewer neighbours repeats its own history in the remaining rows. A
    window of fewer than two series, or whose largest origin group holds more than 8,000, gets
    none.
    """
    if size < 2:
        raise ValueError(f"joint neighbours need a group of at least two series, got {size}")
    if len(histories) < 2:
        return None
    _, z = _z_scores(histories, stamps, span)
    seen = np.isfinite(z)
    filled = np.where(seen, z, np.float32(0.0))
    origin = _origins(stamps)
    if np.bincount(origin).max() > _MAX_JOINT_GROUP:
        return None
    neighbours: list[list[int]] = [[] for _ in histories]
    for group in range(origin.max() + 1):
        members = np.flatnonzero(origin == group)
        if members.size < 2:
            continue
        zf, sf = filled[members], seen[members].astype(np.float32)
        overlap = sf @ sf.T
        corr = (zf @ zf.T) / np.maximum(overlap, 1.0)
        corr[overlap < _MIN_OVERLAP] = -np.inf
        np.fill_diagonal(corr, -np.inf)
        for r, member in enumerate(members.tolist()):
            order = np.argsort(-corr[r], kind="stable")[: size - 1]
            neighbours[member] = [int(members[o]) for o in order if np.isfinite(corr[r, o])]
    out = []
    for i, (own, s) in enumerate(zip(histories, stamps, strict=True)):
        rows = [own] + [_on_own_stamps(histories[j], stamps[j], s) for j in neighbours[i]]
        rows += [own] * (size - len(rows))
        out.append(np.stack(rows))
    return out


def chunk_groups(origins: Sequence[int], size: int) -> list[list[int]]:
    """Split a window's series into consecutive chunks of up to ``size`` with one origin.

    ``origins`` holds each series' last history timestamp. Series keep the window's order within
    a chunk, and a chunk may hold one series.
    """
    if size < 2:
        raise ValueError(f"chunks need room for at least two series, got {size}")
    by_origin: dict[int, list[int]] = {}
    for index, origin in enumerate(origins):
        by_origin.setdefault(origin, []).append(index)
    return [
        members[start : start + size]
        for members in by_origin.values()
        for start in range(0, len(members), size)
    ]
