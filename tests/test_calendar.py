"""Tests for the calendar rows that evaluation windows get as known covariates."""

import numpy as np
import pytest

from continuousintelligence.ci_proto._calendar import (
    calendar_channels,
    frequency_step_ns,
)

HOUR = 3_600 * 10**9
DAY = 24 * HOUR


@pytest.mark.parametrize(
    ("freq", "step"),
    [
        ("15min", 15 * 60 * 10**9),
        ("15T", 15 * 60 * 10**9),
        ("h", HOUR),
        ("H", HOUR),
        ("D", DAY),
        ("W-SUN", 7 * DAY),
        ("2W", 0),
        ("ME", 30 * DAY),
        ("M", 30 * DAY),
        ("MS", 30 * DAY),
        ("QE-DEC", 91 * DAY),
        ("Q", 91 * DAY),
        ("YE-DEC", 0),
        ("B", 0),
        ("unknown", 0),
        ("", 0),
        (None, 0),
    ],
)
def test_frequency_step_counts_months_and_quarters_in_days(freq: str | None, step: int) -> None:
    """Steps come out in nanoseconds; a month is 30 days, a quarter 91, and other steps none."""
    assert frequency_step_ns(freq) == step


def test_daily_calendar_includes_movable_holidays() -> None:
    """Daily rows identify weekdays, leap-year day numbers and the release's holiday set."""
    stamps = np.arange("2024-03-29", "2024-04-03", dtype="datetime64[D]").astype("datetime64[ns]")
    rows = calendar_channels(stamps.astype(np.int64), DAY)
    np.testing.assert_array_equal(rows[0], [4, 5, 6, 0, 1])
    np.testing.assert_array_equal(rows[1], [88, 89, 90, 91, 92])
    np.testing.assert_array_equal(rows[2], [1, 0, 0, 1, 0])


@pytest.mark.parametrize(
    "freq, rows", [("h", 4), ("D", 3), ("W", 2), ("ME", 2), ("Q", 2), ("2D", 0)]
)
def test_calendar_channel_count_follows_frequency(freq: str, rows: int) -> None:
    """The supplied sampling frequency selects the calendar channels used in training."""
    stamps = np.array(["2024-01-01", "2024-12-31"], dtype="datetime64[ns]").astype(np.int64)
    values = calendar_channels(stamps, frequency_step_ns(freq))
    assert values.shape == (rows, 2)
    assert values.dtype == np.float32


@pytest.mark.parametrize(
    "freq, dates, expected",
    [
        (
            "ME",
            ["2023-02-28", "2024-02-29", "2024-03-31", "2024-04-30", "2024-12-31"],
            [20, 21, 20, 21, 18],
        ),
        ("QE", ["2024-01-31", "2024-03-31", "2024-04-30", "2024-12-31"], [63, 63, 64, 61]),
    ],
)
def test_calendar_counts_business_days_in_the_whole_period(
    freq: str, dates: list[str], expected: list[int]
) -> None:
    """Monthly and quarterly rows include leap days and exclude weekend and holiday dates."""
    stamps = np.array(dates, dtype="datetime64[ns]").astype(np.int64)
    step = frequency_step_ns(freq)
    values = calendar_channels(stamps, step)
    np.testing.assert_array_equal(values[0], [1, 1, 2, 3, 11] if freq == "ME" else [0, 0, 1, 3])
    np.testing.assert_array_equal(values[-1], expected)
    assert values.dtype == np.float32
