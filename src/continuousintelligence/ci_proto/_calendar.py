"""The calendar covariates used by ci-proto training and inference."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_DAY_NS = 86_400 * 10**9
_HOUR_NS = 3_600 * 10**9
_HOLIDAY_DATES = ((1, 1), (12, 24), (12, 25), (12, 26), (12, 31))
_LEGACY_ALIASES = {
    "T": "min",
    "H": "h",
    "S": "s",
    "L": "ms",
    "U": "us",
    "N": "ns",
    "M": "ME",
    "Q": "QE",
    "A": "YE",
    "Y": "YE",
}


def frequency_step_ns(freq: str | None) -> int:
    """Return the step of a pandas frequency in nanoseconds, or ``0`` for none or an irregular one.

    A month counts 30 days and a quarter 91, so they select the monthly and quarterly calendar
    rows. A step of several weeks, months or quarters returns ``0``.
    """
    if not freq:
        return 0
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        match = re.fullmatch(r"(\d*)([A-Za-z]+)(-.*)?", freq)
        if match is None or match[2] not in _LEGACY_ALIASES:
            return 0
        offset = pd.tseries.frequencies.to_offset(
            match[1] + _LEGACY_ALIASES[match[2]] + (match[3] or "")
        )
    if isinstance(offset, pd.offsets.Day):
        return int(offset.n) * _DAY_NS
    for kind, days in (
        (pd.offsets.Week, 7),
        (pd.offsets.MonthBegin, 30),
        (pd.offsets.MonthEnd, 30),
        (pd.offsets.QuarterBegin, 91),
        (pd.offsets.QuarterEnd, 91),
    ):
        if isinstance(offset, kind):
            return days * _DAY_NS if offset.n == 1 else 0
    if isinstance(offset, pd.offsets.Tick):
        return int(pd.Timedelta(offset).value)
    return 0


def _easter_day(year: int) -> int:
    """Days since 1970-01-01 of Easter Sunday, by the anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    w = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * w) // 451
    month, day = divmod(h + w - 7 * m + 114, 31)
    return int(np.datetime64(f"{year:04d}-{month:02d}-{day + 1:02d}").astype(np.int64))


def _holiday_flags(days: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Whether each day (days since 1970-01-01) is a holiday.

    The holidays are New Year's Day, Christmas Eve to Boxing Day, New Year's Eve, Good Friday,
    Easter Monday and US Thanksgiving; the movable ones only for years 1800 to 2300.
    """
    dates = days.astype("datetime64[D]")
    months = dates.astype("datetime64[M]").astype(np.int64) % 12 + 1
    day_of_month = (dates - dates.astype("datetime64[M]")).astype(np.int64) + 1
    flags = np.zeros(days.shape, dtype=bool)
    for month, day in _HOLIDAY_DATES:
        flags |= (months == month) & (day_of_month == day)
    years = dates.astype("datetime64[Y]").astype(np.int64) + 1970
    for year in np.unique(years).tolist():
        if not 1800 <= year <= 2300:
            continue
        easter = _easter_day(year)
        november = int(np.datetime64(f"{year:04d}-11-01").astype(np.int64))
        # The fourth Thursday; 1970-01-01, day 0, was a Thursday.
        thanksgiving = november + (3 - (november + 3) % 7) % 7 + 21
        flags |= np.isin(days, (easter - 2, easter + 1, thanksgiving))
    return flags


def _business_days(timestamps: NDArray[np.int64], step_ns: int) -> NDArray[np.float32] | None:
    """Return the weekdays that are not holidays in each step's month or quarter.

    Monthly steps (28 to 31 days) count their calendar month and quarterly steps (89 to 92
    days) their calendar quarter; any other step returns ``None``.
    """
    months = (timestamps // _DAY_NS).astype("datetime64[D]").astype("datetime64[M]")
    if 28 * _DAY_NS <= step_ns <= 31 * _DAY_NS:
        starts = months
        ends = months + np.timedelta64(1, "M")
    elif 89 * _DAY_NS <= step_ns <= 92 * _DAY_NS:
        starts = months - (months.astype(np.int64) % 3).astype("timedelta64[M]")
        ends = starts + np.timedelta64(3, "M")
    else:
        return None
    first, last = starts.astype("datetime64[D]"), ends.astype("datetime64[D]")
    span = np.arange(first.min(), last.max(), dtype="datetime64[D]")
    holidays = span[_holiday_flags(span.astype(np.int64))]
    return np.busday_count(first, last, holidays=holidays).astype(np.float32)


def calendar_channels(timestamps: NDArray[np.int64], step_ns: int) -> NDArray[np.float32]:
    """``[channels, time]`` calendar features of a series sampled every ``step_ns``.

    Steps that divide the day give hour of day (sub-daily steps only), day of week (Monday
    ``0``), day of year (``0``-``365``) and a holiday flag; weekly steps give week of year
    (``0``-``52``) and whether a holiday falls within three days; monthly steps (28 to 31 days)
    give month of year (``0``-``11``); quarterly steps (89 to 92 days) give the quarter
    (``0``-``3``). Both also include the period's business-day count. Any other step gives
    no channels.
    """
    days = timestamps // _DAY_NS
    dates = days.astype("datetime64[D]")
    day_of_year = (dates - dates.astype("datetime64[Y]")).astype(np.int64)
    rows: list[NDArray[np.int64] | NDArray[np.bool_]]
    if 0 < step_ns <= _DAY_NS and _DAY_NS % step_ns == 0:
        # 1970-01-01, day 0, was a Thursday.
        rows = [(days + 3) % 7, day_of_year, _holiday_flags(days)]
        if step_ns < _DAY_NS:
            rows.insert(0, (timestamps % _DAY_NS) // _HOUR_NS)
    elif step_ns == 7 * _DAY_NS:
        window = (days[:, None] + np.arange(-3, 4)).ravel()
        rows = [day_of_year // 7, _holiday_flags(window).reshape(days.shape[0], 7).any(axis=1)]
    elif 28 * _DAY_NS <= step_ns <= 31 * _DAY_NS:
        rows = [
            dates.astype("datetime64[M]").astype(np.int64) % 12,
            _business_days(timestamps, step_ns),
        ]
    elif 89 * _DAY_NS <= step_ns <= 92 * _DAY_NS:
        rows = [
            (dates.astype("datetime64[M]").astype(np.int64) % 12) // 3,
            _business_days(timestamps, step_ns),
        ]
    else:
        return np.empty((0, timestamps.shape[0]), dtype=np.float32)
    return np.stack([row.astype(np.float32) for row in rows])
