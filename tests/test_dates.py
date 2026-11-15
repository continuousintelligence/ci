"""Date-aware pandas and CSV forecasts use the existing calendar API."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_forecast import TTA, Probe

from continuousintelligence import load_model
from continuousintelligence.ci_proto import Forecaster


@pytest.mark.parametrize("cohort", [False, True])
@pytest.mark.parametrize("kind", ["dataframe", "series", "csv_path", "csv_string"])
def test_dated_forecast_matches_explicit_calendar_inputs(release_files, tmp_path, kind, cohort):
    model = load_model(*release_files, device="cpu")
    dates = pd.date_range("2026-11-01 09:00", periods=7, freq="h", name="timestamp")
    frame = pd.DataFrame(
        {"load": [0.25, -0.5, np.nan, 0.75], "temperature": [1.5, -2.5, 3.5, -4.5]},
        # Drop frequency metadata so the frequency must be inferred from the dates.
        index=pd.DatetimeIndex(list(dates[:4]), name="timestamp"),
    )
    if kind == "series":
        frame = frame[["load"]]
        source = frame["load"]
    elif kind == "dataframe":
        source = frame.copy()
    else:
        path = tmp_path / "history.csv"
        frame.to_csv(path)
        source = path if kind == "csv_path" else str(path)
    before = frame.copy(deep=True)
    arguments = {"series_ids": list(reversed(frame.columns))} if cohort else {}
    expected = model.forecast(
        torch.from_numpy(frame.to_numpy().T.copy()),
        3,
        timestamps=torch.tensor(dates.as_unit("ns").asi8),
        freq="h",
        **arguments,
    )
    actual = model.forecast(source, 3, **arguments)
    expected = pd.DataFrame(
        expected.permute(2, 0, 1).numpy().reshape(3, -1),
        index=dates[4:],
        columns=pd.MultiIndex.from_product(
            [frame.columns, model.quantiles.tolist()], names=["variable", "quantile"]
        ),
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, before)
    if kind == "dataframe":
        pd.testing.assert_frame_equal(source, before)
    elif kind == "series":
        pd.testing.assert_series_equal(source, before["load"])


@pytest.mark.parametrize("freq", ["MS", "ME", "W-MON", "D"])
def test_calendar_frequency_preserves_future_dates(freq):
    dates = pd.date_range("2024-01-01", periods=7, freq=freq, name="date")
    frame = pd.DataFrame({"load": [1.5, -2.5, 3.5, -4.5]}, index=dates[:4])
    model = Forecaster(Probe(), TTA)
    prediction = model.forecast(frame, 3)
    pd.testing.assert_index_equal(prediction.index, dates[4:])
    assert np.isfinite(prediction.to_numpy()).all()


def test_timezone_and_daylight_saving_time_are_preserved():
    dates = pd.date_range(
        "2026-03-08 00:00", periods=6, freq="h", tz="America/New_York", name="time"
    )
    frame = pd.DataFrame({"load": [0.25, -0.5, 0.75]}, index=dates[:3])
    prediction = Forecaster(Probe(), TTA).forecast(frame, 3)
    pd.testing.assert_index_equal(prediction.index, dates[3:])


def test_utc_csv_preserves_dates(tmp_path):
    dates = pd.date_range("2026-11-01", periods=6, freq="h", tz="UTC", name="timestamp")
    frame = pd.DataFrame({"load": [0.25, -0.5, 0.75, -0.25]}, index=dates[:4])
    path = tmp_path / "history.csv"
    frame.to_csv(path)
    prediction = Forecaster(Probe(), TTA).forecast(path, 2)
    pd.testing.assert_index_equal(prediction.index, dates[4:])


@pytest.mark.parametrize("length", [1, 2])
def test_short_history_accepts_explicit_frequency(length):
    dates = pd.date_range("2026-11-01", periods=length + 2, freq="h")
    frame = pd.DataFrame(
        {"load": [0.25, -0.5][:length]}, index=pd.DatetimeIndex(list(dates[:length]))
    )
    model = Forecaster(Probe(), TTA)
    with pytest.raises(ValueError, match="infer.*frequency"):
        model.forecast(frame, 2)
    prediction = model.forecast(frame, 2, freq="h")
    pd.testing.assert_index_equal(prediction.index, dates[length:])


@pytest.mark.parametrize(
    "dates,freq,message",
    [
        (["2026-11-01", "2026-11-01"], "D", "unique"),
        (["2026-11-02", "2026-11-01"], "D", "increasing"),
        (["2026-11-01", None], "D", "increasing"),
        (["2026-11-01", "2026-11-02", "2026-11-04"], None, "infer.*frequency"),
        (["2026-11-01", "2026-11-02", "2026-11-04"], "D", "regularly spaced"),
    ],
)
def test_invalid_dates_fail_before_inference(dates, freq, message):
    neural = Probe()
    frame = pd.DataFrame({"load": np.ones(len(dates))}, index=pd.to_datetime(dates))
    with pytest.raises(ValueError, match=message):
        Forecaster(neural, TTA).forecast(frame, 2, freq=freq)
    assert not neural.batches


@pytest.mark.parametrize("per_series", [False, True])
def test_explicit_timestamps_must_agree_with_dated_history(per_series):
    dates = pd.date_range("2026-11-01", periods=6, freq="h")
    frame = pd.DataFrame({"load": [0.25, -0.5, 0.75, -0.25]}, index=dates[:4])
    stamps = torch.tensor(dates.as_unit("ns").asi8)
    if per_series:
        stamps = stamps[None]
    model = Forecaster(Probe(), TTA)
    actual = model.forecast(frame, 2, timestamps=stamps)
    expected = model.forecast(frame, 2)
    pd.testing.assert_frame_equal(actual, expected)
    for index in (0, -1):
        damaged = stamps.clone()
        damaged[..., index] += 1
        with pytest.raises(ValueError, match="timestamps must match"):
            model.forecast(frame, 2, timestamps=damaged)


def test_dated_csv_covariates(tmp_path):
    dates = pd.date_range("2026-11-01", periods=6, freq="h", name="timestamp")
    history = pd.DataFrame({"load": [0.25, -0.5, 0.75, -0.25]}, index=dates[:4])
    past = pd.DataFrame({"past": [1.0, 2.0, 3.0, 4.0]}, index=dates[:4])
    future = pd.DataFrame({"known": [0, 0, 1, 1, 0, 1]}, index=dates)
    past_path, future_path = tmp_path / "past.csv", tmp_path / "future.csv"
    past.to_csv(past_path)
    future.to_csv(future_path)
    model = Forecaster(Probe(), TTA)
    actual = model.forecast(
        history,
        2,
        past_covariates=past_path,
        future_covariates=future_path,
        categorical_future=(0,),
    )
    expected = model.forecast(
        history,
        2,
        past_covariates=past,
        future_covariates=future,
        categorical_future=(0,),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_example_csv():
    path = Path(__file__).resolve().parents[1] / "examples" / "history.csv"
    prediction = Forecaster(Probe(), TTA).forecast(path, 3)
    expected = pd.date_range("2026-11-02", periods=3, freq="h", name="timestamp")
    pd.testing.assert_index_equal(prediction.index, expected)
    assert prediction.columns.get_level_values("variable").unique().tolist() == [
        "store_a",
        "store_b",
    ]
