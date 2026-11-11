"""Forecast a time series from a CSV file and plot ci-proto's median and quantile bands.

The default CSV holds five days of hourly bicycle rentals in Washington, D.C. from
UCI's Bike Sharing dataset (CC BY 4.0), followed by the 24 hours to forecast.
Any CSV with a timestamp column and a value column works:

- With a ``split`` column, rows marked ``history`` are observed and the rows marked
  ``forecast`` that follow them set the horizon and its timestamps.
- Without one, every row is history. Pass ``--horizon``; future timestamps continue the
  frequency inferred from the history.
"""

import argparse
import os
from importlib.resources import files
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import torch

from continuousintelligence import load_model, plot_forecast

DATA_PATH = files("continuousintelligence").joinpath("_data/bike_sharing.csv")
LABELS = {"rentals": "Bike rentals per hour"}
LEVELS = torch.arange(1, 10, dtype=torch.float32) / 10


def read_series(
    path: Path, time_column: str, value_column: str, horizon: int | None
) -> tuple[pd.DatetimeIndex, torch.Tensor, pd.DatetimeIndex]:
    """Return history timestamps, history values ``[1, T]`` and forecast timestamps."""
    data = pd.read_csv(path)
    missing = {time_column, value_column} - set(data.columns)
    if missing:
        raise ValueError(f"{path} has no column(s) {sorted(missing)}")
    times = pd.DatetimeIndex(pd.to_datetime(data[time_column]))
    if "split" in data:
        if horizon is not None:
            raise ValueError("--horizon applies only to CSVs without a split column")
        split = data["split"]
        if not split.isin(["history", "forecast"]).all():
            raise ValueError("split values must be 'history' or 'forecast'")
        observed = split.eq("history").to_numpy()
        count = int(observed.sum())
        if not observed[:count].all():
            raise ValueError("history rows must come before forecast rows")
        history_times, future_times = times[:count], times[count:]
        values = data[value_column].iloc[:count]
    else:
        if horizon is None:
            raise ValueError("pass --horizon for a CSV without a split column")
        freq = pd.infer_freq(times) if len(times) >= 3 else None
        if freq is None:
            raise ValueError("timestamps have no regular frequency; add a split column")
        history_times = times
        future_times = pd.date_range(times[-1], periods=horizon + 1, freq=freq)[1:]
        values = data[value_column]
    if history_times.empty or future_times.empty:
        raise ValueError("the CSV needs at least one history row and one forecast step")
    history = torch.tensor(values.to_numpy(dtype=float), dtype=torch.float32)[None]
    return history_times, history, future_times


def main() -> None:
    """Read a CSV series, forecast it and save the plot."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, help="local bundle directory or Hugging Face model ID"
    )
    parser.add_argument("--revision", help="Hub revision; a commit SHA pins the model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV file to forecast")
    parser.add_argument("--time-column", default="timestamp", help="column of timestamps")
    parser.add_argument("--value-column", default="rentals", help="column to forecast")
    parser.add_argument("--horizon", type=int, help="steps to forecast without a split column")
    parser.add_argument("--ylabel", help="y-axis label (default: the value column)")
    parser.add_argument("--output", type=Path, default=Path("outputs/forecast.png"))
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.horizon is not None and args.horizon < 1:
        parser.error("--horizon must be positive")
    try:
        history_times, history, future_times = read_series(
            args.data, args.time_column, args.value_column, args.horizon
        )
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    model = load_model(args.model, device=args.device, revision=args.revision)
    horizon = len(future_times)
    prediction = model.forecast(history, horizon=horizon)
    columns = (model.quantiles[:, None] - LEVELS).abs().argmin(dim=0)
    if not torch.allclose(model.quantiles[columns], LEVELS, atol=1e-6, rtol=0):
        raise ValueError("the checkpoint must provide quantiles 0.1 to 0.9 in increments of 0.1")
    prediction = prediction[:, columns]
    if not torch.isfinite(prediction).all():
        raise RuntimeError("the model returned nonfinite forecasts")
    ylabel = args.ylabel or LABELS.get(args.value_column, args.value_column)
    plot_forecast(
        history,
        prediction,
        model.quantiles[columns],
        timestamps=history_times.append(future_times),
        ylabel=ylabel,
        output=args.output,
    )
    print(f"Device: {model.device} | History: {history.shape[-1]} | Horizon: {horizon}")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
