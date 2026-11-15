"""Convert forecast inputs and restore their containers at the API boundary."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

ForecastInput = Tensor | np.ndarray | pd.DataFrame | pd.Series | str | Path
ForecastOutput = Tensor | np.ndarray | pd.DataFrame


def _read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path).expanduser())
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame


def _dates(
    index: pd.DatetimeIndex, horizon: int, freq: str | None
) -> tuple[pd.DatetimeIndex, Tensor, str]:
    if index.empty or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("history dates must be nonempty, unique, and increasing")
    freq = freq or index.freqstr
    if freq is None and len(index) >= 3:
        freq = pd.DatetimeIndex(index, freq="infer").freqstr
    if freq is None:
        raise ValueError("cannot infer history frequency; pass freq, for example freq='h'")
    dates = pd.date_range(index[0], periods=len(index) + horizon, freq=freq, name=index.name)
    if not (dates[: len(index)] == index).all():
        raise ValueError("history dates must be regularly spaced at freq; reindex missing steps")
    stamps = torch.from_numpy(dates.as_unit("ns").asi8.copy())
    return dates, stamps, freq


def as_tensor(value: ForecastInput, name: str) -> Tensor:
    """Arrays use [variables, time]; tables use time rows and variable columns."""
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (str, Path)):
        value = _read_csv(value)
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        array = value.to_numpy()
        if array.dtype == object and all(dtype.kind in "biuf" for dtype in value.dtypes):
            array = value.to_numpy(dtype=np.float64, na_value=np.nan)
        value = array.T
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a tensor, NumPy array, pandas object, or CSV path")
    try:
        # Copy to support negative strides and read-only pandas/NumPy buffers.
        return torch.from_numpy(value.copy())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values or NaN") from error


def prepare_history(
    history: ForecastInput,
    quantiles: Tensor,
    horizon: int,
    timestamps: Tensor | None,
    freq: str | None,
) -> tuple[Tensor, Callable[[Tensor], ForecastOutput], Tensor | None, str | None]:
    """Remember output labels and the caller's device before converting history."""
    if isinstance(history, (str, Path)):
        history = _read_csv(history)
    if isinstance(history, pd.Series):
        history = history.to_frame()
    tensor = as_tensor(history, "history")
    if isinstance(history, Tensor):
        device = history.device
        return tensor, lambda prediction: prediction.to(device), timestamps, freq
    if isinstance(history, np.ndarray):
        return tensor, lambda prediction: prediction.cpu().numpy(), timestamps, freq

    future_index = pd.RangeIndex(horizon, name="step")
    if isinstance(history.index, pd.DatetimeIndex):
        dates, stamps, freq = _dates(history.index, horizon, freq)
        if timestamps is not None:
            if (
                not isinstance(timestamps, Tensor)
                or timestamps.dtype != torch.int64
                or timestamps.ndim not in (1, 2)
                or timestamps.shape[-1] < len(dates)
                or not torch.equal(
                    timestamps[..., : len(dates)].detach().cpu(),
                    stamps.expand_as(timestamps[..., : len(dates)]),
                )
            ):
                raise ValueError("timestamps must match the history dates and forecast frequency")
        else:
            timestamps = stamps
        future_index = dates[len(history) :]

    levels = quantiles.tolist()
    columns = pd.MultiIndex.from_product([history.columns, levels], names=["variable", "quantile"])

    def restore_frame(prediction: Tensor) -> pd.DataFrame:
        values = prediction.permute(2, 0, 1).cpu().numpy().reshape(horizon, -1)
        return pd.DataFrame(values, index=future_index, columns=columns)

    return tensor, restore_frame, timestamps, freq
