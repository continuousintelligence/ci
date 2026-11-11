"""Plot observed history and forecast quantiles."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_forecast(
    history: Tensor,
    prediction: Tensor,
    quantiles: Tensor,
    *,
    timestamps: Sequence | Tensor | None = None,
    ylabel: str = "Value",
    output: str | Path | None = None,
) -> Figure:
    """Plot ``[variables, time]`` history and ``[variables, quantiles, horizon]`` predictions.

    ``quantiles`` is the model's probability grid. Each variable gets one panel,
    showing the median and available nested intervals between 0.1 and 0.9.
    ``timestamps`` may supply a shared date axis covering history plus horizon;
    otherwise the forecast starts at step zero. Returns a Matplotlib figure and
    optionally saves it to ``output``.
    """
    if any(not isinstance(value, Tensor) for value in (history, prediction, quantiles)):
        raise ValueError("history, prediction and quantiles must be PyTorch tensors")
    if any(value.is_complex() for value in (history, prediction, quantiles)):
        raise ValueError("plot inputs must be real")
    if history.ndim != 2 or min(history.shape) < 1:
        raise ValueError("history must be a nonempty [variables, time] tensor")
    if prediction.ndim != 3 or min(prediction.shape) < 1 or prediction.shape[0] != history.shape[0]:
        raise ValueError("prediction must have shape [variables, quantiles, horizon]")
    if (
        quantiles.ndim != 1
        or len(quantiles) != prediction.shape[1]
        or not bool(((quantiles > 0) & (quantiles < 1)).all())
        or not bool((quantiles.diff() > 0).all())
    ):
        raise ValueError("quantiles must be a matching, strictly increasing probability grid")
    if bool(history.isinf().any()) or not bool(prediction.isfinite().all()):
        raise ValueError("history must be finite or NaN and predictions must be finite")
    history, prediction, quantiles = (
        value.detach().cpu().float() for value in (history, prediction, quantiles)
    )
    levels = torch.arange(1, 10) / 10
    columns = (quantiles[:, None] - levels).abs().argmin(dim=0)
    available = torch.isclose(quantiles[columns], levels, atol=1e-6, rtol=0)
    if not available[4]:
        raise ValueError("quantiles must include the median (0.5)")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    count, length = history.shape
    horizon = prediction.shape[-1]
    if timestamps is None:
        times = torch.arange(-length, horizon).numpy()
    else:
        import pandas as pd

        if isinstance(timestamps, Tensor):
            timestamps = timestamps.detach().cpu().numpy()
        times = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if len(times) != length + horizon or times.hasnans or not times.is_unique:
            raise ValueError("timestamps must cover history plus horizon without missing dates")
        if not times.is_monotonic_increasing:
            raise ValueError("timestamps must be strictly increasing")
    past, future = times[:length], times[length:]
    background, panel, teal = "#173446", "#1d3d50", "#549a8b"
    white, muted, grid = "#fafbfb", "#b6c3c9", "#2d4d5d"
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    with plt.rc_context(
        {
            "font.family": "Aptos" if "Aptos" in fonts else "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 16,
            "text.color": white,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "savefig.dpi": 180,
        }
    ):
        figure, panels = plt.subplots(count, 1, figsize=(13.5, 5.8 * count), squeeze=False)
        figure.set_facecolor(background)
        figure.subplots_adjust(left=0.09, right=0.98, bottom=0.15 / count, top=0.97, hspace=0.3)
        for index, axes in enumerate(panels[:, 0]):
            axes.set_facecolor(panel)
            axes.axvspan(future[0], future[-1], color="#214452", linewidth=0)
            axes.axvline(future[0], color=teal, linewidth=1.2, linestyle=":")
            axes.grid(color=grid, linewidth=0.7)
            axes.set_axisbelow(True)
            axes.spines[["top", "right", "left"]].set_visible(False)
            axes.spines["bottom"].set_color(grid)
            axes.tick_params(length=0, pad=8)
            axes.set_xlim(past[0], future[-1])
            axes.plot(past, history[index], color=white, linewidth=1.7, label="History")
            for lo, hi, alpha in [(0, 8, 0.22), (1, 7, 0.20), (2, 6, 0.18), (3, 5, 0.16)]:
                if available[lo] and available[hi]:
                    axes.fill_between(
                        future,
                        prediction[index, columns[lo]],
                        prediction[index, columns[hi]],
                        color=teal,
                        alpha=alpha,
                        linewidth=0,
                        label="10–90% quantiles" if lo == 0 else None,
                    )
            median = prediction[index, columns[4]]
            axes.plot(
                future, median, color="#76c6b3", linewidth=2.4, label="Median forecast", zorder=4
            )
            axes.plot(
                [past[-1], future[0]],
                [float(history[index, -1]), float(median[0])],
                color="#76c6b3",
                linewidth=1.5,
                linestyle=":",
            )
            axes.set_ylabel(ylabel)
            if count > 1:
                axes.set_title(f"Series {index + 1}", color=white)
            low, high = axes.get_ylim()
            if low < 0 < high:
                axes.axhline(0, color=muted, alpha=0.3, linewidth=0.8)
            if timestamps is not None:
                if times[-1] - times[0] <= pd.Timedelta(days=14):
                    axes.xaxis.set_major_locator(mdates.DayLocator())
                    axes.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
                else:
                    locator = mdates.AutoDateLocator()
                    axes.xaxis.set_major_locator(locator)
                    axes.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            axes.legend(
                loc="upper left",
                frameon=True,
                facecolor=background,
                edgecolor=grid,
                framealpha=0.95,
                fontsize=15,
                labelcolor=white,
                borderpad=0.9,
            )
        if timestamps is None:
            panels[-1, 0].set_xlabel("Time step relative to forecast origin")
        if output is not None:
            path = Path(output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, facecolor=figure.get_facecolor())
    return figure
