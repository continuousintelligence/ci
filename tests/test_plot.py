"""Plotting consumes the public tensor shapes without changing model output."""

import matplotlib.pyplot as plt
import pytest
import torch

from continuousintelligence import plot_forecast


@pytest.mark.parametrize("count, variables", [(9, 1), (99, 2), (3, 1)])
def test_plot_forecast_preserves_quantiles_and_input_tensors(tmp_path, count, variables):
    levels = (
        torch.tensor([0.1, 0.5, 0.9]) if count == 3 else torch.arange(1, count + 1) / (count + 1)
    )
    history = torch.arange(12).float().repeat(variables, 1)
    prediction = (levels[None, :, None] + torch.arange(5)[None, None]).repeat(variables, 1, 1)
    before = prediction.clone()
    output = tmp_path / "plots/forecast.png"
    figure = plot_forecast(history, prediction, levels, output=output)
    try:
        assert output.is_file() and output.stat().st_size > 1000
        assert len(figure.axes) == variables
        for axes in figure.axes:
            assert axes.get_xlim() == (-12, 4)
            median = next(line for line in axes.lines if line.get_label() == "Median forecast")
            torch.testing.assert_close(torch.as_tensor(median.get_ydata()), before[0, count // 2])
        torch.testing.assert_close(prediction, before, rtol=0, atol=0)
    finally:
        plt.close(figure)


def test_plot_forecast_uses_given_dates_and_restores_style():
    import pandas as pd

    original_size = plt.rcParams["font.size"]
    dates = pd.date_range("2024-01-01", periods=16, freq="h")
    figure = plot_forecast(
        torch.ones(1, 12),
        torch.ones(1, 3, 4),
        torch.tensor([0.1, 0.5, 0.9]),
        timestamps=dates,
        ylabel="Benzene (µg/m³)",
    )
    try:
        assert figure.axes[0].get_ylabel() == "Benzene (µg/m³)"
        assert plt.rcParams["font.size"] == original_size
    finally:
        plt.close(figure)


@pytest.mark.parametrize("damage", ["variables", "quantiles", "median", "nonfinite", "timestamps"])
def test_invalid_plot_inputs_fail(damage):
    history, prediction, levels = (
        torch.ones(1, 12),
        torch.ones(1, 3, 4),
        torch.tensor([0.1, 0.5, 0.9]),
    )
    kwargs = {}
    if damage == "variables":
        prediction = prediction.repeat(2, 1, 1)
    elif damage == "quantiles":
        levels = torch.tensor([0.5, 0.1, 0.9])
    elif damage == "median":
        levels = torch.tensor([0.1, 0.4, 0.9])
    elif damage == "nonfinite":
        prediction[0, 0, 0] = float("nan")
    else:
        kwargs["timestamps"] = list(range(15))
    with pytest.raises(ValueError):
        plot_forecast(history, prediction, levels, **kwargs)
