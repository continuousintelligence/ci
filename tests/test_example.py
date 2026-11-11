"""The bundled quick-start data is independent of the caller's working directory."""

import matplotlib.pyplot as plt
import torch

from continuousintelligence import load_example, load_model, plot_forecast


def test_example_is_observed_history_and_returns_independent_tensors(tmp_path, monkeypatch):
    """The 24 empty forecast rows never enter the history returned to the model."""
    monkeypatch.chdir(tmp_path)
    history = load_example()
    assert history.shape == (1, 120)
    assert history.dtype == torch.float32 and history.device.type == "cpu"
    assert history.isfinite().all()
    torch.testing.assert_close(history[0, :3], torch.tensor([206.0, 123.0, 89.0]))
    torch.testing.assert_close(history[0, -2:], torch.tensor([238.0, 153.0]))
    original = history.clone()
    history.zero_()
    torch.testing.assert_close(load_example(), original)


def test_example_forecasts_and_plots_with_the_public_api(release_files, tmp_path):
    """The quick start goes from bundled history to a saved plot."""
    model = load_model(*release_files, device="cpu")
    history = load_example()
    prediction = model.forecast(history, horizon=24)
    assert prediction.shape == (1, 3, 24) and prediction.isfinite().all()
    output = tmp_path / "forecast.png"
    figure = plot_forecast(history, prediction, model.quantiles, output=output)
    try:
        assert output.is_file() and output.stat().st_size > 1000
    finally:
        plt.close(figure)
