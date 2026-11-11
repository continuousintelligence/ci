"""Parity and information boundaries for batched inference and reduced padding."""

import copy

import numpy as np
import pytest
import torch
from test_forecast import TTA, Probe, forecast

from continuousintelligence import load_model
from continuousintelligence.ci_proto import _calendar
from continuousintelligence.ci_proto._forecast import _forecast_once
from continuousintelligence.ci_proto._groups import _on_own_stamps
from continuousintelligence.ci_proto._model import forecast_layout


@pytest.mark.parametrize(
    "length, cap, padded",
    [(1, 8, 2), (3, 8, 4), (4, 8, 4), (5, 8, 8), (9, 8, 8), (17, None, 32), (8, 6, 6)],
)
def test_context_padding_preserves_observations(length: int, cap: int | None, padded: int) -> None:
    """Padding reaches the next power of two, respecting the checkpoint cap and patch minimum."""
    model = Probe()
    model.config.context_len = cap
    history = torch.arange(length).float()[None]
    forecast(model, history, 3, {**TTA, "postprocess": False})
    (batch,) = model.batches
    assert batch["x"].shape == (1, padded + 4)
    kept = min(length, padded)
    torch.testing.assert_close(batch["x"][:, padded - kept : padded], history[:, -kept:])
    assert batch["x"][:, : padded - kept].isnan().all()
    assert batch["x"][:, padded:].isnan().all()


@pytest.mark.parametrize("targets, known, past", [(1, 0, 0), (1, 1, 1), (2, 1, 0)])
def test_regular_layout_matches_generic_groups(
    release_files, targets: int, known: int, past: int
) -> None:
    """Regular packing preserves attention, covariate masking and independence between groups."""
    model = load_model(*release_files, device="cpu")._model
    channels = targets + known + past
    layout = forecast_layout(3, channels, targets, known, torch.device("cpu"))
    batch = {
        "x": torch.randn(3 * channels, 24),
        "group_vector": layout.groups,
        "target_mask": layout.targets,
        "future_covariate_mask": layout.future,
        "past_covariate_mask": layout.past,
    }
    with torch.no_grad():
        expected = model(batch)
        actual = model(batch, layout=layout)
        changed = batch["x"].clone()
        changed[channels:] += 30
        isolated = model({**batch, "x": changed}, layout=layout)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(isolated[:channels], actual[:channels], atol=0, rtol=0)
    assert forecast_layout(3, channels, targets, known, torch.device("cpu")) is layout


@pytest.mark.parametrize(
    "length, future_len, horizon", [(24, 8, 8), (23, 8, 5), (25, 7, 7), (24, 8, 1)]
)
@pytest.mark.parametrize("quantiles", [9, 99])
def test_forecast_projection_matches_full_field(
    release_files, length: int, future_len: int, horizon: int, quantiles: int
) -> None:
    """Partial patches retain the forecast shift and every checkpoint quantile."""
    from continuousintelligence.ci_proto._model import Model

    config = torch.load(release_files[0], weights_only=True)["hyper_parameters"]["model"]
    config["quantiles"] = [i / (quantiles + 1) for i in range(1, quantiles + 1)]
    model = Model(config).eval()
    batch = {"x": torch.randn(2, length)}
    projected = []
    handle = model.output_patch_embedding.register_forward_pre_hook(
        lambda module, args: projected.append(args[0].shape[1])
    )
    with torch.no_grad():
        full = model(batch, future_len=future_len)
        actual = model(batch, future_len=future_len, horizon=horizon)
    handle.remove()
    start = full.shape[-1] - future_len - config["output_patch_size"]
    torch.testing.assert_close(actual, full[..., start : start + horizon], atol=1e-6, rtol=1e-6)
    assert actual.shape == (2, quantiles, horizon)
    assert projected[1] < projected[0]


def test_cached_layout_rejects_mismatched_roles(release_files) -> None:
    """An unrelated layout cannot bypass validation of user-supplied masks."""
    model = load_model(*release_files, device="cpu")._model
    layout = forecast_layout(1, 2, 1, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="layout must match"):
        model({"x": torch.randn(2, 24)}, layout=layout)


@pytest.mark.parametrize("horizon", [1, 3, 9])
def test_paired_signs_match_separate_passes(release_files, horizon: int) -> None:
    """Trend anchors, missing values and binary scaling survive shared sign preprocessing."""
    model = load_model(*release_files, device="cpu")._model
    rows = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [torch.nan, torch.nan, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
            [torch.nan] * 6,
            [3.0] * 6,
        ]
    )
    context = torch.stack([rows, rows * 2 + 0.5])
    lengths = torch.tensor([6, 6])
    future = torch.randn(2, 1, horizon)
    config = {**TTA, "diff_band_exponent": 0.59, "diff_band_scale": 1.68}
    positive = _forecast_once(model, context, future, 3, horizon, config, lengths)
    negative = _forecast_once(model, -context, -future, 3, horizon, config, lengths)
    paired = _forecast_once(model, context, future, 3, horizon, config, lengths, sign_flip=True)
    torch.testing.assert_close(paired, torch.cat([positive, negative]), atol=1e-5, rtol=1e-5)


def test_time_mixer_batches_tied_directions(release_files) -> None:
    """Evaluation uses one cell call and matches separate forward and reverse passes."""
    mixer = load_model(*release_files, device="cpu")._model.stack[0].time_mixer
    separate = copy.deepcopy(mixer).train()
    values = torch.randn(3, 7, 16)
    mask = torch.tensor([False, True, False])
    calls = []
    handle = mixer.fwd_cell.register_forward_pre_hook(
        lambda module, args: calls.append(len(args[0]))
    )
    with torch.no_grad():
        actual = mixer(values, mask)
        expected = separate(values, mask)
    handle.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert calls == [4]


def test_calendar_reuses_only_identical_grids(monkeypatch) -> None:
    """Equal date grids share features; different grids with the same length remain separate."""
    original = _calendar.calendar_channels
    calls = []

    def counted(stamps, step):
        calls.append(stamps.copy())
        return original(stamps, step)

    monkeypatch.setattr(_calendar, "calendar_channels", counted)
    stamps = torch.arange(8)[None].repeat(3, 1) * 3_600_000_000_000
    stamps[2] += 3_600_000_000_000
    model = Probe()
    forecast(
        model, torch.ones(3, 6), 2, TTA, timestamps=stamps, freq="h", series_ids=["c", "a", "b"]
    )
    assert len(calls) == 2
    batch = model.batches[0]["x"]
    torch.testing.assert_close(batch[1, 2:10], batch[11, 2:10])
    assert not torch.equal(batch[1, 2:10], batch[6, 2:10])


@pytest.mark.parametrize(
    "grid, stamps", [([1, 2, 3], [1, 2, 3]), ([1, 1, 2], [1, 1, 2]), ([1, 3, 4], [1, 2, 4])]
)
def test_timestamp_shortcut_retains_searchsorted_semantics(
    grid: list[int], stamps: list[int]
) -> None:
    """Aligned, duplicate and missing timestamps preserve first-match alignment."""
    values = np.array([2.0, 5.0, 7.0])
    g, s = np.array(grid), np.array(stamps)
    expected = np.array([values[grid.index(t)] if t in grid else np.nan for t in stamps])
    actual = _on_own_stamps(values, g, s)
    np.testing.assert_array_equal(actual, expected)
    actual[0] = 99
    assert values[0] == 2


@pytest.mark.parametrize("known", [False, True])
def test_missing_prefix_is_trimmed_only_when_all_rows_are_missing(known: bool) -> None:
    """Observed covariates keep their context even when every target starts with missing values."""
    model = Probe()
    model.config.context_len = 32
    history = torch.full((1, 16), torch.nan)
    history[:, -3:] = torch.tensor([0.5, -0.5, 1.5])
    options = {"future_covariates": torch.ones(1, 18)} if known else {}
    forecast(model, history, 2, {**TTA, "postprocess": False}, **options)
    (batch,) = model.batches
    context = 16 if known else 4
    assert batch["x"].shape[-1] == context + 4
    torch.testing.assert_close(batch["x"][0, context - 3 : context], history[0, -3:])


def test_all_missing_history_uses_one_patch() -> None:
    """A history without any observations still has a valid minimum context and finite output."""
    model = Probe()
    result = forecast(model, torch.full((1, 8), torch.nan), 2, TTA)
    assert (
        model.batches[0]["x"].shape[-1] == model.config.output_patch_size + model.config.future_len
    )
    assert result.isfinite().all()
