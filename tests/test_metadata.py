"""Calendar, cohort conditioning and per-series information boundaries."""

import pytest
import torch
from test_forecast import TTA, Probe, forecast

from continuousintelligence import load_model

DAY = 86_400 * 10**9


def test_calendar_is_known_and_is_not_target_encoded() -> None:
    """Real calendar rows follow encoded known covariates and reveal only known future data."""
    model = Probe()
    stamps = torch.arange(6, dtype=torch.int64) * 3_600 * 10**9
    forecast(
        model,
        torch.tensor([[2.0, 4.0, 2.0, 4.0]]),
        2,
        {**TTA, "calendar_covariates": True, "differencing_max_horizon": 1},
        timestamps=stamps,
        freq="h",
        future_covariates=torch.tensor([[0, 0, 1, 1, 0, 1]]),
        categorical_future=(0,),
    )
    first = model.batches[0]
    assert first["future_covariate_mask"].tolist() == [False, True, True, True, True, True] * 2
    torch.testing.assert_close(first["x"][2, :6], torch.arange(6).float())
    torch.testing.assert_close(first["x"][3, :6], torch.full((6,), 3.0))
    assert first["x"][0, -4:].isnan().all()
    assert first["x"][2:, -2:].isnan().all()


def test_business_days_are_future_known_covariates() -> None:
    """The release calendar supplies business-day rows for both observed and future months."""
    model = Probe()
    stamps = torch.tensor([1_709_164_800, 1_711_843_200, 1_714_435_200]) * 10**9
    forecast(
        model,
        torch.tensor([[1.5, 2.5]]),
        1,
        {**TTA, "calendar_covariates": True},
        timestamps=stamps,
        freq="ME",
    )
    first = model.batches[0]
    assert first["future_covariate_mask"].tolist() == [False, True, True] * 2
    torch.testing.assert_close(first["x"][2, :3], torch.tensor([21.0, 20.0, 21.0]))
    assert first["x"][0, 2:].isnan().all()


def test_calendar_can_be_disabled_with_timestamps_supplied() -> None:
    """Dates alone do not force calendar features when the release flag is disabled."""
    model = Probe()
    history = torch.tensor([[1.5, -2.5, 3.5, -4.5]])
    actual = forecast(
        model, history, 2, {**TTA, "calendar_covariates": False}, timestamps=torch.arange(6)
    )
    expected = forecast(Probe(), history, 2, TTA)
    assert model.batches[0]["x"].shape[0] == 2
    torch.testing.assert_close(actual, expected)


def test_chunks_follow_series_ids_and_restore_input_order() -> None:
    """Bare cohorts are forecast jointly in bounded chunks, in stable textual id order."""
    history = torch.tensor([[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0], [9.0, -1.0, 2.0, -3.0]])
    model = Probe()
    result = forecast(model, history, 2, {**TTA, "chunk_joint": 2}, series_ids=["b", "c", "a"])
    torch.testing.assert_close(model.batches[0]["x"][:2, :4], history[[2, 0]])
    torch.testing.assert_close(model.batches[1]["x"][:1, :4], history[[1]])
    expected = torch.empty_like(result)
    expected[[2, 0]] = forecast(Probe(), history[[2, 0]], 2, TTA)
    expected[[1]] = forecast(Probe(), history[[1]], 2, TTA)
    torch.testing.assert_close(result, expected)


def test_different_forecast_origins_never_share_a_chunk() -> None:
    """Even a large chunk budget cannot mix series ending at different timestamps."""
    model = Probe()
    stamps = torch.arange(6).expand(2, -1).clone() * DAY
    stamps[1] += DAY
    forecast(
        model,
        torch.ones(2, 4),
        2,
        {**TTA, "chunk_joint": 100},
        timestamps=stamps,
        freq="D",
        series_ids=["a", "b"],
    )
    assert [batch["x"].shape[0] for batch in model.batches] == [16]
    assert model.batches[0]["group_vector"].tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4


def test_neighbours_are_targets_and_covariates_stay_with_the_owner() -> None:
    """Correlated neighbours join each target without borrowing another series' covariates."""
    model = Probe()
    model.config.context_len = 64
    base = torch.sin(torch.arange(64) * 0.3)
    history = torch.stack([base, -base, base * 3 + 1])
    past = torch.stack([torch.full((1, 64), value) for value in (5.0, 6.0, 7.0)])
    forecast(
        model,
        history,
        2,
        {**TTA, "joint_neighbours": 2, "chunk_joint": 100, "differencing_max_horizon": 1},
        past_covariates=past,
        series_ids=["c", "b", "a"],
    )
    first = model.batches[0]
    torch.testing.assert_close(first["x"][:2, :64], history[[2, 0]])
    assert first["target_mask"].sum() == 12
    torch.testing.assert_close(first["x"][[2, 5, 8], :64], past[[2, 1, 0], 0])
    assert first["group_vector"].tolist() == [i for i in range(6) for _ in range(3)]


def test_static_groups_are_past_covariates_excluding_the_owner() -> None:
    """Static context pools only other histories and never becomes a future-known row."""
    model = Probe()
    model.config.context_len = 64
    generator = torch.Generator().manual_seed(4)
    history = torch.randn(6, 64, generator=generator)
    forecast(
        model,
        history,
        2,
        {**TTA, "static_groups": 2, "differencing_max_horizon": 1},
        series_ids=[str(i) for i in range(6)],
        static_features={"kind": ["a", "a", "a", "b", "b", "b"]},
    )
    values = history.double()
    z = (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, correction=0, keepdim=True)
    first = model.batches[0]
    torch.testing.assert_close(
        first["x"][1, :64], ((z[1] + z[2]) / 2).float(), atol=1e-6, rtol=1e-6
    )
    assert first["past_covariate_mask"].tolist() == [False, True] * 12
    assert first["x"][1::2, -4:].isnan().all()


def test_categorical_encodings_are_fitted_per_series() -> None:
    """A shared categorical row is encoded using each independent series' history only."""
    model = Probe()
    history = torch.tensor([[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]])
    forecast(
        model,
        history,
        3,
        {**TTA, "differencing_max_horizon": 1},
        future_covariates=torch.tensor([[0, 0, 1, 0, 1, 2]]),
        categorical_future=(0,),
        series_ids=["a", "b"],
    )
    actual = model.batches[0]["x"][[1, 3], 1:7]
    expected = torch.tensor(
        [[10 / 3, 10 / 3, 5, 10 / 3, 5, 4], [50 / 3, 50 / 3, 25, 50 / 3, 25, 20]]
    )
    torch.testing.assert_close(actual, expected)


def test_ragged_history_padding_does_not_change_differencing() -> None:
    """Packed histories use each series' real length and put the first difference at its start."""
    model = Probe()
    history = torch.tensor([[torch.nan, torch.nan, 0, 1, 2, 3], [10, 11, 12, 13, 14, 15]])
    stamps = torch.arange(8).expand(2, -1).clone() * DAY
    stamps[0, :2] = torch.iinfo(torch.int64).min
    actual = forecast(model, history, 2, TTA, timestamps=stamps, freq="D", series_ids=["a", "b"])
    expected = torch.cat(
        [forecast(Probe(), history[:1], 2, TTA), forecast(Probe(), history[1:], 2, TTA)]
    )
    torch.testing.assert_close(actual, expected)
    assert model.batches[0]["x"][0, :4].isnan().all()


def test_packed_model_matches_separate_covariate_forecasts(release_files) -> None:
    """Independent cases share a neural call while preserving their own known covariates."""
    model = load_model(*release_files, device="cpu")
    generator = torch.Generator().manual_seed(10)
    history = torch.randn(3, 7, generator=generator)
    known = torch.randn(3, 1, 10, generator=generator)
    actual = model.forecast(history, 3, future_covariates=known, series_ids=["c", "a", "b"])
    expected = torch.cat(
        [model.forecast(history[i : i + 1], 3, future_covariates=known[i]) for i in range(3)]
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"freq": "D"}, "timestamps"),
        ({"timestamps": torch.arange(6).float()}, "int64"),
        ({"timestamps": torch.arange(4)}, "cover"),
        ({"timestamps": torch.tensor([0, 1, 2, 3, 3, 4])}, "increasing"),
        ({"series_ids": ["same", "same"]}, "unique"),
        ({"series_ids": ["one"]}, "unique"),
        ({"static_features": {"kind": ["a", "b"]}}, "series_ids"),
        ({"series_ids": ["a", "b"], "static_features": {"kind": ["a"]}}, "value per series"),
        ({"series_ids": ["a", "b"], "future_covariates": torch.ones(3, 1, 6)}, "every series"),
    ],
)
def test_bad_metadata_fails_before_neural_inference(kwargs: dict, message: str) -> None:
    """Malformed metadata is rejected before any model call, rather than silently ignored."""
    model = Probe()
    with pytest.raises(ValueError, match=message):
        forecast(model, torch.ones(2, 4), 2, TTA, **kwargs)
    assert not model.batches


@pytest.mark.parametrize("series_ids", [None, ["a"]])
def test_long_horizon_preserves_calendar_and_covariate_roles(series_ids):
    """An extended window reveals all supplied known values and masks target and past futures."""
    model = Probe()
    horizon = 9
    known = torch.arange(4 + horizon).float()[None]
    actual = forecast(
        model,
        torch.tensor([[-0.5, 0.25, -0.25, 0.75]]),
        horizon,
        {**TTA, "postprocess": False},
        future_covariates=known,
        past_covariates=torch.ones(1, 4),
        timestamps=1_735_689_600_000_000_000 + torch.arange(4 + horizon) * 3_600_000_000_000,
        freq="h",
        series_ids=series_ids,
    )
    batch = model.batches[0]
    assert actual.shape == (1, 3, horizon)
    assert batch["x"].shape == (7, 14)
    torch.testing.assert_close(batch["x"][1, 4:13], known[0, 4:])
    assert batch["x"][[0, 6], 4:].isnan().all()
    assert batch["x"][:, -1].isnan().all()
    assert model.config.future_len == 4
