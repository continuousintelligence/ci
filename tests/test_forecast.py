"""Checks for release preprocessing, information boundaries and quantile forecasts."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from continuousintelligence.ci_proto._forecast import Forecaster, _calibrate, _difference_mask
from continuousintelligence.ci_proto._model import ForecastLayout

TTA = {
    "postprocess": True,
    "tta_diff": True,
    "tta_sign_flip": True,
    "tta_running_totals": True,
    "target_encode_categoricals": True,
    "clip_nonnegative": True,
    "integer_round": True,
    "differencing_max_horizon": 48,
    "calendar_covariates": True,
    "bfloat16": False,
    "static_groups": 0,
    "joint_neighbours": 0,
    "chunk_joint": 0,
    "trend_threshold": 1.5,
    "trend_min_r2": 0.87,
    "trend_window": 1.0,
    "diff_band_exponent": 0.5,
    "diff_band_scale": 1.0,
    "raw_band_exponent": 0.5,
    "raw_band_scale": 1.0,
}


def forecast(
    model: nn.Module, history: Tensor, horizon: int, tta: dict[str, float], **kwargs
) -> Tensor:
    """Exercise the public forecaster with an instrumented neural model."""
    return Forecaster(model, tta).forecast(history, horizon, **kwargs)


class Probe(nn.Module):
    """Capture inputs and emit a time-indexed field or supplied quantile fans."""

    def __init__(self, fans: list[list[float]] | None = None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(context_len=8, future_len=4, output_patch_size=2)
        self.register_buffer("quantiles", torch.tensor([0.1, 0.5, 0.9]))
        self.batches: list[dict[str, Tensor]] = []
        self.fans = fans
        self.eval()

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        future_len: int | None = None,
        horizon: int | None = None,
        layout: ForecastLayout | None = None,
    ) -> Tensor:
        assert not torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        assert not self.training
        self.batches.append({key: value.clone() for key, value in batch.items()})
        rows, width = batch["x"].shape
        if self.fans is not None:
            fans = torch.tensor(self.fans).repeat_interleave(rows // len(self.fans), dim=0)
            field = fans[:, :, None].expand(rows, -1, width)
        else:
            direction = batch["x"].nan_to_num().sum(-1).sign()[:, None, None]
            steps = torch.arange(width).float()[None, None]
            offsets = torch.tensor([-0.4, 0.2, 1.6])[None, :, None]
            field = direction * steps + offsets
        if horizon is None:
            return field
        start = width - future_len - self.config.output_patch_size
        return field[..., start : start + horizon]


def test_padding_roles_and_forecast_shift() -> None:
    """Only known covariates reveal future values; the output is shifted by one patch."""
    model = Probe()
    history = torch.tensor([[0.25, 0.75, -0.5, 0.25]])
    past = torch.tensor([[11.0, 12.0, 13.0, 14.0]])
    future = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 999.0]])
    output = forecast(model, history, 2, TTA, past_covariates=past, future_covariates=future)
    (first,) = model.batches
    assert first["x"].shape == (6, 8)
    assert first["x"][[0, 2], -4:].isnan().all()
    torch.testing.assert_close(
        first["x"][1, -4:], torch.tensor([5.0, 6.0, torch.nan, torch.nan]), equal_nan=True
    )
    torch.testing.assert_close(first["x"][0, :4], history[0])
    torch.testing.assert_close(first["x"][2, :4], past[0])
    assert first["target_mask"].tolist() == [True, False, False] * 2
    assert first["future_covariate_mask"].tolist() == [False, True, False] * 2
    assert first["past_covariate_mask"].tolist() == [False, False, True] * 2
    torch.testing.assert_close(first["x"][3:], -first["x"][:3], equal_nan=True)
    torch.testing.assert_close(output, torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]]))


def test_release_context_preserves_history_beyond_8192_steps() -> None:
    """A 16,384-step checkpoint retains its full context and discards only older observations."""
    model = Probe()
    model.config.context_len = 16384
    history = torch.arange(17000).float()[None] / 100
    forecast(model, history, 2, {**TTA, "differencing_max_horizon": 1})
    first = model.batches[0]["x"]
    assert first.shape == (2, 16384 + model.config.future_len)
    torch.testing.assert_close(first[:1, :16384], history[:, -16384:], rtol=0, atol=0)
    assert first[:, 16384:].isnan().all()


def test_sign_flip_precedes_clipping_and_rounding() -> None:
    """Ensemble in level space before clipping nonnegative targets and rounding counts."""
    model = Probe([[-3.4, 0.4, 2.4], [-3.6, -0.8, 2.8]])
    history = torch.tensor([[0.0, 2.0, 0.0, 1.0], [-0.5, 0.25, -0.25, 0.75]])
    output = forecast(model, history, 1, TTA)
    torch.testing.assert_close(output[..., 0], torch.tensor([[0.0, 1.0, 3.0], [-3.1, 0.6, 3.0]]))


def test_sign_flip_can_be_disabled() -> None:
    """Disabling sign flipping returns the single calibrated pass without a second neural call."""
    model = Probe([[-2.4, 0.4, 2.4]])
    actual = forecast(
        model, torch.tensor([[-0.5, 0.25, -0.25]]), 1, {**TTA, "tta_sign_flip": False}
    )
    assert len(model.batches) == 1
    torch.testing.assert_close(actual[..., 0], torch.tensor([[-2.4, 0.4, 2.4]]))


def test_postprocessing_can_be_disabled() -> None:
    """Raw inference bypasses differencing, band calibration, sorting and sign flipping."""
    model = Probe([[5.0, -2.0, 1.0]])
    history = torch.tensor([[-4.5, -3.5, -2.5, -1.5]])
    actual = forecast(model, history, 1, {**TTA, "postprocess": False})
    assert len(model.batches) == 1
    torch.testing.assert_close(model.batches[0]["x"][:1, :4], history)
    torch.testing.assert_close(actual[..., 0], torch.tensor([[5.0, -2.0, 1.0]]))


def test_differencing_can_be_disabled() -> None:
    """Turning off differencing preserves a strongly trending history in level space."""
    model = Probe()
    history = torch.tensor([[-4.5, -3.5, -2.5, -1.5]])
    forecast(model, history, 1, {**TTA, "tta_diff": False})
    torch.testing.assert_close(model.batches[0]["x"][:1, :4], history)


def test_running_totals_can_be_disabled_independently() -> None:
    """Disabling running totals preserves ordinary trend detection."""
    rows = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 100.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
    actual = _difference_mask(rows, {**TTA, "tta_running_totals": False})
    torch.testing.assert_close(actual, torch.tensor([False, True]))


def test_categorical_target_encoding_can_be_disabled() -> None:
    """Categorical rows keep their numeric codes when target encoding is disabled."""
    model = Probe()
    codes = torch.tensor([[0, 0, 1, 1, 0, 1]])
    forecast(
        model,
        torch.tensor([[-0.5, 0.25, -0.25, 0.75]]),
        2,
        {**TTA, "target_encode_categoricals": False},
        future_covariates=codes,
        categorical_future=(0,),
    )
    torch.testing.assert_close(model.batches[0]["x"][1, :6], codes[0].float())


@pytest.mark.parametrize(
    "clip, rounded, expected",
    [
        (False, False, [-1.4, 0.4, 2.4]),
        (True, False, [0.0, 0.4, 2.4]),
        (False, True, [-1.0, 0.0, 2.0]),
        (True, True, [0.0, 0.0, 2.0]),
    ],
)
def test_clipping_and_integer_rounding_can_be_disabled_independently(
    clip: bool, rounded: bool, expected: list[float]
) -> None:
    """Clipping and rounding each follow their own option for nonnegative count histories."""
    model = Probe([[-1.4, 0.4, 2.4]])
    actual = forecast(
        model,
        torch.tensor([[0.0, 1.0, 2.0]]),
        1,
        {**TTA, "tta_sign_flip": False, "clip_nonnegative": clip, "integer_round": rounded},
    )
    torch.testing.assert_close(actual[..., 0], torch.tensor([expected]))


@pytest.mark.parametrize("count", [9, 99])
def test_sign_flip_respects_quantile_count(count: int) -> None:
    """Symmetric nine- and ninety-nine-level grids reverse their full quantile axis."""
    index = torch.arange(count, dtype=torch.float32)
    positive, negative = index.square() + 0.25, index / 2 - 3
    model = Probe([positive.tolist(), negative.tolist()])
    model.quantiles = torch.arange(1, count + 1, dtype=torch.float32) / (count + 1)
    history = torch.tensor([[-0.5, 0.25, -0.25, 0.75]])
    output = forecast(model, history, 3, TTA)
    expected = ((positive - negative.flip(0)) / 2)[None, :, None].expand(1, count, 3)
    torch.testing.assert_close(output, expected)


def test_trends_and_running_totals() -> None:
    """A bent running total is differenced despite a poor linear fit; sparse jumps are not."""
    rows = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 100.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, -1.0, 0.0, 1.0, -1.0],
            [torch.nan, 0.0, 1.0, torch.nan, 2.0, 3.0],
        ]
    )
    expected = torch.tensor([True, True, False, False, True])
    torch.testing.assert_close(_difference_mask(rows, TTA), expected)
    torch.testing.assert_close(_difference_mask(-rows, TTA), expected)
    model = Probe()
    forecast(model, rows[:2], 1, TTA)
    torch.testing.assert_close(
        model.batches[0]["x"][:2, 2:8],
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 96.0],
            ]
        ),
    )


@pytest.mark.parametrize("horizon, cap", [(48, 48), (49, 48)])
def test_differencing_horizon_cap(horizon: int, cap: int) -> None:
    """The cap includes its boundary and disables both trend and running-total differencing."""
    model = Probe()
    model.config.future_len = 64
    history = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 100.0]])
    tta = {**TTA, "differencing_max_horizon": cap}
    result = forecast(model, history, horizon, tta)
    seen = model.batches[0]["x"][:2, :4]
    expected = (
        torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 98.0]]) if horizon <= cap else history
    )
    torch.testing.assert_close(seen, expected)
    assert result.shape == (2, 3, horizon)


def test_difference_band_integrates_median_and_quadrature_spread() -> None:
    """Two unit increments anchor at the last observation with square-root uncertainty."""
    pred = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
    history = torch.tensor([[8.0, 9.0, 10.0, torch.nan]])
    output = _calibrate(pred, history, torch.tensor([True]), TTA)
    root_two = 2**0.5
    expected = torch.tensor([[[10.0, 12.0 - root_two], [11.0, 12.0], [12.0, 12.0 + root_two]]])
    torch.testing.assert_close(output, expected)


def test_categories_fit_history_and_look_up_future() -> None:
    """Future categories use historical target means; unseen codes use the global mean."""
    model = Probe()
    history = torch.tensor([[2.0, 4.0, 6.0]])
    codes = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 2.0]])
    forecast(model, history, 3, TTA, future_covariates=codes, categorical_future=(0,))
    encoded = model.batches[0]["x"][1, 1:7]
    torch.testing.assert_close(encoded, torch.tensor([10 / 3, 10 / 3, 5.0, 10 / 3, 5.0, 4.0]))
    torch.testing.assert_close(codes, torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 2.0]]))


@pytest.mark.parametrize("kind", ["past", "future"])
@pytest.mark.parametrize("base, dtype", [(2**24, torch.float64), (2**60, torch.int64)])
def test_large_category_codes_preserve_identity(kind: str, base: int, dtype: torch.dtype) -> None:
    """Distinct large codes retain their separate, fractional historical target means."""
    model = Probe()
    history = torch.tensor([[2.0, 5.0]])
    codes = torch.tensor([[base, base + 1, base, base + 1]], dtype=dtype)
    if kind == "past":
        codes = codes[:, :2]
    before = codes.clone()
    forecast(
        model,
        history,
        2,
        TTA,
        **{f"{kind}_covariates": codes, f"categorical_{kind}": (0,)},
    )
    encoded = model.batches[0]["x"][1, : codes.shape[1]]
    expected = torch.tensor([2.75, 4.25] * (codes.shape[1] // 2))
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
    torch.testing.assert_close(codes, before)


def test_numpy_integer_category_indices_are_accepted() -> None:
    """NumPy integer indices select categorical rows; boolean indices remain invalid."""
    np = pytest.importorskip("numpy")
    history = torch.tensor([[2.0, 5.0]])
    codes = torch.tensor([[0, 1, 0]])
    for index in (np.int32(0), np.int64(0)):
        model = Probe()
        forecast(model, history, 1, TTA, future_covariates=codes, categorical_future=(index,))
        torch.testing.assert_close(model.batches[0]["x"][1, :3], torch.tensor([2.75, 4.25, 2.75]))
    for index in (False, True, np.bool_(False)):
        with pytest.raises(ValueError, match="indices"):
            forecast(Probe(), history, 1, TTA, future_covariates=codes, categorical_future=(index,))


@pytest.mark.parametrize("categorical", [(), (0,)])
def test_unused_future_tail_is_ignored_before_value_validation(
    categorical: tuple[int, ...],
) -> None:
    """Values beyond the requested horizon cannot invalidate or change a forecast."""
    history = torch.tensor([[-0.5, 0.25]])
    prefix = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    tail = torch.tensor([[torch.inf, -torch.inf, 1e300]], dtype=torch.float64)
    extended = torch.cat([prefix, tail], dim=1)
    short, long = Probe(), Probe()
    expected = forecast(
        short, history, 1, TTA, future_covariates=prefix, categorical_future=categorical
    )
    actual = forecast(
        long, history, 1, TTA, future_covariates=extended, categorical_future=categorical
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(long.batches[0]["x"], short.batches[0]["x"], equal_nan=True)


@pytest.mark.parametrize(
    "history",
    [
        torch.zeros(2, 3),
        torch.full((2, 3), torch.nan),
        torch.tensor([[0.25, torch.nan, -0.5], [3.5, 3.5, 3.5]]),
    ],
)
def test_missing_and_constant_histories_are_finite(history: Tensor) -> None:
    """Missing and constant observations preserve shape, finiteness and sorted quantiles."""
    output = forecast(Probe(), history, 3, TTA)
    assert output.shape == (2, 3, 3)
    assert output.device.type == "cpu" and output.dtype == torch.float32
    assert torch.isfinite(output).all() and (output.diff(dim=1) >= 0).all()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"horizon": 0}, "horizon"),
        ({"horizon": -1}, "horizon"),
        ({"horizon": 1.5}, "horizon"),
        ({"horizon": None}, "horizon"),
        ({"horizon": True}, "horizon"),
        ({"history": torch.ones(4)}, "history"),
        ({"history": torch.tensor([[torch.inf]])}, "finite"),
        ({"future_covariates": torch.ones(1, 4)}, "cover"),
        ({"future_covariates": torch.tensor([[0.0, 0.0, 0.0, 0.0, torch.inf]])}, "finite"),
        (
            {"future_covariates": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1e300]], dtype=torch.float64)},
            "float32 range",
        ),
        ({"past_covariates": torch.ones(1, 5)}, "same history"),
        ({"categorical_past": (0,)}, "requires"),
        ({"future_covariates": torch.ones(1, 5), "categorical_future": (1,)}, "indices"),
    ],
)
def test_invalid_inputs_fail_before_forward(kwargs: dict, message: str) -> None:
    """Invalid geometry and metadata never reach the model."""
    model = Probe()
    arguments = {"history": torch.ones(1, 4), "horizon": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        forecast(model, tta=TTA, **arguments)
    assert not model.batches


def test_nonfinite_model_output_is_rejected() -> None:
    """A broken checkpoint cannot silently return a nonfinite forecast."""
    model = Probe([[0.0, torch.nan, 1.0], [0.0, 0.5, 1.0]])
    with pytest.raises(ValueError, match="invalid forecast"):
        forecast(model, torch.ones(1, 3), 1, TTA)
