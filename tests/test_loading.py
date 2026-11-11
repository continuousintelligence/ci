"""Loading, device selection and the installed public forecasting interface."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from continuousintelligence import ci_proto, load_model
from continuousintelligence._loader import _device
from continuousintelligence.ci_proto._checkpoint import _load_tta
from continuousintelligence.ci_proto._model import Model


def test_loading_and_forecasting(release_files):
    """A configured model loads strictly and produces repeatable ordered CPU quantiles."""
    model = load_model(*release_files, device="cpu")
    assert isinstance(model, ci_proto.Forecaster)
    assert not model._model.training
    history = torch.tensor([[1.2, 3.4, float("nan"), 2.8], [-2.1, 0.4, -1.6, 0.2]])
    before = history.clone()
    actual = model.forecast(history, horizon=5)
    assert model.device == torch.device("cpu")
    assert actual.shape == (2, 3, 5)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert torch.isfinite(actual).all() and not actual.requires_grad
    assert (actual.diff(dim=1) >= 0).all()
    assert (actual[0] >= 0).all()
    torch.testing.assert_close(actual, model.forecast(history, 5), rtol=0, atol=0)
    torch.testing.assert_close(history, before, equal_nan=True)
    torch.testing.assert_close(model.quantiles, torch.tensor([0.1, 0.5, 0.9]))
    model.quantiles.zero_()
    assert model.quantiles[1] == 0.5


@pytest.mark.parametrize("count", [9, 99])
def test_quantile_heads_come_from_checkpoint_metadata(release_files, count):
    """Both head counts reconstruct strictly and return the checkpoint's quantile grid."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    config = payload["hyper_parameters"]["model"]
    config["quantiles"] = [i / (count + 1) for i in range(1, count + 1)]
    payload["state_dict"] = Model(config).state_dict()
    torch.save(payload, checkpoint)
    model = load_model(checkpoint, tta, device="cpu")
    torch.testing.assert_close(model.quantiles, torch.tensor(config["quantiles"]))
    result = model.forecast(torch.tensor([[0.2, -0.8, 0.4, 1.1]]), 3)
    assert result.shape == (1, count, 3)
    assert result.isfinite().all() and (result.diff(dim=1) >= 0).all()


def test_quantile_metadata_must_match_weights(release_files):
    """Changing the recorded grid size without matching weights fails strict loading."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    payload["hyper_parameters"]["model"]["quantiles"] = [i / 10 for i in range(1, 10)]
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_model(checkpoint, tta, device="cpu")


def test_tensor_quantile_metadata_fails_with_value_error(release_files):
    """Quantile levels stored as a tensor are rejected with a clear validation error."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    payload["hyper_parameters"]["model"]["quantiles"] = torch.tensor([0.1, 0.5, 0.9])
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="quantiles"):
        load_model(checkpoint, tta, device="cpu")


@pytest.mark.parametrize("levels", [[0.1, 0.4, 0.9], [0.1, 0.5, 0.8], [0.2, 0.8]])
def test_release_grid_requires_a_median_and_symmetric_levels(release_files, levels):
    """Calibration rejects incompatible grids even when parameter shapes match."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    config = payload["hyper_parameters"]["model"]
    config["quantiles"] = levels
    payload["state_dict"] = Model(config).state_dict()
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="symmetric quantiles"):
        load_model(checkpoint, tta, device="cpu")


@pytest.mark.parametrize("damage", ["missing_config", "missing_weight", "wrong_shape"])
def test_incompatible_checkpoint_fails(release_files, damage):
    """Missing metadata and incompatible parameters cannot silently produce forecasts."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    if damage == "missing_config":
        del payload["hyper_parameters"]
    elif damage == "missing_weight":
        payload["state_dict"].pop("model.input_patch_embedding.hidden_layer.bias")
    else:
        payload["state_dict"]["model.input_patch_embedding.hidden_layer.bias"] = torch.zeros(2)
    torch.save(payload, checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        load_model(checkpoint, tta, device="cpu")


@pytest.mark.parametrize("metadata", [None, [], {"model": None}, {"model": []}])
def test_malformed_checkpoint_metadata_fails_with_value_error(release_files, metadata):
    """Checkpoint metadata must contain a model configuration mapping."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    payload["hyper_parameters"] = metadata
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="hyper_parameters.model"):
        load_model(checkpoint, tta, device="cpu")


@pytest.mark.parametrize("state", [None, {}, {0: torch.zeros(1)}])
def test_malformed_state_dictionary_fails_with_value_error(release_files, state):
    """Weights must be a nonempty mapping with string parameter names."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    payload["state_dict"] = state
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="state_dict"):
        load_model(checkpoint, tta, device="cpu")


@pytest.mark.parametrize(
    "path",
    [
        ("scaler_cfg",),
        ("quantiles",),
        ("stack_config",),
        ("h_expand",),
        ("stack_config", "templates", "mamba", "time_mixer"),
    ],
)
@pytest.mark.parametrize("damage", ["missing", "null", "wrong_type"])
def test_malformed_model_configuration_fails_with_value_error(release_files, path, damage):
    """Missing, null and incorrectly typed architecture settings fail during loading."""
    checkpoint, tta = release_files
    payload = torch.load(checkpoint, weights_only=True)
    config = payload["hyper_parameters"]["model"]
    for key in path[:-1]:
        config = config[key]
    if damage == "missing":
        del config[path[-1]]
    else:
        config[path[-1]] = None if damage == "null" else "invalid"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        load_model(checkpoint, tta, device="cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"tta_diff": "true"},
        {"unknown": 1},
        {"trend_window": 0},
        {"trend_min_r2": 1.1},
        {"raw_band_scale": float("nan")},
        {"diff_band_exponent": -1},
        {"trend_threshold": True},
        {"differencing_max_horizon": 0},
        {"differencing_max_horizon": -1},
        {"differencing_max_horizon": True},
        {"differencing_max_horizon": 48.0},
        {"differencing_max_horizon": "48"},
        {"differencing_max_horizon": None},
        {"bfloat16": 1},
        {"bfloat16": "true"},
        {"bfloat16": None},
        {"calendar_covariates": "true"},
        {"calendar_covariates": 1},
        {"calendar_covariates": None},
        {"calendar_covariates": []},
        {"static_groups": -1},
        {"static_groups": True},
        {"joint_neighbours": 1},
        {"joint_neighbours": 8.0},
        {"chunk_joint": 1},
        {"chunk_joint": -1},
    ],
)
def test_invalid_release_configuration_fails(release_files, change):
    """The loader rejects malformed types and invalid calibration values."""
    _, path = release_files
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data | change))
    with pytest.raises(ValueError):
        _load_tta(path)


@pytest.mark.parametrize(
    "key",
    [
        "diff_band_scale",
        "differencing_max_horizon",
        "bfloat16",
        "calendar_covariates",
        "static_groups",
        "joint_neighbours",
        "chunk_joint",
    ],
)
def test_missing_calibration_fails(release_files, key: str) -> None:
    """Missing fitted settings never fall back to another checkpoint's defaults."""
    _, path = release_files
    data = json.loads(path.read_text())
    del data[key]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing"):
        _load_tta(path)


@pytest.mark.parametrize(
    "key",
    [
        "postprocess",
        "tta_diff",
        "tta_sign_flip",
        "tta_running_totals",
        "target_encode_categoricals",
        "clip_nonnegative",
        "integer_round",
        "bfloat16",
        "calendar_covariates",
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_boolean_options_accept_both_values(release_files, key: str, enabled: bool) -> None:
    """Every boolean option can be explicitly enabled or disabled in the same configuration."""
    _, path = release_files
    data = json.loads(path.read_text())
    data[key] = enabled
    path.write_text(json.dumps(data))
    assert _load_tta(path)[key] is enabled


@pytest.mark.parametrize("bfloat16", [False, True])
def test_release_precision_follows_tta(release_files, bfloat16: bool) -> None:
    """The release file controls autocast while forecasts remain CPU float32 tensors."""
    checkpoint, tta = release_files
    data = json.loads(tta.read_text())
    data.update(differencing_max_horizon=48, bfloat16=bfloat16)
    tta.write_text(json.dumps(data))
    model = load_model(checkpoint, tta, device="cpu")
    dtypes = []
    handle = model._model.input_patch_embedding.hidden_layer.register_forward_hook(
        lambda _module, _inputs, output: dtypes.append(output.dtype)
    )
    try:
        result = model.forecast(torch.tensor([[-0.5, 0.25, -0.25, 0.75]]), 3)
    finally:
        handle.remove()
    assert dtypes == [torch.bfloat16 if bfloat16 else torch.float32]
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert result.isfinite().all() and (result.diff(dim=1) >= 0).all()
    assert not torch.is_autocast_enabled("cpu")


def test_complete_release_tta_settings_are_retained(release_files) -> None:
    """All inference settings survive loading, including calendar and cohort policies."""
    _, path = release_files
    options = {
        "bfloat16": True,
        "calendar_covariates": True,
        "differencing_max_horizon": 48,
        "static_groups": 2,
        "joint_neighbours": 8,
        "chunk_joint": 100,
    }
    path.write_text(json.dumps(json.loads(path.read_text()) | options))
    loaded = _load_tta(path)
    assert {key: loaded[key] for key in options} == options


def test_device_selection(monkeypatch):
    """Automatic detection selects available CUDA, and explicit CPU always stays CPU."""
    for available in (False, True):
        monkeypatch.setattr(torch.cuda, "is_available", lambda available=available: available)
        assert _device("cpu").type == "cpu"
        assert _device("auto").type == ("cuda" if available else "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        _device("cuda")
    with pytest.raises(ValueError, match="device"):
        _device("mps")


def test_cpu_does_not_import_cuda_kernels(release_files):
    """A simple CPU forecast skips kernels and metadata helpers even with release features enabled."""
    _, path = release_files
    settings = json.loads(path.read_text())
    settings.update(calendar_covariates=True, static_groups=2, joint_neighbours=8, chunk_joint=100)
    path.write_text(json.dumps(settings))
    program = """
import sys, importlib.abc
import torch
class NoKernels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'triton', 'tilelang', 'mamba_ssm', 'causal_conv1d', 'xlstm', 'flashrnn'}
                or fullname in {'continuousintelligence.ci_proto._calendar', 'continuousintelligence.ci_proto._groups'}):
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, NoKernels())
from continuousintelligence import load_model
torch.set_num_threads(2)
model = load_model(sys.argv[1], sys.argv[2], device='cpu')
assert torch.isfinite(model.forecast(torch.ones(1, 5), 3)).all()
"""
    subprocess.run([sys.executable, "-c", program, *map(str, release_files)], check=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_forecast_matches_cpu(release_files):
    """The same reference model also executes on CUDA when available."""
    cpu = load_model(*release_files, device="cpu")
    gpu = load_model(*release_files, device="cuda")
    history = torch.tensor([[0.5, -1.2, 1.4, -0.3]])
    actual = gpu.forecast(history, 5)
    torch.testing.assert_close(actual, cpu.forecast(history, 5), atol=5e-4, rtol=5e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_checkpoint_when_supplied(device):
    """Validate local release weights without storing them or their config in the package."""
    checkpoint = os.environ.get("CONTINUOUSINTELLIGENCE_CHECKPOINT")
    tta = os.environ.get("CONTINUOUSINTELLIGENCE_TTA_CONFIG")
    if not checkpoint or not tta:
        pytest.skip(
            "set CONTINUOUSINTELLIGENCE_CHECKPOINT and CONTINUOUSINTELLIGENCE_TTA_CONFIG to check release weights"
        )
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    model = load_model(Path(checkpoint), Path(tta), device=device)
    history = torch.sin(torch.arange(64) * 0.3).repeat(2, 1)
    horizon = model._model.config.future_len
    actual = model.forecast(history, horizon)
    assert actual.shape == (2, len(model.quantiles), horizon)
    assert torch.isfinite(actual).all() and (actual.diff(dim=1) >= 0).all()


@pytest.mark.parametrize("horizon", [9, 321, 1001])
def test_forecast_horizons_beyond_training_window(release_files, horizon):
    """Requests beyond the training window return every requested step without changing the model."""
    model = load_model(*release_files, device="cpu")
    history = torch.tensor([[0.25, -0.5, 1.25, -0.75]])
    short = model.forecast(history, 3)
    actual = model.forecast(history, horizon)
    assert actual.shape == (1, 3, horizon)
    assert actual.isfinite().all() and (actual.diff(dim=1) >= 0).all()
    assert model._model.config.future_len == 8
    torch.testing.assert_close(model.forecast(history, 3), short, rtol=0, atol=0)


def test_default_forecast_horizon_is_320(release_files):
    """Omitting the horizon is equivalent to requesting 320 future steps explicitly."""
    model = load_model(*release_files, device="cpu")
    history = torch.tensor([[0.25, -0.5, 1.25, -0.75]])
    actual = model.forecast(history)
    assert actual.shape == (1, 3, 320)
    torch.testing.assert_close(actual, model.forecast(history, 320), rtol=0, atol=0)
