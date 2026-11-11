"""Reconstruct ci-proto weights, calibration and inference backend."""

import json
import math
from pathlib import Path

import torch

from ._forecast import Forecaster
from ._model import Model


def _load_tta(path: str | Path) -> dict[str, float]:
    booleans = {
        "postprocess",
        "tta_diff",
        "tta_sign_flip",
        "tta_running_totals",
        "target_encode_categoricals",
        "clip_nonnegative",
        "integer_round",
        "bfloat16",
        "calendar_covariates",
    }
    constants = {
        "trend_threshold",
        "trend_min_r2",
        "trend_window",
        "diff_band_exponent",
        "diff_band_scale",
        "raw_band_exponent",
        "raw_band_scale",
        "differencing_max_horizon",
        "static_groups",
        "joint_neighbours",
        "chunk_joint",
    }
    required = booleans | constants
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("tta config must contain a JSON object")
    missing = required - raw.keys()
    unknown = raw.keys() - required
    if missing or unknown:
        raise ValueError(
            f"invalid tta fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    result = {}
    for key in booleans:
        if type(raw[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
        result[key] = raw[key]
    for key in constants:
        value = raw[key]
        if key in ("differencing_max_horizon", "static_groups", "joint_neighbours", "chunk_joint"):
            valid = (
                type(value) is int
                and value >= (1 if key == "differencing_max_horizon" else 0)
                and not (key in ("joint_neighbours", "chunk_joint") and value == 1)
            )
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            valid = value >= 0 if key.endswith("exponent") else value > 0
            if key in ("trend_min_r2", "trend_window"):
                valid = 0 < value <= 1
            value = float(value)
        if not valid:
            raise ValueError(f"invalid {key}: {value}")
        result[key] = value
    return result


def from_checkpoint(payload: object, tta_config: str | Path, device: torch.device) -> Forecaster:
    """Build ci-proto from a loaded checkpoint and its calibration."""
    tta = _load_tta(tta_config)
    metadata = payload.get("hyper_parameters") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or not isinstance(metadata.get("model"), dict):
        raise ValueError("checkpoint must contain a hyper_parameters.model mapping")
    try:
        model = Model(metadata["model"])
    except (KeyError, AttributeError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"invalid checkpoint model configuration: {error}") from error
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state or any(not isinstance(key, str) for key in state):
        raise ValueError("checkpoint must contain a nonempty state_dict with string keys")
    if all(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    levels = model.quantiles
    if (
        len(levels) % 2 == 0
        or not torch.isclose(levels[len(levels) // 2], torch.tensor(0.5), atol=1e-6, rtol=0)
        or not torch.allclose(
            levels,
            1 - levels.flip(0),
            atol=1e-6,
            rtol=0,
        )
    ):
        raise ValueError("ci-proto TTA requires symmetric quantiles centred on the median")
    model = model.to(device).eval()
    from ._cuda import configure_backend

    configure_backend(model)
    return Forecaster(model, tta)
