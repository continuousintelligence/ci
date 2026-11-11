"""Small checkpoints keep the package checks independent of release weights."""

import json

import pytest
import torch

from continuousintelligence.ci_proto._model import Model


@pytest.fixture
def release_files(tmp_path):
    """Write an intentionally small model and neutral calibration for tests."""
    torch.set_num_threads(2)
    config = {
        "num_blocks": 1,
        "embedding_dim": 16,
        "input_patch_size": 4,
        "output_patch_size": 4,
        "context_len": 16,
        "future_len": 8,
        "quantiles": [0.1, 0.5, 0.9],
        "input_ff_dim": 24,
        "h_expand": 2,
        "dropout": 0.0,
        "scaler_cfg": {"eps": 1e-8, "use_arcsinh": True, "binaryaware": True},
        "stack_out_norm_config": {"eps": 1e-7, "use_rmsnorm": False},
        "stack_config": {
            "recipe": ["mamba"],
            "templates": {
                "mamba": {
                    "time_mixer": {
                        "model_type": "bi-mamba3",
                        "num_heads": 2,
                        "mamba_state_dim": 8,
                        "mamba_expand": 2,
                        "mamba_rank": 2,
                    },
                    "variate_mixer": {"num_heads": 2, "use_qk_norm": True},
                }
            },
        },
        "xlstm": {"ffn_proj_factor": 1, "ffn_round_up_to_multiple_of": 8},
    }
    torch.manual_seed(11)
    model = Model(config)
    payload = {
        "hyper_parameters": {"model": config},
        "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
    }
    checkpoint = tmp_path / "tiny.ckpt"
    torch.save(payload, checkpoint)
    tta = {
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
        "trend_min_r2": 0.9,
        "trend_window": 1.0,
        "diff_band_exponent": 0.5,
        "diff_band_scale": 1.0,
        "raw_band_exponent": 0.5,
        "raw_band_scale": 1.0,
    }
    tta_path = tmp_path / "tta.json"
    tta_path.write_text(json.dumps(tta))
    return checkpoint, tta_path
