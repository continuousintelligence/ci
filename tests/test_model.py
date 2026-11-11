"""Configuration-driven geometry and differentiable model execution."""

import copy

import pytest
import torch

from continuousintelligence.ci_proto._mamba import Mamba3Reference, RMSNorm
from continuousintelligence.ci_proto._model import Model, RMSLayerNorm


@pytest.fixture
def model_config(release_files):
    return torch.load(release_files[0], weights_only=True)["hyper_parameters"]["model"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rank", [None, 1, 2, 4])
def test_mamba_gradients_can_be_enabled_inside_no_grad(device: str, rank: int | None) -> None:
    """The shared recurrence permits finite input and parameter gradients when enabled."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    torch.manual_seed(7)
    cell = Mamba3Reference(
        d_model=8, d_state=8, expand=2, headdim=8, is_mimo=rank is not None, mimo_rank=rank or 1
    ).to(device)
    with torch.no_grad(), torch.enable_grad():
        values = torch.randn(2, 5, 8, device=device, requires_grad=True)
        output = cell(values)
        gradients = torch.autograd.grad(
            output.square().mean(),
            (values, cell.in_proj.weight, cell.dt_bias, cell.out_proj.weight),
        )
    assert output.shape == values.shape
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("width, blocks, patch", [(8, 1, 2), (24, 2, 4)])
def test_model_geometry_comes_from_configuration(model_config, width, blocks, patch) -> None:
    """Changing width, depth and patch geometry produces the configured quantile field."""
    config = copy.deepcopy(model_config)
    config.update(
        embedding_dim=width,
        num_blocks=blocks,
        input_patch_size=patch,
        output_patch_size=patch,
        context_len=4 * patch,
        future_len=2 * patch,
    )
    config["stack_config"]["recipe"] = ["mamba"] * blocks
    model = Model(config).eval()
    with torch.no_grad():
        output = model({"x": torch.linspace(-1, 1, 6 * patch).repeat(2, 1)})
    assert len(model.stack) == blocks
    assert model.input_patch_embedding.output_layer.out_features == width
    assert output.shape == (2, 3, 6 * patch)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("state_dim", [0, 2, 5, 127, 8.0])
def test_invalid_state_dimension_fails_during_construction(model_config, state_dim) -> None:
    """Unsupported state dimensions are rejected when the model is constructed."""
    model_config["stack_config"]["templates"]["mamba"]["time_mixer"]["mamba_state_dim"] = state_dim
    with pytest.raises(ValueError, match="mamba_state_dim must be an even integer"):
        Model(model_config)


@pytest.mark.parametrize("factor", [-1, 0, 0.001, float("nan"), float("inf")])
def test_invalid_output_expansion_fails_during_construction(model_config, factor) -> None:
    """Output expansion must be finite, positive and produce a nonempty hidden layer."""
    model_config["h_expand"] = factor
    with pytest.raises(ValueError, match="h_expand"):
        Model(model_config)


@pytest.mark.parametrize(
    "section, name, value",
    [
        ("time_mixer", "num_heads", 0),
        ("time_mixer", "num_heads", -2),
        ("time_mixer", "mamba_expand", -2),
        ("variate_mixer", "num_heads", -2),
        ("xlstm", "ffn_proj_factor", -1),
        ("xlstm", "ffn_proj_factor", float("nan")),
        ("xlstm", "ffn_round_up_to_multiple_of", 0),
        ("xlstm", "ffn_round_up_to_multiple_of", -8),
    ],
)
def test_invalid_mixer_geometry_fails_during_construction(
    model_config, section, name, value
) -> None:
    """Invalid mixer dimensions fail with an error identifying the configuration field."""
    config = (
        model_config["xlstm"]
        if section == "xlstm"
        else model_config["stack_config"]["templates"]["mamba"][section]
    )
    config[name] = value
    with pytest.raises(ValueError, match=name):
        Model(model_config)


@pytest.mark.parametrize("mask_name", ["future_covariate_mask", "past_covariate_mask"])
def test_declared_roles_require_a_complete_future_window(model_config, mask_name) -> None:
    """Explicit covariate roles require enough input positions for the full future window."""
    model = Model(model_config).eval()
    values = torch.ones(1, model.config.future_len - 1)
    with torch.no_grad():
        assert torch.isfinite(model({"x": values})).all()
        with pytest.raises(ValueError, match="full future window"):
            model(
                {
                    "x": values,
                    "group_vector": torch.zeros(1, dtype=torch.long),
                    "target_mask": torch.ones(1, dtype=torch.bool),
                    mask_name: torch.zeros(1, dtype=torch.bool),
                }
            )


@pytest.mark.parametrize("future_len", [None, 12])
def test_future_target_and_past_covariate_values_are_isolated(model_config, future_len) -> None:
    """Only known future covariates affect predictions, and input tensors stay unchanged."""
    model = Model(model_config).eval()
    window = model.config.future_len if future_len is None else future_len
    values = torch.randn(3, model.config.context_len + window)
    roles = {
        "group_vector": torch.zeros(3, dtype=torch.long),
        "target_mask": torch.tensor([True, False, False]),
        "future_covariate_mask": torch.tensor([False, True, False]),
        "past_covariate_mask": torch.tensor([False, False, True]),
    }
    original = values.clone()
    changed = values.clone()
    changed[[0, 2], -window:] = 1000
    with torch.no_grad():
        expected = model({"x": values, **roles}, future_len=future_len)
        actual = model({"x": changed, **roles}, future_len=future_len)
        changed[1, -window:] += 100
        known_future_changed = model({"x": changed, **roles}, future_len=future_len)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert not torch.allclose(known_future_changed[0], expected[0])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_final_norm_preserves_float32_normalised_activations(dtype) -> None:
    """Final normalisation retains float32 precision for float16 and bfloat16 inputs."""
    values = torch.tensor([[0.3, -0.8, 1.4, 2.1]], dtype=dtype)
    final_norm = RMSLayerNorm(4, eps=1e-6)
    mixer_norm = RMSNorm(4, eps=1e-6)
    expected = values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
    actual = final_norm(values)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, mixer_norm(values))
