"""Real fused-kernel execution, padding, device placement and JIT cache reuse."""

import pytest
import torch

from continuousintelligence.ci_proto._cuda import configure_backend
from continuousintelligence.ci_proto._mamba import Mamba3, Mamba3Reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
@pytest.mark.parametrize("rank", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_mimo_matches_portable_and_reuses_kernel(rank, dtype):
    """Whole and partial chunks use the fused kernel and preserve the recurrence."""
    torch.manual_seed(7)
    cell = Mamba3(64, d_state=128, headdim=64, is_mimo=True, mimo_rank=rank).cuda().eval()
    configure_backend(cell)
    if cell.backend != "mamba3":
        pytest.skip("Fused Mamba-3 is unavailable on this system")
    if (
        torch.cuda.get_device_properties(cell.in_proj.weight.device).shared_memory_per_block_optin
        >= 196608
    ):
        assert cell.chunk_size == max(8, 64 // rank)
    reference = Mamba3Reference(64, d_state=128, headdim=64, is_mimo=True, mimo_rank=rank)
    reference.load_state_dict(cell.state_dict(), strict=True)
    reference.cuda().eval()
    from continuousintelligence._vendor.mamba3.mamba3_mimo_fwd import mamba_mimo_fwd

    with torch.no_grad():
        for length in (17, 32):
            values = torch.randn(2, length, 64, device="cuda")
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                actual = cell(values)
            expected = reference(values)
            assert actual.shape == values.shape
            # Allow tensor-core rounding, with a tighter bound for release precision.
            relative_rmse = (actual.float() - expected).norm() / expected.norm()
            assert relative_rmse < (0.003 if dtype == torch.float32 else 0.01)
        cache_size = len(mamba_mimo_fwd._kernel_cache)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            torch.testing.assert_close(cell(values), actual, rtol=0, atol=0)
        assert len(mamba_mimo_fwd._kernel_cache) == cache_size


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
def test_fused_uses_inputs_cuda_device():
    """Kernel outputs are allocated beside inputs even when another device is current."""
    cell = Mamba3(64, d_state=128, headdim=64, is_mimo=True).to("cuda:1").eval()
    configure_backend(cell)
    if cell.backend != "mamba3":
        pytest.skip("Fused Mamba-3 is unavailable on this system")
    current = torch.cuda.current_device()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = cell(torch.randn(1, 17, 64, device="cuda:1"))
    assert output.device == torch.device("cuda:1")
    assert torch.cuda.current_device() == current
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_smaller_shared_memory_selects_smaller_chunks(monkeypatch):
    """The Ampere memory budget selects causal padding that still runs correctly."""
    cell = Mamba3(64, d_state=128, headdim=64, is_mimo=True).cuda().eval()
    actual_properties = torch.cuda.get_device_properties(cell.in_proj.weight.device)

    class SmallerMemory:
        shared_memory_per_block_optin = 163840

        def __getattr__(self, name):
            return getattr(actual_properties, name)

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SmallerMemory(),
    )
    configure_backend(cell)
    if cell.backend != "mamba3":
        pytest.skip("Fused Mamba-3 is unavailable on this system")
    assert cell.chunk_size == 8
    # Restore real device properties before Triton queries its launch configuration.
    monkeypatch.undo()
    values = torch.randn(1, 17, 64, device="cuda")
    reference = Mamba3Reference(64, d_state=128, headdim=64, is_mimo=True).cuda().eval()
    reference.load_state_dict(cell.state_dict())
    with torch.no_grad():
        output = cell(values)
        expected = reference(values)
    assert (output - expected).norm() / expected.norm() < 0.003


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_time_directions_match_separate_fused_calls(dtype):
    """Shared forward/reverse batching retains fused Mamba outputs and kernel cache reuse."""
    import copy

    from continuousintelligence.ci_proto._model import TimeMixer

    torch.manual_seed(17)
    mixer = (
        TimeMixer(
            64,
            {
                "model_type": "bi-mamba3",
                "num_heads": 2,
                "mamba_state_dim": 128,
                "mamba_expand": 2,
                "mamba_rank": 4,
            },
            {"ffn_proj_factor": 1, "ffn_round_up_to_multiple_of": 8},
        )
        .cuda()
        .eval()
    )
    configure_backend(mixer)
    assert mixer.backend == "mamba3"
    separate = copy.deepcopy(mixer).train()
    values = torch.randn(3, 17, 64, device="cuda")
    mask = torch.tensor([False, True, True], device="cuda")
    from continuousintelligence._vendor.mamba3.mamba3_mimo_fwd import mamba_mimo_fwd

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        actual = mixer(values, mask)
        expected = separate(values, mask)
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.001
        cache_size = len(mamba_mimo_fwd._kernel_cache)
        torch.testing.assert_close(mixer(values, mask), actual, rtol=0, atol=0)
        assert len(mamba_mimo_fwd._kernel_cache) == cache_size
