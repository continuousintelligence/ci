"""Automatic selection and inference wrapper for the bundled Mamba-3 MIMO kernels.

Includes Mamba-3 forward integration adapted by Continuous Intelligence:
https://github.com/state-spaces/mamba/tree/a14b1dff0454a3bc27d9eb31355dc01e4b2490ec

Copyright (c) 2026, Dao AI Lab, Goombalab.
Upstream portions are licensed under Apache-2.0; see NOTICE.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from ._mamba import Mamba3


def configure_backend(model: nn.Module) -> None:
    """Enable fused inference only for supported hardware and cell geometry."""
    model.backend = "torch"
    device = next(model.parameters()).device
    if device.type != "cuda" or torch.version.hip is not None:
        return
    cells = [cell for cell in model.modules() if isinstance(cell, Mamba3)]
    if torch.cuda.get_device_capability(device)[0] < 8 or not cells:
        return
    # Float32 tiles need more shared memory than bf16. Use smaller chunks on
    # Ampere and consumer GPUs.
    chunk_budget = (
        64
        if torch.cuda.get_device_properties(device).shared_memory_per_block_optin >= 196608
        else 32
    )
    if any(
        not cell.is_mimo
        or cell.mimo_rank not in (1, 2, 4, 8)
        or cell.d_state not in (32, 64, 128)
        or cell.headdim not in (32, 64, 128)
        or max(8, chunk_budget // cell.mimo_rank) * cell.mimo_rank > chunk_budget
        for cell in cells
    ):
        return
    try:
        from .._vendor.mamba3.mamba3_mimo_fwd import mamba_mimo_forward  # noqa: F401
    except (ImportError, OSError) as error:
        warnings.warn(
            f"Fused Mamba-3 is unavailable; using portable PyTorch on CUDA: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    for cell in cells:
        cell.chunk_size = max(8, chunk_budget // cell.mimo_rank)
        cell.fused = True
    model.backend = "mamba3"


def mamba3_forward(cell: Mamba3, u: Tensor) -> Tensor:
    """Run the forward path, including rank-scaled chunk padding."""
    from .._vendor.mamba3.angle_dt import angle_dt_fwd
    from .._vendor.mamba3.layernorm_gated import _layer_norm_fwd
    from .._vendor.mamba3.mamba3_mimo_fwd import mamba_mimo_forward
    from .._vendor.mamba3.mamba3_mimo_utils import compute_dacs_segsum_triton

    seq = u.shape[1]
    chunk = cell.chunk_size
    u = F.pad(u, (0, 0, 0, (-seq) % chunk))
    bc = cell.d_state * cell.mimo_rank
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        cell.in_proj(u),
        [
            cell.d_inner,
            cell.d_inner,
            bc,
            bc,
            cell.nheads,
            cell.nheads,
            cell.nheads,
            cell.num_rope_angles,
        ],
        dim=-1,
    )
    x = rearrange(x, "b l (h p) -> b l h p", p=cell.headdim)
    z = rearrange(z, "b l (h p) -> b l h p", p=cell.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=cell.mimo_rank, g=1)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=cell.mimo_rank, g=1)
    # Keep the kernel's reduction and rounding order, including 1 / sqrt.
    B = _layer_norm_fwd(
        B.reshape(-1, cell.d_state),
        cell.B_norm.weight,
        None,
        cell.B_norm.eps,
        is_rms_norm=True,
    )[0].view_as(B)
    C = _layer_norm_fwd(
        C.reshape(-1, cell.d_state),
        cell.C_norm.weight,
        None,
        cell.C_norm.eps,
        is_rms_norm=True,
    )[0].view_as(C)
    A = (-F.softplus(dd_A.float())).clamp(max=-1e-4)
    dt = F.softplus(dd_dt + cell.dt_bias)
    adt = rearrange(A * dt, "b l h -> b h l").contiguous()
    dt = rearrange(dt, "b l h -> b h l").contiguous()
    angles = angles[:, :, None].expand(-1, -1, cell.nheads, -1).float().contiguous()
    with torch.cuda.device(u.device):
        cumulative, _ = angle_dt_fwd(angles, dt, chunk_size=chunk, return_output_state=True)
        da_cs, da_cs_rev, segsum = compute_dacs_segsum_triton(adt, chunk)
        y, _, _ = mamba_mimo_forward(
            C.contiguous(),
            B.contiguous(),
            x.contiguous(),
            cell.C_bias,
            cell.B_bias,
            cell.mimo_x,
            cell.mimo_o,
            z.contiguous(),
            cell.D,
            cell.mimo_z,
            cumulative,
            da_cs,
            da_cs_rev,
            dt,
            rearrange(trap, "b l h -> b h l").contiguous(),
            segsum,
            chunk_size=chunk,
            rotary_dim_divisor=4,
            dtype=x.dtype,
        )
    return cell.out_proj(rearrange(y, "b l h p -> b l (h p)").to(u.dtype))[:, :seq]
