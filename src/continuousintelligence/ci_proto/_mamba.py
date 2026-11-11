"""PyTorch Mamba-3 recurrence for CPU and CUDA inference.

Adapted by Continuous Intelligence from Mamba's module and
SISO/MIMO reference calculations:
https://github.com/state-spaces/mamba/tree/a14b1dff0454a3bc27d9eb31355dc01e4b2490ec

RMS normalisation is adapted from xLSTM Large:
https://github.com/NX-AI/xlstm/blob/ab22eadbd245f293dd8dec38ed29963d73758a12/xlstm/xlstm_large/components.py

Copyright (c) 2026, Dao AI Lab, Goombalab.
Copyright 2024 NXAI GmbH and its affiliates 2024
Upstream portions are licensed under Apache-2.0; see NOTICE.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMS norm with optional float32 reductions."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-06,
        force_float32_reductions: bool = True,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.force_float32_reductions = force_float32_reductions
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features)) if use_bias else None

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        if self.force_float32_reductions:
            x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x.to(in_dtype)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def _rotate_pairs(x: Tensor, cos: Tensor, sin: Tensor, interleaved: bool) -> Tensor:
    """Rotate pairs of the last axis; angles beyond ``cos`` leave their pairs unrotated."""
    if interleaved:
        pairs = x.view(*x.shape[:-1], -1, 2)
        first, second = (pairs[..., 0], pairs[..., 1])
    else:
        half = x.shape[-1] // 2
        first, second = (x[..., :half], x[..., half:])
    if cos.shape[-1] < first.shape[-1]:
        pad = first.shape[-1] - cos.shape[-1]
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    if interleaved:
        return torch.stack([rotated_first, rotated_second], dim=-1).view_as(x)
    return torch.cat([rotated_first, rotated_second], dim=-1)


class Mamba3Reference(nn.Module):
    """Mamba-3 block evaluated step by step in plain PyTorch."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
    ) -> None:
        super().__init__()
        if type(d_state) is not int or d_state < 4 or d_state % 2:
            raise ValueError("mamba_state_dim must be an even integer of at least 4")
        self.d_state = d_state
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.d_inner = expand * d_model
        if self.d_inner % headdim != 0:
            raise ValueError(f"expand * d_model {self.d_inner} not divisible by headdim {headdim}")
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        split = d_state // 2 - d_state // 2 % 2
        self.num_rope_angles = split // 2
        bc = self.d_state * self.mimo_rank
        d_in_proj = 2 * self.d_inner + 2 * bc + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, d_state))
        self.B_norm = RMSNorm(d_state, eps=1e-05)
        self.C_norm = RMSNorm(d_state, eps=1e-05)
        if is_mimo:
            shape = (self.nheads, self.mimo_rank, headdim)
            self.mimo_x = nn.Parameter(torch.ones(shape) / self.mimo_rank)
            self.mimo_z = nn.Parameter(torch.ones(shape))
            self.mimo_o = nn.Parameter(torch.ones(shape) / self.mimo_rank)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: Tensor) -> Tensor:
        batch, seq, _ = u.shape
        bc = self.d_state * self.mimo_rank
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            self.in_proj(u),
            [
                self.d_inner,
                self.d_inner,
                bc,
                bc,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).to(torch.float32)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).to(torch.float32)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=1)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=1)
        A = torch.clamp(-F.softplus(dd_A.to(torch.float32)), max=-1e-4)
        DT = F.softplus(dd_dt.to(torch.float32) + self.dt_bias)
        alpha = torch.exp(A * DT)
        gate = torch.sigmoid(trap.to(torch.float32))
        beta = (1.0 - gate) * DT * alpha
        gamma = gate * DT
        k = self.B_norm(B).expand(-1, -1, -1, self.nheads, -1).to(torch.float32)
        q = self.C_norm(C).expand(-1, -1, -1, self.nheads, -1).to(torch.float32)
        k = k + rearrange(self.B_bias, "h r n -> r h n")
        q = q + rearrange(self.C_bias, "h r n -> r h n")
        turns = torch.tanh(angles.to(torch.float32))[:, :, None, :] * DT[..., None] * math.pi
        cumulative = torch.cumsum(turns, dim=1)
        cos, sin = (torch.cos(cumulative)[:, :, None], torch.sin(cumulative)[:, :, None])
        q = _rotate_pairs(q, cos, sin, interleaved=not self.is_mimo)
        k = _rotate_pairs(k, cos, sin, interleaved=not self.is_mimo)
        if self.is_mimo:
            v = torch.einsum("blhp,hrp->blrhp", x, self.mimo_x.to(torch.float32))
            gates = torch.einsum("blhp,hrp->blrhp", z, self.mimo_z.to(torch.float32))
        else:
            v, gates = (x[:, :, None], z[:, :, None])
        state = u.new_zeros((batch, self.nheads, self.headdim, self.d_state), dtype=torch.float32)
        previous = torch.zeros_like(state)
        outputs = []
        for t in range(seq):
            current = torch.einsum("brhn,brhp->bhpn", k[:, t], v[:, t])
            state = (
                alpha[:, t, :, None, None] * state
                + beta[:, t, :, None, None] * previous
                + gamma[:, t, :, None, None] * current
            )
            previous = current
            step = torch.einsum("bhpn,brhn->brhp", state, q[:, t])
            step = step + self.D[None, None, :, None] * v[:, t]
            step = step * F.silu(gates[:, t])
            if self.is_mimo:
                step = torch.einsum("brhp,hrp->bhp", step, self.mimo_o.to(torch.float32))
            else:
                step = step[:, 0]
            outputs.append(step)
        y = rearrange(torch.stack(outputs, dim=1), "b l h p -> b l (h p)")
        return self.out_proj(y.to(u.dtype))


class Mamba3(Mamba3Reference):
    """Checkpoint-compatible cell with fused CUDA inference when configured by the loader."""

    fused = False

    def forward(self, u: Tensor) -> Tensor:
        if self.fused and u.is_cuda and not torch.is_grad_enabled():
            from ._cuda import mamba3_forward

            return mamba3_forward(self, u)
        return super().forward(u)
