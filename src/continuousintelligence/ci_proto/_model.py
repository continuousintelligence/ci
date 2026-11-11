"""Continuous Intelligence's patch model and multivariate attention.

Includes model components adapted by Continuous Intelligence from TiRex-2:
https://github.com/NX-AI/tirex-2/tree/bb5427f4a97379a17756e67bc846c4f4b569eb48/src/tirex2/model

The gated feed-forward and RMS normalisation components also derive from xLSTM:
https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12/xlstm/xlstm_large

Copyright (c) NXAI GmbH.
Copyright 2024 NXAI GmbH and its affiliates 2024
Upstream portions are licensed under Apache-2.0; see NOTICE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._mamba import Mamba3, RMSNorm


class ResidualBlock(nn.Module):
    """Patch embedding and quantile projection with a linear residual."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, act: nn.Module) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(self.act(self.hidden_layer(x))) + self.residual_layer(x)


class RMSLayerNorm(nn.Module):
    """Final norm preserving float32 normalisation for reduced-precision activations."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


class GatedFFN(nn.Module):
    """The time mixer's gated feed-forward projection."""

    def __init__(self, dim: int, constants: dict[str, Any]) -> None:
        super().__init__()
        multiple = constants.get("ffn_round_up_to_multiple_of", 64)
        factor = constants.get("ffn_proj_factor", 2.6667)
        if type(multiple) is not int or multiple < 1:
            raise ValueError("ffn_round_up_to_multiple_of must be a positive integer")
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("ffn_proj_factor must be positive and finite")
        width = math.ceil(dim * factor / multiple) * multiple
        bias = constants.get("use_bias", False)
        self.proj_up_gate = nn.Linear(dim, width, bias=bias)
        self.proj_up = nn.Linear(dim, width, bias=bias)
        self.proj_down = nn.Linear(width, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj_down(F.silu(self.proj_up_gate(x)) * self.proj_up(x))


class MambaCell(nn.Module):
    """Keep the checkpoint's cell prefix around the selected Mamba recurrence."""

    def __init__(self, dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        expand, heads = cfg.get("mamba_expand", 2), cfg.get("num_heads", 4)
        if any(type(value) is not int or value < 1 for value in (expand, heads)):
            raise ValueError("mamba_expand and time num_heads must be positive integers")
        if expand * dim % heads:
            raise ValueError("expanded embedding dimension must be divisible by time heads")
        if cfg.get("model_type") != "bi-mamba3" or cfg.get("mamba_conv_width") is not None:
            raise ValueError("checkpoint must use the convolution-free Mamba-3 cell")
        rank = cfg.get("mamba_rank")
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError("mamba_rank must be a positive integer")
        self.mamba = Mamba3(
            d_model=dim,
            d_state=cfg.get("mamba_state_dim", 128),
            expand=expand,
            headdim=expand * dim // heads,
            is_mimo=rank is not None,
            mimo_rank=rank if rank is not None else 1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mamba(x)


class TimeMixer(nn.Module):
    """Causal target mixing and tied bidirectional mixing for known covariates."""

    def __init__(self, dim: int, cfg: dict[str, Any], constants: dict[str, Any]) -> None:
        super().__init__()
        norm = dict(
            eps=constants.get("norm_eps", 1e-6),
            force_float32_reductions=constants.get("norm_reduction_force_float32", True),
            use_bias=constants.get("use_bias", False),
        )
        self.norm_lstm = RMSNorm(dim, **norm)
        self.fwd_cell = MambaCell(dim, cfg)
        self.rev_cell = self.fwd_cell
        self.combination = nn.Linear(2 * dim, dim, bias=False)
        self.norm_ffn = RMSNorm(dim, **norm)
        self.ffn = GatedFFN(dim, constants)

    def forward(
        self, x: Tensor, future_mask: Tensor, reverse_indices: Tensor | None = None
    ) -> Tensor:
        normalised = self.norm_lstm(x)
        indices = torch.where(future_mask)[0] if reverse_indices is None else reverse_indices
        if indices.numel() and not self.training:
            joined = torch.cat([normalised, normalised[indices].flip(1)])
            mixed = self.fwd_cell(joined)
            out, reverse = mixed[: len(x)], mixed[len(x) :].flip(1)
        else:
            out = self.fwd_cell(normalised)
            if indices.numel():
                reverse = self.rev_cell(normalised[indices].flip(1).contiguous()).flip(1)
        if indices.numel():
            mixed = self.combination(torch.cat([out[indices], reverse], dim=-1))
            out = out.clone()
            out[indices] = mixed.to(out.dtype)
        out = x + out
        return out + self.ffn(self.norm_ffn(out))


def _pack_groups(groups: Tensor, targets: Tensor) -> SimpleNamespace | None:
    groups = groups.squeeze(-1) if groups.ndim == 2 else groups
    if groups.shape != targets.shape:
        raise ValueError("group_vector and target_mask must have the same shape")
    _, inverse, counts = torch.unique(groups, return_inverse=True, return_counts=True)
    keep = counts[inverse] > 1
    if not bool(keep.any()):
        return None
    _, group_index, sizes = torch.unique(inverse[keep], return_inverse=True, return_counts=True)
    count, n_groups, width = group_index.numel(), sizes.numel(), int(sizes.max())
    order = torch.argsort(group_index, stable=True)
    first = torch.cumsum(sizes, dim=0) - sizes
    slot = torch.empty_like(group_index)
    slot[order] = torch.arange(count, device=groups.device) - first[group_index[order]]
    flat_idx = group_index * width + slot
    take = torch.full((n_groups * width,), count, device=groups.device)
    take[flat_idx] = torch.arange(count, device=groups.device)
    valid = torch.zeros(n_groups, width, dtype=torch.bool, device=groups.device)
    valid[group_index, slot] = True
    target = torch.zeros_like(valid)
    target[group_index, slot] = targets[keep]
    allowed = valid[:, :, None] & valid[:, None, :]
    allowed &= ~(~target[:, :, None] & target[:, None, :])
    allowed |= ~valid[:, :, None] & torch.eye(width, dtype=torch.bool, device=groups.device)
    return SimpleNamespace(
        regular=False,
        n_groups=n_groups,
        keep=keep,
        take=take,
        flat_idx=flat_idx,
        width=width,
        bias=torch.where(allowed, 0.0, -torch.inf)[:, None],
    )


@dataclass(frozen=True)
class ForecastLayout:
    """Cached row roles and attention geometry; contained tensors are read-only."""

    groups: Tensor
    targets: Tensor
    future: Tensor
    past: Tensor
    reverse_indices: Tensor
    packing: SimpleNamespace | None


@lru_cache(maxsize=32)
def forecast_layout(
    samples: int, channels: int, targets: int, known: int, device: torch.device
) -> ForecastLayout:
    """Reuse geometry for independent samples with equal target and covariate counts."""
    if samples < 1 or not 0 < targets <= channels or not 0 <= known <= channels - targets:
        raise ValueError("forecast layout needs samples and valid channel roles")
    channel = torch.arange(channels, device=device)
    target = channel < targets
    future = (channel >= targets) & (channel < targets + known)
    allowed = ~(~target[:, None] & target[None, :])
    packing = (
        None
        if channels == 1
        else SimpleNamespace(
            regular=True,
            n_groups=samples,
            width=channels,
            bias=torch.where(allowed, 0.0, -torch.inf)[None, None],
        )
    )
    reverse = (
        torch.arange(samples, device=device)[:, None] * channels
        + torch.arange(targets, targets + known, device=device)[None, :]
    ).flatten()
    return ForecastLayout(
        torch.arange(samples, device=device).repeat_interleave(channels),
        target.repeat(samples),
        future.repeat(samples),
        (~(target | future)).repeat(samples),
        reverse,
        packing,
    )


class GroupedAttention(nn.Module):
    """Attention within each multivariate sample, with asymmetric covariate masking."""

    def __init__(self, dim: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.heads = cfg.get("num_heads", 4)
        if type(self.heads) is not int or self.heads < 1:
            raise ValueError("variate num_heads must be a positive integer")
        if dim % self.heads:
            raise ValueError("embedding dimension must be divisible by variate heads")
        self.head_dim = dim // self.heads
        self.use_qk_norm = cfg.get("use_qk_norm", True)
        self.WK = nn.Linear(dim, dim, bias=False)
        self.WQ = nn.Linear(dim, dim, bias=False)
        self.WV = nn.Linear(dim, dim, bias=False)
        self.WO = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, packing: SimpleNamespace) -> Tensor:
        length, rows, _ = x.shape
        heads, hd, width = self.heads, self.head_dim, packing.width
        n_groups = packing.n_groups
        q = self.WQ(x).view(length, rows, heads, hd)
        k = self.WK(x).view(length, rows, heads, hd)
        v = self.WV(x).view(length, rows, heads, hd)
        if self.use_qk_norm:
            q = F.rms_norm(q, (hd,)).to(v.dtype)
            k = F.rms_norm(k, (hd,)).to(v.dtype)
        qkv = torch.cat((q, k, v), dim=2).flatten(2)
        if not packing.regular:
            qkv = F.pad(qkv, (0, 0, 0, 1)).index_select(1, packing.take)
        packed = qkv.view(length, n_groups, width, 3, heads, hd)
        q = packed[:, :, :, 0].permute(0, 1, 3, 2, 4)
        k = packed[:, :, :, 1].permute(0, 1, 3, 2, 4)
        v = packed[:, :, :, 2].permute(0, 1, 3, 2, 4)
        scores = q @ k.transpose(-1, -2) * (1 / math.sqrt(hd)) + packing.bias.to(q.dtype)
        out = (torch.softmax(scores, dim=-1) @ v).permute(0, 1, 3, 2, 4)
        out = out.reshape(length, n_groups * width, heads * hd)
        return self.WO(out if packing.regular else out.index_select(1, packing.flat_idx))


class MLP(nn.Module):
    """The variate mixer's feed-forward projection."""

    def __init__(self, dim: int, act: nn.Module) -> None:
        super().__init__()
        self.wi = nn.Linear(dim, 4 * dim, bias=False)
        self.wo = nn.Linear(4 * dim, dim, bias=False)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return self.wo(self.act(self.wi(x)))


class VariateMixer(nn.Module):
    """Mix related rows; singleton groups pass through unchanged."""

    def __init__(self, dim: int, cfg: dict[str, Any], act: nn.Module) -> None:
        super().__init__()
        self.norm_attn = nn.RMSNorm(dim, eps=1e-6)
        self.attn = GroupedAttention(dim, cfg)
        self.norm_ffn = nn.RMSNorm(dim, eps=1e-6)
        self.ffn = MLP(dim, act)

    def forward(self, x: Tensor, packing: SimpleNamespace | None) -> Tensor:
        if packing is None:
            return x
        kept = x if packing.regular else x[:, packing.keep]
        kept = kept + self.attn(self.norm_attn(kept), packing)
        kept = kept + self.ffn(self.norm_ffn(kept))
        if packing.regular:
            return kept
        out = x.clone()
        out[:, packing.keep] = kept
        return out


class Block(nn.Module):
    """The time mixer followed by its variate mixer."""

    def __init__(
        self,
        dim: int,
        cfg: dict[str, Any],
        constants: dict[str, Any],
        act: nn.Module,
    ) -> None:
        super().__init__()
        if cfg.get("mixing_order", "time_first") != "time_first":
            raise ValueError("checkpoint must use time_first mixing")
        self.time_mixer = TimeMixer(dim, cfg["time_mixer"], constants)
        self.variate_mixer = VariateMixer(dim, cfg["variate_mixer"], act)

    def forward(
        self,
        x: Tensor,
        packing: SimpleNamespace | None,
        future_mask: Tensor,
        reverse_indices: Tensor,
    ) -> Tensor:
        x = self.time_mixer(x, future_mask, reverse_indices)
        return self.variate_mixer(x.transpose(0, 1), packing).transpose(0, 1)


class Model(nn.Module):
    """Inference-only model; dimensions come from the checkpoint configuration.

    ``forward`` maps flattened ``[variables, time]`` rows to
    ``[variables, quantiles, padded_time]``. Each output patch predicts the next input patch.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__()
        self.config = SimpleNamespace(**raw)
        cfg = self.config
        if raw.get("input_embedding") is not None or raw.get("quantile_mlp", False):
            raise ValueError("checkpoint must use the patch embeddings")
        if any(raw["scaler_cfg"].get(key, False) for key in ("causal_patch", "known_whole_row")):
            raise ValueError("checkpoint must use whole-row observed-value scaling")
        for name in (
            "embedding_dim",
            "input_patch_size",
            "output_patch_size",
            "context_len",
            "future_len",
            "num_blocks",
            "input_ff_dim",
        ):
            if name == "context_len" and raw[name] is None:
                continue
            if type(raw[name]) is not int or raw[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(cfg.h_expand) or cfg.h_expand <= 0:
            raise ValueError("h_expand must be positive and finite")
        if int(cfg.embedding_dim * cfg.h_expand) < 1:
            raise ValueError("h_expand must produce at least one hidden feature")
        if cfg.input_patch_size != cfg.output_patch_size:
            raise ValueError("input and output patch sizes must match")
        if (
            not isinstance(cfg.quantiles, (list, tuple))
            or not cfg.quantiles
            or any(not 0 < q < 1 for q in cfg.quantiles)
        ):
            raise ValueError("quantiles must be levels in (0, 1)")
        if any(b <= a for a, b in zip(cfg.quantiles, cfg.quantiles[1:], strict=False)):
            raise ValueError("quantiles must be strictly increasing")
        recipe = cfg.stack_config["recipe"]
        if len(recipe) != cfg.num_blocks:
            raise ValueError("recipe length must equal num_blocks")
        activation = raw.get("act_func", "SiLU").removeprefix("torch.nn.")
        if activation not in ("SiLU", "GELU", "ReLU"):
            raise ValueError(f"unsupported activation {activation!r}")
        act = getattr(nn, activation)()
        dim = cfg.embedding_dim
        self.input_patch_embedding = ResidualBlock(
            cfg.input_patch_size * 2, cfg.input_ff_dim, dim, act
        )
        self.stack = nn.ModuleList(
            Block(dim, cfg.stack_config["templates"][name], raw.get("xlstm", {}), act)
            for name in recipe
        )
        norm = raw.get("stack_out_norm_config")
        self.stack_out_norm = (
            nn.Identity()
            if norm is None
            else nn.RMSNorm(dim, eps=norm.get("eps", 1e-7))
            if norm.get("use_rmsnorm", False)
            else RMSLayerNorm(dim, norm.get("eps", 1e-7))
        )
        self.register_buffer("quantiles", torch.tensor(cfg.quantiles), persistent=False)
        self.output_patch_embedding = ResidualBlock(
            dim,
            int(dim * cfg.h_expand),
            len(cfg.quantiles) * cfg.output_patch_size,
            act,
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        *,
        future_len: int | None = None,
        horizon: int | None = None,
        layout: ForecastLayout | None = None,
    ) -> Tensor:
        """Run inference, projecting only forecast tokens when ``horizon`` is supplied."""
        x = batch["x"]
        future_len = self.config.future_len if future_len is None else future_len
        groups, targets = batch.get("group_vector"), batch.get("target_mask")
        future, past = batch.get("future_covariate_mask"), batch.get("past_covariate_mask")
        if groups is None:
            if targets is not None:
                raise ValueError("target_mask requires group_vector")
            groups = torch.arange(x.shape[0], device=x.device)
        if targets is None:
            if future is not None or past is not None:
                raise ValueError("covariate masks require target_mask")
            targets = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        declared = future is not None or past is not None
        future = torch.zeros_like(targets) if future is None else future
        past = ~(targets | future) if past is None else past
        if layout is not None:
            if (
                groups is not layout.groups
                or targets is not layout.targets
                or future is not layout.future
                or past is not layout.past
                or groups.shape != (x.shape[0],)
                or groups.device != x.device
            ):
                raise ValueError("forecast layout must match the batch masks and rows")
        else:
            if bool((future & targets).any()) or bool((past & (targets | future)).any()):
                raise ValueError("target and covariate masks must not overlap")
            if not bool((targets | future | past).all()):
                raise ValueError("target and covariate masks must cover every row")
        if horizon is not None and (type(horizon) is not int or not 0 < horizon <= future_len):
            raise ValueError("horizon must be a positive integer within the future window")
        if declared:
            if x.shape[-1] < future_len:
                raise ValueError("inputs with covariate masks must include the full future window")
            window = torch.zeros_like(x, dtype=torch.bool)
            window[:, -future_len:] = (targets | past)[:, None]
            x = torch.where(window, torch.nan, x)

        scaler = self.config.scaler_cfg
        wide = x.double()
        loc = torch.nan_to_num(torch.nanmean(wide, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num(
            torch.nanmean((wide - loc) ** 2, dim=-1, keepdim=True).sqrt(), nan=1.0
        )
        scale = scale.clamp_min(scaler.get("eps", 1e-8))
        binary = torch.zeros_like(loc, dtype=torch.bool)
        if scaler.get("binaryaware", False):
            binary = ((x == 0) | (x == 1) | torch.isnan(x)).all(dim=-1, keepdim=True)
            loc = torch.where(binary, 0.0, loc)
            scale = torch.where(binary, 1.0, scale)
        values = ((wide - loc) / scale).to(x.dtype)
        loc, scale = loc.to(x.dtype), scale.to(x.dtype)
        if scaler.get("use_arcsinh", False):
            values = torch.where(binary, values, values.asinh())
        patch = self.config.input_patch_size
        values = F.pad(values, ((-values.shape[-1]) % patch, 0), value=torch.nan)
        values = values.unfold(-1, patch, patch)
        present = (~torch.isnan(values)).to(values.dtype)
        x = self.input_patch_embedding(
            torch.cat((torch.nan_to_num(values, nan=0.0), present), dim=-1)
        )
        packing = _pack_groups(groups, targets) if layout is None else layout.packing
        reverse_indices = torch.where(future)[0] if layout is None else layout.reverse_indices
        for block in self.stack:
            x = block(x, packing, future, reverse_indices)
        if horizon is not None:
            start = x.shape[1] * patch - future_len - patch
            if start < 0:
                raise ValueError("forecast projection needs a context patch before the future")
            offset = start % patch
            x = x[:, start // patch : -(-(start + horizon) // patch)]
        x = self.output_patch_embedding(self.stack_out_norm(x))
        x = x.unflatten(-1, (len(self.config.quantiles), self.config.output_patch_size))
        x = x.transpose(1, 2).flatten(2)
        if horizon is not None:
            x = x[..., offset : offset + horizon]
        if scaler.get("use_arcsinh", False):
            x = torch.where(binary[:, None], x, torch.sinh(x.clamp(-20.0, 20.0)))
        return x * scale[:, None] + loc[:, None]
