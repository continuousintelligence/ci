"""The ci-proto preprocessing and quantile forecasting path.

Includes postprocessing adapted by Continuous Intelligence from TiRex-2:
https://github.com/NX-AI/tirex-2/blob/bb5427f4a97379a17756e67bc846c4f4b569eb48/src/tirex2/model/component/postprocessor.py

Copyright (c) NXAI GmbH.
Upstream portions are licensed under Apache-2.0; see NOTICE.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn.functional import pad

from ._model import Model, forecast_layout


class Forecaster:
    """ci-proto with preprocessing and calibration."""

    def __init__(self, model: Model, tta: dict[str, float]) -> None:
        self._model = model
        self._tta = tta
        self._long_model: Callable[[], Forecaster] | None = None
        self._info: dict | None = None

    @property
    def device(self) -> torch.device:
        """Device used for the neural network."""
        return next(self._model.parameters()).device

    @property
    def quantiles(self) -> Tensor:
        """CPU tensor of quantile levels, shape ``[quantiles]``.

        ci-proto returns nine quantiles between 0.1 and 0.9 in increments of 0.1.
        """
        return self._model.quantiles.detach().cpu().clone()

    @property
    def info(self) -> dict | None:
        """Model description from the bundle's ``config.json``: name, parameter count, quantile
        levels and licence. ``None`` when loaded without one."""
        return None if self._info is None else json.loads(json.dumps(self._info))

    @property
    def backend(self) -> str:
        """Selected inference backend: ``mamba3`` (fused CUDA) or ``torch`` (portable)."""
        return getattr(self._model, "backend", "torch")

    def forecast(
        self,
        history: Tensor,
        horizon: int = 320,
        *,
        past_covariates: Tensor | None = None,
        future_covariates: Tensor | None = None,
        categorical_past: tuple[int, ...] = (),
        categorical_future: tuple[int, ...] = (),
        timestamps: Tensor | None = None,
        freq: str | None = None,
        series_ids: Sequence[str] | None = None,
        static_features: Mapping[str, Sequence[object]] | None = None,
    ) -> Tensor:
        """Forecast joint variables, or independent series identified by ``series_ids``.

        ``history``: float tensor ``[variables, time]``; NaN denotes missing data.
        ``past_covariates``: ``[past_variables, time]``.
        ``future_covariates``: ``[known_variables, time + horizon]``; extra future
        steps are ignored. Only supply values known at forecast time. Categorical
        row indices refer to numeric category codes in the corresponding tensor.

        ``timestamps``: int64 epoch nanoseconds over history plus horizon, shared
        ``[time + horizon]`` or per-series ``[variables, time + horizon]``. Supply
        ``freq`` to generate ci-proto's calendar features. Without timestamps,
        calendar features are omitted. Leading padding uses int64's minimum value.

        ``series_ids`` enables cohort policies for independent univariate rows.
        Supply the complete cohort for a window. ``static_features`` maps column
        names to values in input row order. Cohort covariates can also be shaped
        ``[variables, covariates, time]``, including the horizon for known values.

        Returns CPU float32 quantiles ``[variables, quantiles, horizon]``, in
        original units, with levels given by ``self.quantiles``. ci-proto returns
        nine quantiles between 0.1 and 0.9 in increments of 0.1. Horizon is any
        positive integer and defaults to 320.
        """
        model, tta = self._model, self._tta
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if (
            self._long_model is not None
            and isinstance(history, Tensor)
            and history.ndim == 2
            and history.shape[-1] > model.config.context_len
        ):
            long = self._long_model()
            model, tta = long._model, long._tta
        device = next(model.parameters()).device
        history = _rows(history, "history", device)
        count, length = history.shape
        stamps = _timestamps(timestamps, freq, count, length, horizon)
        if stamps is not None and tta["calendar_covariates"] and freq is None:
            raise ValueError("freq is required to generate calendar covariates")
        if series_ids is not None:
            past = _cohort_covariates(
                past_covariates, "past_covariates", count, length, 0, device, categorical_past
            )
            future = _cohort_covariates(
                future_covariates,
                "future_covariates",
                count,
                length,
                horizon,
                device,
                categorical_future,
            )
            cases = _cohort_cases(
                history,
                horizon,
                tta,
                past,
                future,
                categorical_past,
                categorical_future,
                stamps,
                freq,
                series_ids,
                static_features,
            )
            return _run_cases(model, cases, count, horizon, tta)
        if static_features is not None:
            raise ValueError("static_features requires series_ids")
        if stamps is not None:
            if not torch.equal(stamps, stamps[:1].expand_as(stamps)):
                raise ValueError("different timestamp axes require series_ids")
            stamps = stamps[0]
            start = int((stamps != torch.iinfo(torch.int64).min).long().argmax())
            history, stamps = history[:, start:], stamps[start:]
            if past_covariates is not None:
                past_covariates = past_covariates[:, start:]
            if future_covariates is not None:
                future_covariates = future_covariates[:, start:]
        context, future = _context(
            history,
            horizon,
            past_covariates,
            future_covariates,
            categorical_past,
            categorical_future,
            bool(tta["target_encode_categoricals"]),
        )
        context, future = _with_calendar(
            context, future, count, stamps, freq, bool(tta["calendar_covariates"])
        )
        return _predict(model, context[None], future[None], count, horizon, tta)[0]


def _rows(
    value: Tensor,
    name: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = torch.float32,
    length: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty [variables, time] tensor")
    value = value[:, :length]
    if value.is_complex() or bool(torch.isinf(value).any()):
        raise ValueError(f"{name} must contain real finite values or nan")
    result = value.to(device=device, dtype=dtype)
    if bool(torch.isinf(result).any()):
        raise ValueError(f"{name} exceeds the float32 range")
    return result


def _target_encode(context: Tensor, codes: Tensor) -> Tensor:
    """Replace categories with smoothed target means fitted only on history."""
    if codes.dtype == torch.bool:
        codes = codes.to(torch.int64)
    rows, past = context.shape
    observed = torch.isfinite(context)
    target = torch.where(observed, context.to(torch.float64), 0.0)
    count = observed.sum(dim=-1)
    mean = torch.where(count > 0, target.sum(dim=-1) / count.clamp(min=1), 0.0)
    missing = torch.isnan(codes)
    categories = torch.unique(codes[~missing])
    found = torch.searchsorted(categories, codes.nan_to_num().contiguous())
    slots = torch.where(missing, categories.numel(), found)
    width = categories.numel() + 1
    offsets = torch.arange(rows, device=codes.device)[:, None] * width
    index = (offsets + slots[:, :past]).flatten()
    sums = torch.bincount(index, weights=target.flatten(), minlength=rows * width)
    counts = torch.bincount(
        index, weights=observed.to(torch.float64).flatten(), minlength=rows * width
    )
    lookup = (mean[:, None] + sums.view(rows, width)) / (1.0 + counts.view(rows, width))
    return lookup.gather(1, slots).to(torch.float32)


def _encode_covariates(
    rows: Tensor, indices: tuple[int, ...], history: Tensor, name: str, encode: bool
) -> Tensor:
    if len(set(indices)) != len(indices) or any(
        not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < rows.shape[0]
        for index in indices
    ):
        raise ValueError(f"{name} must contain unique valid covariate row indices")
    encoded = rows.to(torch.float32, copy=True)
    if indices and encode:
        index = [int(value) for value in indices]
        context = torch.nanmean(history.to(torch.float64), dim=0, keepdim=True)
        encoded[index] = _target_encode(context.expand(len(index), -1), rows[index])
    if bool(torch.isinf(encoded).any()):
        raise ValueError("numeric covariates exceed the float32 range")
    return encoded


def _difference_mask(
    rows: Tensor, config: Mapping[str, float], lengths: Tensor | None = None
) -> Tensor:
    """Detect strong linear trends and monotone running totals on observed history."""
    total = rows.shape[-1]
    window = (
        max(1, round(config["trend_window"] * total))
        if lengths is None
        else (lengths.double() * config["trend_window"]).round().clamp_min(1)
    )
    values = rows.to(torch.float64)
    index = torch.arange(total, dtype=torch.float64, device=rows.device).expand_as(values)
    start = total - window if lengths is None else total - window[:, None]
    observed = torch.isfinite(values) & (index >= start)
    v = torch.where(observed, values, torch.zeros_like(values))
    t = torch.where(observed, index, torch.zeros_like(index))
    n = observed.sum(dim=-1).to(torch.float64)
    n_safe = n.clamp(min=2.0)
    var_t = ((t * t).sum(dim=-1) - t.sum(dim=-1) ** 2 / n_safe).clamp(min=1e-12)
    var_v = (v * v).sum(dim=-1) - v.sum(dim=-1) ** 2 / n_safe
    cov_tv = (t * v).sum(dim=-1) - t.sum(dim=-1) * v.sum(dim=-1) / n_safe
    deviation = torch.sqrt((var_v / (n_safe - 1.0)).clamp(min=0.0))
    score = (cov_tv / var_t / deviation.clamp(min=1e-12)).abs() * window
    r_squared = cov_tv**2 / (var_t * var_v.clamp(min=1e-12))
    trending = (
        (score >= config["trend_threshold"])
        & (r_squared >= config["trend_min_r2"])
        & (n >= 4)
        & (deviation > 1e-12)
    )

    if not config["tta_running_totals"]:
        return trending
    observed = torch.isfinite(rows)
    index = torch.arange(total, device=rows.device).expand_as(rows)
    latest = torch.where(observed, index, torch.zeros_like(index)).cummax(dim=-1).values
    filled = rows.gather(-1, latest)
    steps = filled[..., 1:] - filled[..., :-1]
    finite = torch.isfinite(steps)
    rising = (~finite | (steps >= 0)).all(dim=-1)
    falling = (~finite | (steps <= 0)).all(dim=-1)
    counted = finite & observed[..., 1:]
    moving = (counted & (steps != 0)).sum(dim=-1)
    share = moving / counted.sum(dim=-1).clamp(min=1)
    running = (observed.sum(dim=-1) >= 4) & (moving > 0) & (share >= 0.5)
    return trending | (running & (rising | falling))


def _last_observed(history: Tensor) -> Tensor:
    observed = torch.isfinite(history)
    last_index = observed.long().cumsum(dim=-1).argmax(dim=-1)
    return torch.nan_to_num(history.gather(-1, last_index[:, None]).squeeze(-1), nan=0.0)


def _calibrate(
    pred: Tensor,
    history: Tensor,
    differenced: Tensor,
    config: Mapping[str, float],
    *,
    anchors: Tensor | None = None,
    has_trend: bool | None = None,
) -> Tensor:
    median = pred[:, pred.shape[1] // 2 : pred.shape[1] // 2 + 1]
    steps = torch.arange(1, pred.shape[-1] + 1, dtype=pred.dtype, device=pred.device)
    factor = config["raw_band_scale"] ** 0.5 * steps ** (config["raw_band_exponent"] - 0.5)
    calibrated = median + factor * (pred - median)
    if bool(differenced.any()) if has_trend is None else has_trend:
        selected = pred[differenced]
        middle = selected[:, selected.shape[1] // 2 : selected.shape[1] // 2 + 1]
        offset = selected - middle
        band = offset.sum(dim=-1, keepdim=True).sign() * torch.sqrt(
            torch.cumsum(offset * offset, dim=-1)
        )
        band = band * steps ** (config["diff_band_exponent"] - 0.5)
        if anchors is None:
            anchors = _last_observed(history)
        calibrated[differenced] = (
            torch.cumsum(middle, dim=-1)
            + config["diff_band_scale"] ** 0.5 * band
            + anchors[differenced, None, None]
        )
    return calibrated.sort(dim=1).values


def _forecast_once(
    model: nn.Module,
    context: Tensor,
    future: Tensor,
    n_targets: int,
    horizon: int,
    tta: Mapping[str, float],
    lengths: Tensor,
    *,
    sign_flip: bool = False,
) -> Tensor:
    """Forecast equal-width groups, optionally including their independent negative views."""
    cfg = model.config
    patch = cfg.output_patch_size
    window_len = max(cfg.future_len, -(-horizon // patch) * patch)
    samples, channels, width = context.shape
    history = context[:, :n_targets].flatten(0, 1)
    row_lengths = lengths.repeat_interleave(n_targets)
    differenced = (
        _difference_mask(history, tta, row_lengths)
        if tta["postprocess"] and tta["tta_diff"] and horizon <= tta["differencing_max_horizon"]
        else torch.zeros(len(history), dtype=torch.bool, device=history.device)
    )
    has_trend = bool(differenced.any())
    anchors = _last_observed(history) if has_trend else None
    if has_trend:
        diff = torch.cat([torch.zeros_like(history[:, :1]), history[:, 1:] - history[:, :-1]], -1)
        start = width - row_lengths
        valid = torch.arange(width, device=history.device) >= start[:, None]
        diff = torch.where(valid, diff, torch.nan).scatter(1, start[:, None], 0.0)
        targets = torch.where(differenced[:, None], diff, history).reshape(
            samples, n_targets, width
        )
        context = torch.cat([targets, context[:, n_targets:]], dim=1)
    n_known = future.shape[1]
    window = context.new_full((samples, channels, window_len), torch.nan)
    window[:, n_targets : n_targets + n_known, :horizon] = future
    observed = torch.isfinite(context).flatten(0, 1).any(0)
    span = width - int(observed.int().argmax()) if bool(observed.any()) else 1
    context_len = max(patch, 1 << (span - 1).bit_length())
    if cfg.context_len is not None:
        context_len = min(context_len, cfg.context_len)
    x = torch.cat([context[..., -context_len:], window], dim=-1)
    x = pad(x, (context_len + window_len - x.shape[-1], 0), value=torch.nan)
    if sign_flip:
        x = torch.cat([x, -x])
        differenced = differenced.repeat(2)
        if anchors is not None:
            anchors = torch.cat([anchors, -anchors])
    layout = forecast_layout(len(x), channels, n_targets, n_known, x.device)
    batch = {
        "x": x.flatten(0, 1),
        "group_vector": layout.groups,
        "target_mask": layout.targets,
        "future_covariate_mask": layout.future,
        "past_covariate_mask": layout.past,
    }
    with torch.no_grad():
        pred = model(batch, future_len=window_len, horizon=horizon, layout=layout)
    pred = pred.reshape(len(x), channels, -1, horizon)[:, :n_targets].flatten(0, 1)
    if tta["postprocess"]:
        pred = _calibrate(pred, history, differenced, tta, anchors=anchors, has_trend=has_trend)
    return pred.reshape(len(x), n_targets, -1, horizon)


def _context(
    history: Tensor,
    horizon: int,
    past_covariates: Tensor | None,
    future_covariates: Tensor | None,
    categorical_past: tuple[int, ...],
    categorical_future: tuple[int, ...],
    encode_categoricals: bool,
) -> tuple[Tensor, Tensor]:
    device = history.device
    n_targets, length = history.shape
    parts = [history]
    future = history.new_empty((0, horizon))
    if future_covariates is not None:
        rows = _rows(
            future_covariates, "future_covariates", device, dtype=None, length=length + horizon
        )
        if rows.shape[1] < length + horizon:
            raise ValueError("future_covariates must cover history plus horizon")
        rows = _encode_covariates(
            rows, categorical_future, history, "categorical_future", encode_categoricals
        )
        parts.append(rows[:, :length])
        future = rows[:, length:]
    elif categorical_future:
        raise ValueError("categorical_future requires future_covariates")
    if past_covariates is not None:
        rows = _rows(past_covariates, "past_covariates", device, dtype=None)
        if rows.shape[1] != length:
            raise ValueError("past_covariates must have the same history length")
        parts.append(
            _encode_covariates(
                rows, categorical_past, history, "categorical_past", encode_categoricals
            )
        )
    elif categorical_past:
        raise ValueError("categorical_past requires past_covariates")
    return torch.cat(parts, dim=0), future


def _predict(
    model: nn.Module,
    context: Tensor,
    future: Tensor,
    n_targets: int,
    horizon: int,
    tta: Mapping[str, float],
    lengths: Tensor | None = None,
) -> Tensor:
    """Forecast ``[samples, channels, time]`` groups with shared sign preprocessing."""
    device = context.device
    samples = len(context)
    history = context[:, :n_targets].flatten(0, 1)
    if lengths is None:
        lengths = torch.full((samples,), context.shape[-1], dtype=torch.long, device=device)
    sign_flip = bool(tta["postprocess"] and tta["tta_sign_flip"])
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(tta["bfloat16"])):
        pred = _forecast_once(
            model, context, future, n_targets, horizon, tta, lengths, sign_flip=sign_flip
        )
        if sign_flip:
            positive, negative = pred.chunk(2)
            # Loading validates a strictly sorted, symmetric grid centred on the median.
            pred = (positive - negative.flip(2)) / 2
    pred = pred.float().flatten(0, 1)
    observed = torch.isfinite(history)
    if tta["clip_nonnegative"]:
        nonnegative = observed.any(dim=-1) & ~(observed & (history < 0)).any(dim=-1)
        pred = torch.where(nonnegative[:, None, None], pred.clamp_min(0), pred)
    if tta["integer_round"]:
        integers = observed.any(dim=-1) & ~(observed & (history != history.round())).any(dim=-1)
        pred = torch.where(integers[:, None, None], pred.round(), pred)
    expected = (samples * n_targets, model.get_buffer("quantiles").numel(), horizon)
    if pred.shape != expected or not bool(torch.isfinite(pred).all()):
        raise ValueError("model returned invalid forecast values or shape")
    return pred.reshape(samples, n_targets, -1, horizon).detach().cpu()


def _timestamps(
    timestamps: Tensor | None, freq: str | None, count: int, length: int, horizon: int
) -> Tensor | None:
    if timestamps is None:
        if freq is not None:
            raise ValueError("freq requires timestamps")
        return None
    if not isinstance(timestamps, Tensor) or timestamps.dtype != torch.int64:
        raise ValueError("timestamps must be an int64 tensor of epoch nanoseconds")
    stamps = timestamps.detach().cpu()
    if stamps.ndim == 1:
        stamps = stamps[None].expand(count, -1)
    if stamps.ndim != 2 or stamps.shape[0] != count or stamps.shape[1] < length + horizon:
        raise ValueError("timestamps must cover history plus horizon for every series")
    stamps = stamps[:, : length + horizon]
    valid = stamps != torch.iinfo(torch.int64).min
    if not valid[:, length - 1 :].all():
        raise ValueError("timestamps must include the forecast origin and every future step")
    start = valid.long().argmax(dim=1)
    if not torch.equal(valid, torch.arange(stamps.shape[1])[None] >= start[:, None]):
        raise ValueError("missing timestamps are only allowed in leading history padding")
    if ((stamps[:, 1:] <= stamps[:, :-1]) & valid[:, :-1]).any():
        raise ValueError("timestamps must be strictly increasing")
    if freq is not None and not isinstance(freq, str):
        raise ValueError("freq must be a frequency string")
    return stamps


def _with_calendar(
    context: Tensor,
    future: Tensor,
    n_targets: int,
    stamps: Tensor | None,
    freq: str | None,
    enabled: bool,
    *,
    cache: dict[bytes, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    if not enabled or stamps is None:
        return context, future
    if freq is None:
        raise ValueError("freq is required to generate calendar covariates")
    from ._calendar import calendar_channels, frequency_step_ns

    key = stamps.numpy().tobytes()
    calendar = None if cache is None else cache.get(key)
    if calendar is None:
        calendar = torch.from_numpy(calendar_channels(stamps.numpy(), frequency_step_ns(freq)))
        calendar = calendar.to(context)
        if cache is not None:
            cache[key] = calendar
    if not calendar.shape[0]:
        return context, future
    length = context.shape[-1]
    at = n_targets + future.shape[0]
    context = torch.cat([context[:at], calendar[:, :length], context[at:]], dim=0)
    return context, torch.cat([future, calendar[:, length:]], dim=0)


@dataclass
class _Case:
    """One independent forecast window and the caller's output row indices."""

    indices: list[int]
    context: Tensor
    future: Tensor
    n_targets: int


def _cohort_covariates(
    value: Tensor | None,
    name: str,
    count: int,
    length: int,
    horizon: int,
    device: torch.device,
    categorical: tuple[int, ...],
) -> Tensor | None:
    if value is None:
        if categorical:
            raise ValueError(f"categorical indices require {name}")
        return None
    if not isinstance(value, Tensor) or value.ndim not in (2, 3):
        raise ValueError(f"{name} must be a two- or three-dimensional tensor")
    rows = value[None].expand(count, -1, -1) if value.ndim == 2 else value
    if rows.shape[0] != count or rows.shape[1] < 1:
        raise ValueError(f"{name} must have covariates for every series")
    width = length + horizon
    if (horizon and rows.shape[-1] < width) or (not horizon and rows.shape[-1] != width):
        raise ValueError(f"{name} must cover the required time steps")
    rows = rows[..., :width]
    flat = _rows(rows.reshape(-1, width), name, device, dtype=None)
    # Validate indices without fitting an encoding across unrelated series.
    if len(set(categorical)) != len(categorical) or any(
        not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < rows.shape[1]
        for index in categorical
    ):
        raise ValueError(f"categorical_{name} must contain unique valid covariate row indices")
    return flat.reshape(count, rows.shape[1], width)


def _cohort_cases(
    history: Tensor,
    horizon: int,
    tta: Mapping[str, float],
    past: Tensor | None,
    future: Tensor | None,
    categorical_past: tuple[int, ...],
    categorical_future: tuple[int, ...],
    stamps: Tensor | None,
    freq: str | None,
    series_ids: Sequence[str],
    static_features: Mapping[str, Sequence[object]] | None,
) -> Iterable[_Case]:
    from ._groups import chunk_groups, joint_neighbour_rows, static_group_rows

    count, length = history.shape
    if (
        isinstance(series_ids, str)
        or len(series_ids) != count
        or any(not isinstance(name, str) or not name for name in series_ids)
        or len(set(series_ids)) != count
    ):
        raise ValueError("series_ids must contain one unique string per history row")
    if static_features is not None and not isinstance(static_features, Mapping):
        raise ValueError("static_features must map column names to per-series values")
    statics = dict(static_features or {})
    for name, values in statics.items():
        if (
            not isinstance(name, str)
            or isinstance(values, (str, bytes, Mapping))
            or not hasattr(values, "__len__")
            or len(values) != count
        ):
            raise ValueError("static_features must have one value per series in each column")
        statics[name] = (
            values.detach().cpu().tolist() if isinstance(values, Tensor) else tuple(values)
        )
    if stamps is None:
        stamps = torch.arange(length + horizon)[None].expand(count, -1)
        calendar_stamps = None
        starts = [0] * count
    else:
        calendar_stamps = stamps
        starts = (stamps != torch.iinfo(torch.int64).min).long().argmax(dim=1).tolist()
    order = sorted(range(count), key=lambda i: series_ids[i])
    stamps_np = stamps.numpy()
    bare = not statics and past is None and future is None
    chunk = int(tta["chunk_joint"]) if bare else 0
    static_rows = None
    neighbours = None
    if (statics and tta["static_groups"]) or (
        not statics and not chunk and tta["joint_neighbours"]
    ):
        histories = history.detach().double().cpu().numpy()
        histories_in_order = [histories[i, starts[i] :] for i in order]
        times = [stamps_np[i, starts[i] : length] for i in order]
    if statics and tta["static_groups"]:
        columns = [[statics[name][i] for i in order] for name in sorted(statics)]
        static_rows = static_group_rows(
            histories_in_order, times, columns, int(tta["static_groups"])
        )
    elif not statics and not chunk and tta["joint_neighbours"]:
        neighbours = joint_neighbour_rows(histories_in_order, times, int(tta["joint_neighbours"]))
    groups = (
        chunk_groups([int(stamps_np[i, length - 1]) for i in order], chunk)
        if chunk
        else [[i] for i in range(count)]
    )
    calendar_cache: dict[bytes, Tensor] = {}
    for members in groups:
        indices = [order[position] for position in members]
        longest = min(indices, key=lambda i: starts[i])
        start = starts[longest]
        targets = history[indices, start:]
        own_past = None if past is None else past[longest, :, start:]
        own_future = None if future is None else future[longest, :, start:]
        if static_rows is not None:
            extra = torch.from_numpy(static_rows[members[0]]).to(history)
            own_past = extra if own_past is None else torch.cat([own_past, extra])
        if neighbours is not None:
            targets = torch.from_numpy(neighbours[members[0]]).to(history)
        context, revealed = _context(
            targets,
            horizon,
            own_past,
            own_future,
            categorical_past,
            categorical_future,
            bool(tta["target_encode_categoricals"]),
        )
        own_stamps = None if calendar_stamps is None else calendar_stamps[longest, start:]
        context, revealed = _with_calendar(
            context,
            revealed,
            len(targets),
            own_stamps,
            freq,
            bool(tta["calendar_covariates"]),
            cache=calendar_cache,
        )
        yield _Case(indices, context, revealed, len(targets))


def _packed_predict(
    model: nn.Module, cases: Sequence[_Case], horizon: int, tta: Mapping[str, float]
) -> Tensor:
    """Stack equal-width groups without changing group membership or covariate ownership."""
    width = max(case.context.shape[-1] for case in cases)
    context = torch.stack(
        [pad(case.context, (width - case.context.shape[-1], 0), value=torch.nan) for case in cases]
    )
    lengths = torch.tensor([case.context.shape[-1] for case in cases], device=context.device)
    return _predict(
        model,
        context,
        torch.stack([case.future for case in cases]),
        cases[0].n_targets,
        horizon,
        tta,
        lengths,
    ).flatten(0, 1)


def _run_cases(
    model: nn.Module,
    cases: Iterable[_Case],
    count: int,
    horizon: int,
    tta: Mapping[str, float],
) -> Tensor:
    out = torch.empty(count, model.get_buffer("quantiles").numel(), horizon)
    pending: dict[tuple[int, int, int], list[_Case]] = {}
    rows = 0
    views = 2 if tta["postprocess"] and tta["tta_sign_flip"] else 1

    def flush() -> None:
        for members in pending.values():
            predictions = _packed_predict(model, members, horizon, tta)
            cursor = 0
            for case in members:
                out[case.indices] = predictions[cursor : cursor + len(case.indices)]
                cursor += case.n_targets
        pending.clear()

    for case in cases:
        if pending and (rows + len(case.context)) * views > 2048:
            flush()
            rows = 0
        key = (case.n_targets, len(case.future), len(case.context))
        pending.setdefault(key, []).append(case)
        rows += len(case.context)
    if pending:
        flush()
    return out
