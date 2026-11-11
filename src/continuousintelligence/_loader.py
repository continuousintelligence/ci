"""Local and Hugging Face model loading for Continuous Intelligence."""

import json
import math
import re
from contextlib import nullcontext
from functools import cache
from pathlib import Path, PureWindowsPath

import torch

from .ci_proto import Forecaster

_COMMIT = re.compile(r"[0-9a-f]{40}")


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def _files(
    source: str | Path, tta_config: str | Path | None, revision: str | None
) -> tuple[Path, Path, Path | None]:
    path = Path(source).expanduser()
    if path.exists():
        if revision is not None:
            raise ValueError("revision applies only to Hugging Face models")
    else:
        if (
            isinstance(source, Path)
            or str(source).startswith(("/", "./", "../", "~", ".\\", "..\\"))
            or "\\" in str(source)
            or PureWindowsPath(source).drive
            or str(source).count("/") != 1
            or path.parent.is_dir()
            or path.suffix in (".ckpt", ".pt", ".pth")
            or tta_config is not None
        ):
            raise FileNotFoundError(path)
        from huggingface_hub import hf_hub_download, snapshot_download

        info = Path(hf_hub_download(repo_id=str(source), filename="config.json", revision=revision))
        commit = info.parent.name if _COMMIT.fullmatch(info.parent.name) else revision
        path = Path(
            snapshot_download(
                repo_id=str(source),
                revision=commit,
                allow_patterns=["model.ckpt", "model-long.ckpt", "tta.json"],
            )
        )
    info = None
    if path.is_dir():
        if tta_config is not None:
            raise ValueError("a model bundle uses its own tta.json")
        info = path / "config.json" if (path / "config.json").is_file() else None
        path, tta_config = path / "model.ckpt", path / "tta.json"
    elif tta_config is None:
        raise ValueError("a standalone checkpoint requires tta_config")
    tta = Path(tta_config).expanduser()
    for file in (path, tta):
        if not file.is_file():
            raise FileNotFoundError(file)
    return path, tta, info


def _model_info(path: Path, forecaster: Forecaster) -> dict:
    """Read a bundle's ``config.json`` and check it describes the loaded weights."""
    info = json.loads(path.read_text())
    if not isinstance(info, dict):
        raise ValueError("config.json must contain a JSON object")
    missing = {"name", "parameters", "quantiles"} - set(info)
    if missing:
        raise ValueError(f"config.json is missing {sorted(missing)}")
    model = forecaster._model
    quantiles = forecaster.quantiles.tolist()
    if (
        not isinstance(info["quantiles"], list)
        or len(info["quantiles"]) != len(quantiles)
        or not all(
            isinstance(given, (int, float)) and math.isclose(given, level, abs_tol=1e-6)
            for given, level in zip(info["quantiles"], quantiles, strict=True)
        )
    ):
        raise ValueError("config.json quantiles do not match the checkpoint")
    if info["parameters"] != sum(parameter.numel() for parameter in model.parameters()):
        raise ValueError("config.json parameters do not match the checkpoint")
    return info


def load_model(
    checkpoint: str | Path,
    tta_config: str | Path | None = None,
    device: str = "auto",
    *,
    revision: str | None = None,
) -> Forecaster:
    """Load a local model bundle, Hugging Face model ID, or checkpoint plus TTA file.

    Bundles contain ``model.ckpt`` (weights and embedded architecture) and
    ``tta.json``. Hub bundles also contain ``config.json``, which describes the
    model and is returned by ``Forecaster.info``; it is checked against the
    weights. Hub files are fetched together from the same revision;
    pass a commit SHA to pin it. Hub IDs use ``owner/model`` strings. Use a
    ``Path`` or ``./`` prefix to mark an ambiguous relative path as local.
    An optional ``model-long.ckpt`` shares the bundle's TTA and is selected
    when history exceeds the primary checkpoint's context length.
    """
    placement = _device(device)
    path, tta_path, info_path = _files(checkpoint, tta_config, revision)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    from .ci_proto._checkpoint import from_checkpoint

    forecaster = from_checkpoint(payload, tta_path, placement)
    placement = forecaster.device
    if info_path is not None:
        forecaster._info = _model_info(info_path, forecaster)
    long_path = path.with_name("model-long.ckpt")
    if tta_config is None and long_path.is_file() and forecaster._model.config.context_len:
        quantiles = forecaster.quantiles
        future_len = forecaster._model.config.future_len

        @cache
        def load_long() -> Forecaster:
            context = torch.cuda.device(placement) if placement.type == "cuda" else nullcontext()
            with context:
                long = load_model(long_path, tta_path, device=placement.type)
            if not torch.equal(long.quantiles, quantiles) or (
                long._model.config.future_len != future_len
            ):
                raise ValueError("long checkpoint must have the same quantiles and forecast window")
            return long

        forecaster._long_model = load_long
    return forecaster
