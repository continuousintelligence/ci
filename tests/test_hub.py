"""Resolve paired model files without making network requests."""

import json
from pathlib import Path
from shutil import copyfile
from unittest.mock import Mock
from weakref import ref

import huggingface_hub
import pytest
import torch

from continuousintelligence import load_model


@pytest.fixture
def info(release_files: tuple[Path, Path]) -> dict:
    """A config.json describing the synthetic checkpoint."""
    model = load_model(*release_files, device="cpu")
    return {
        "name": "test-model",
        "parameters": sum(parameter.numel() for parameter in model._model.parameters()),
        "quantiles": [0.1, 0.5, 0.9],
        "licence": "test-licence",
    }


@pytest.fixture
def bundle(release_files: tuple[Path, Path], info: dict, tmp_path: Path) -> Path:
    """A synthetic model bundle has the same layout locally and in a Hub snapshot."""
    folder = tmp_path / "bundle"
    folder.mkdir()
    for source, filename in zip(release_files, ("model.ckpt", "tta.json"), strict=True):
        copyfile(source, folder / filename)
    (folder / "config.json").write_text(json.dumps(info))
    return folder


@pytest.fixture
def config_download(monkeypatch: pytest.MonkeyPatch, bundle: Path) -> Mock:
    """Replace the counted config.json request with the local synthetic snapshot."""
    mocked = Mock(side_effect=lambda repo_id, filename, revision=None: str(bundle / filename))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", mocked)
    return mocked


@pytest.fixture
def download(monkeypatch: pytest.MonkeyPatch, bundle: Path, config_download: Mock) -> Mock:
    """Replace the network boundary with a local synthetic snapshot."""
    mocked = Mock(return_value=str(bundle))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", mocked)
    return mocked


@pytest.mark.parametrize("revision", [None, "v1", "a" * 40])
def test_hub_resolves_both_files_in_one_snapshot(
    release_files: tuple[Path, Path], download: Mock, revision: str | None
) -> None:
    """One filtered snapshot supplies matched weights and calibration to the local loader."""
    hub = load_model("test-owner/test-model", device="cpu", revision=revision)
    local = load_model(*release_files, device="cpu")
    download.assert_called_once()
    assert download.call_args.kwargs["repo_id"] == "test-owner/test-model"
    assert download.call_args.kwargs.get("revision") == revision
    assert set(download.call_args.kwargs["allow_patterns"]) == {
        "model.ckpt",
        "model-long.ckpt",
        "tta.json",
    }
    assert "local_dir" not in download.call_args.kwargs
    history = torch.tensor([[0.2, 1.3, -0.7, 0.5]])
    torch.testing.assert_close(hub.forecast(history, 3), local.forecast(history, 3), rtol=0, atol=0)


@pytest.mark.parametrize("revision", [None, "v1"])
def test_every_hub_load_requests_config(
    download: Mock, config_download: Mock, revision: str | None
) -> None:
    """Each Hub load makes one config.json request, which the Hub counts as a download."""
    for loads in (1, 2):
        load_model("test-owner/test-model", device="cpu", revision=revision)
        assert config_download.call_count == loads
        assert config_download.call_args.kwargs == {
            "repo_id": "test-owner/test-model",
            "filename": "config.json",
            "revision": revision,
        }


def test_hub_files_come_from_the_config_commit(
    bundle: Path, download: Mock, config_download: Mock, tmp_path: Path
) -> None:
    """The weights and calibration are fetched at the commit the config.json request resolved."""
    commit = "c" * 40
    snapshot = tmp_path / "snapshots" / commit
    snapshot.mkdir(parents=True)
    copyfile(bundle / "config.json", snapshot / "config.json")
    config_download.side_effect = lambda repo_id, filename, revision=None: str(snapshot / filename)
    load_model("test-owner/test-model", device="cpu", revision="main")
    assert download.call_args.kwargs["revision"] == commit


def test_missing_hub_config_fails_before_downloading_weights(
    download: Mock, config_download: Mock
) -> None:
    """A Hub bundle without config.json is rejected before its weights are fetched."""
    config_download.side_effect = huggingface_hub.errors.EntryNotFoundError("config.json")
    with pytest.raises(huggingface_hub.errors.EntryNotFoundError):
        load_model("test-owner/test-model", device="cpu")
    download.assert_not_called()


@pytest.mark.parametrize("remote", [False, True])
def test_model_info_describes_the_bundle(
    bundle: Path, download: Mock, info: dict, remote: bool
) -> None:
    """A bundle's config.json is exposed as model.info, and edits to it do not reach the model."""
    model = load_model("test-owner/test-model" if remote else bundle, device="cpu")
    assert model.info == info
    model.info["name"] = "changed"
    assert model.info == info


def test_model_info_is_absent_without_config(
    bundle: Path, release_files: tuple[Path, Path]
) -> None:
    """Local bundles may omit config.json, and standalone checkpoints have none."""
    (bundle / "config.json").unlink()
    assert load_model(bundle, device="cpu").info is None
    assert load_model(*release_files, device="cpu").info is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantiles", [0.1, 0.5, 0.8]),
        ("quantiles", [0.1, 0.9]),
        ("quantiles", "0.1, 0.5, 0.9"),
        ("parameters", 1),
    ],
)
def test_model_info_must_match_the_weights(
    bundle: Path, info: dict, field: str, value: object
) -> None:
    """A config.json that disagrees with the checkpoint is rejected."""
    (bundle / "config.json").write_text(json.dumps({**info, field: value}))
    with pytest.raises(ValueError, match=field):
        load_model(bundle, device="cpu")


@pytest.mark.parametrize("content", [[], {"name": "test-model"}])
def test_model_info_requires_an_object_with_its_fields(bundle: Path, content: object) -> None:
    """config.json must be an object giving the name, size and quantiles."""
    (bundle / "config.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="config.json"):
        load_model(bundle, device="cpu")


@pytest.mark.parametrize("as_string", [False, True])
def test_local_bundle_never_downloads(bundle: Path, download: Mock, as_string: bool) -> None:
    """Existing directories resolve their paired model files locally."""
    model = load_model(str(bundle) if as_string else bundle, device="cpu")
    assert model.quantiles.shape == (3,)
    download.assert_not_called()


@pytest.mark.parametrize("filename", ["model.ckpt", "tta.json"])
@pytest.mark.parametrize("remote", [False, True])
def test_incomplete_bundle_fails(bundle: Path, download: Mock, filename: str, remote: bool) -> None:
    """A filtered snapshot is not accepted when either required file is absent."""
    (bundle / filename).unlink()
    source = "test-owner/test-model" if remote else bundle
    with pytest.raises(FileNotFoundError):
        load_model(source, device="cpu")
    assert download.call_count == int(remote)


@pytest.mark.parametrize(
    "source",
    [
        Path("missing/model"),
        "./missing/model",
        "../missing/model",
        "/missing/model",
        "missing.ckpt",
        "missing.pt",
        "missing.pth",
        "missing-model",
        "missing/nested/model",
        r"C:\models\missing",
        "C:/models/missing",
        r"C:missing",
        r"models\missing",
        r"\\server\models\missing",
    ],
)
def test_missing_explicit_local_path_never_downloads(source: str | Path, download: Mock) -> None:
    """Missing paths and checkpoint filenames are not reinterpreted as Hub identifiers."""
    with pytest.raises(FileNotFoundError):
        load_model(source, device="cpu")
    download.assert_not_called()


def test_missing_child_of_local_directory_never_downloads(
    tmp_path: Path, download: Mock, monkeypatch
):
    """An existing parent identifies a missing bare relative path as local."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    with pytest.raises(FileNotFoundError):
        load_model("models/missing", device="cpu")
    download.assert_not_called()


@pytest.mark.parametrize("remote", [False, True])
def test_bundle_cannot_override_its_calibration(
    bundle: Path, release_files: tuple[Path, Path], download: Mock, remote: bool
) -> None:
    """Bundle weights cannot be paired with an unrelated local TTA file."""
    source = "test-owner/test-model" if remote else bundle
    with pytest.raises((ValueError, FileNotFoundError)):
        load_model(source, tta_config=release_files[1], device="cpu")
    download.assert_not_called()


@pytest.mark.parametrize("directory", [False, True])
def test_local_loading_rejects_revision(
    bundle: Path, release_files: tuple[Path, Path], download: Mock, directory: bool
) -> None:
    """A Hub revision cannot be silently ignored for local files."""
    arguments = (bundle,) if directory else release_files
    with pytest.raises(ValueError, match="revision"):
        load_model(*arguments, device="cpu", revision="v1")
    download.assert_not_called()


def test_local_file_requires_calibration(release_files: tuple[Path, Path], download: Mock) -> None:
    """A standalone checkpoint needs its explicit TTA file."""
    with pytest.raises(ValueError):
        load_model(release_files[0], device="cpu")
    download.assert_not_called()


def test_failed_download_propagates(download: Mock) -> None:
    """Download failures remain visible instead of falling back to unrelated local files."""
    download.side_effect = OSError("synthetic download failure")
    with pytest.raises(OSError, match="synthetic download failure"):
        load_model("test-owner/test-model", device="cpu")


@pytest.fixture
def long_bundle(bundle: Path) -> Path:
    """Add distinct long-model weights with an unrestricted context to the same bundle."""
    from continuousintelligence.ci_proto._model import Model

    payload = torch.load(bundle / "model.ckpt", weights_only=True)
    config = payload["hyper_parameters"]["model"]
    config["context_len"] = None
    torch.manual_seed(29)
    payload["state_dict"] = Model(config).state_dict()
    torch.save(payload, bundle / "model-long.ckpt")
    return bundle


@pytest.mark.parametrize("remote", [False, True])
def test_long_model_selection_and_reuse(long_bundle: Path, download: Mock, monkeypatch, remote):
    """Boundary histories use the standard weights; longer inputs load and reuse the long weights."""
    source = "test-owner/test-model" if remote else long_bundle
    standard = load_model(long_bundle / "model.ckpt", long_bundle / "tta.json", device="cpu")
    long = load_model(long_bundle / "model-long.ckpt", long_bundle / "tta.json", device="cpu")
    loads = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", loads)
    auto = load_model(source, device="cpu")
    assert loads.call_count == 1
    for length, expected_loads in ((15, 1), (16, 1), (17, 2), (33, 2), (16, 2)):
        history = torch.sin(torch.arange(length) * 0.7).repeat(2, 1)
        expected = (long if length > 16 else standard).forecast(history, 3)
        torch.testing.assert_close(auto.forecast(history, 3), expected, rtol=0, atol=0)
        assert loads.call_count == expected_loads
    assert loads.call_count == 2
    assert download.call_count == int(remote)
    reference = ref(auto)
    del auto
    assert reference() is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
@pytest.mark.parametrize("device_index", [0, 1])
def test_long_model_keeps_original_cuda_device(long_bundle: Path, device_index: int) -> None:
    """Lazy loading retains the model's GPU and restores the caller's current device."""
    with torch.cuda.device(device_index):
        model = load_model(long_bundle, device="cuda")
    caller_device = 1 - device_index
    with torch.cuda.device(caller_device):
        prediction = model.forecast(torch.ones(1, 17), 3)
        assert model._long_model().device == model.device == torch.device("cuda", device_index)
        assert torch.cuda.current_device() == caller_device
    assert prediction.isfinite().all()


def test_long_model_preserves_full_history_and_metadata(long_bundle: Path):
    """The routed forecast keeps early observations, covariates and cohort metadata."""
    auto = load_model(long_bundle, device="cpu")
    long = load_model(long_bundle / "model-long.ckpt", long_bundle / "tta.json", device="cpu")
    history = torch.sin(torch.arange(37) * 0.7).repeat(2, 1)
    arguments = {
        "past_covariates": torch.cos(torch.arange(37) * 0.5)[None],
        "timestamps": 1_735_689_600_000_000_000 + torch.arange(40) * 3_600_000_000_000,
        "freq": "h",
        "series_ids": ["b", "a"],
        "static_features": {"region": ["north", "north"]},
    }
    seen = []
    hook = long._model.register_forward_pre_hook(lambda _model, args: seen.append(args[0]["x"]))
    try:
        expected = long.forecast(history, 3, **arguments)
    finally:
        hook.remove()
    assert seen[0].shape[-1] == 64 + long._model.config.future_len
    torch.testing.assert_close(seen[0][[0, 6], 27:64], history, rtol=0, atol=0)
    torch.testing.assert_close(auto.forecast(history, 3, **arguments), expected, rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [("future_len", 4), ("quantiles", [0.2, 0.5, 0.8])])
def test_incompatible_long_model_fails(long_bundle: Path, field, value):
    """Automatic switching cannot silently change the output horizon or quantile levels."""
    payload = torch.load(long_bundle / "model-long.ckpt", weights_only=True)
    payload["hyper_parameters"]["model"][field] = value
    torch.save(payload, long_bundle / "model-long.ckpt")
    auto = load_model(long_bundle, device="cpu")
    with pytest.raises(ValueError, match="same quantiles and forecast window"):
        auto.forecast(torch.ones(1, 17), 3)
