"""Post-build checks: install the wheel, then run ``pytest tests/distribution.py``."""

import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSE_EXPRESSION = "LicenseRef-ci-proto AND Apache-2.0 AND CC-BY-4.0"
LICENSE_FILES = {
    "LICENCE",
    "NOTICE",
    "src/continuousintelligence/_vendor/mamba3/LICENSE",
    "src/continuousintelligence/_data/NOTICE",
}


def _check_file(name: str, size: int) -> None:
    path = Path(name)
    assert path.suffix not in {".ckpt", ".pt", ".pth", ".safetensors", ".bin"}, name
    assert not (
        path.name.startswith(("tta", "model-config")) and path.suffix in {".json", ".yaml", ".yml"}
    ), name
    assert size < 5_000_000, name


def test_repository_contents() -> None:
    """Tracked files exclude model weights, local configurations and oversized artifacts."""
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, names):
        _check_file(name, (ROOT / name).stat().st_size)


def test_distribution_contents() -> None:
    """Wheel and source archives exist and obey the repository's file restrictions."""
    wheels = list((ROOT / "dist").glob("*.whl"))
    sources = list((ROOT / "dist").glob("*.tar.gz"))
    assert wheels and sources, "run uv build before the distribution checks"
    for path in wheels:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                _check_file(entry.filename, entry.file_size)
            metadata_path = next(name for name in archive.namelist() if name.endswith("/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
            assert metadata["License-Expression"] == LICENSE_EXPRESSION
            assert set(metadata.get_all("License-File")) == LICENSE_FILES
            folder = str(Path(metadata_path).parent / "licenses")
            for name in metadata.get_all("License-File"):
                assert archive.read(f"{folder}/{name}") == (ROOT / name).read_bytes()
            for name in ("bike_sharing.csv", "NOTICE"):
                assert (
                    archive.read(f"continuousintelligence/_data/{name}")
                    == (ROOT / "src/continuousintelligence/_data" / name).read_bytes()
                )
    for path in sources:
        with tarfile.open(path) as archive:
            for entry in archive.getmembers():
                _check_file(entry.name, entry.size)
            folder = archive.getnames()[0].split("/")[0]
            for name in LICENSE_FILES:
                with archive.extractfile(f"{folder}/{name}") as handle:
                    assert handle.read() == (ROOT / name).read_bytes()
            for name in ("bike_sharing.csv", "NOTICE"):
                source = f"src/continuousintelligence/_data/{name}"
                with archive.extractfile(f"{folder}/{source}") as handle:
                    assert handle.read() == (ROOT / source).read_bytes()


def test_installed_wheel_outside_checkout(tmp_path: Path) -> None:
    """An isolated process imports the installed package outside the checkout."""
    program = """
from pathlib import Path
from importlib.metadata import metadata
from importlib.util import find_spec
from packaging.requirements import Requirement
import sysconfig
import continuousintelligence
from continuousintelligence import ci_proto

assert Path(continuousintelligence.__file__).resolve().is_relative_to(Path(sysconfig.get_path('purelib')).resolve())
assert Path(ci_proto.__file__).resolve().is_relative_to(Path(sysconfig.get_path('purelib')).resolve())
assert callable(continuousintelligence.load_model)
assert callable(continuousintelligence.plot_forecast)
history = continuousintelligence.load_example()
assert history.shape == (1, 120) and history.isfinite().all()
assert continuousintelligence.Forecaster is ci_proto.Forecaster
assert find_spec("ci_proto") is None
assert metadata("continuousintelligence")["Name"] == "continuousintelligence"
requirements = [Requirement(value) for value in metadata("continuousintelligence").get_all("Requires-Dist")]
assert any(req.name == "matplotlib" and req.marker is None for req in requirements)
assert metadata("continuousintelligence")["License-Expression"] == "LicenseRef-ci-proto AND Apache-2.0 AND CC-BY-4.0"
"""
    subprocess.run([sys.executable, "-I", "-c", program], cwd=tmp_path, check=True)
