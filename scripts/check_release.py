"""Require a GitHub version tag matching the package installed by uv sync."""

import os
from importlib.metadata import version

expected_tag = f"v{version('continuousintelligence')}"

if os.environ.get("GITHUB_REF_TYPE") != "tag":
    raise SystemExit("Select a version tag to publish.")
if os.environ.get("GITHUB_REF_NAME") != expected_tag:
    raise SystemExit(f"Release tag must be {expected_tag}, matching the package version.")
