"""Data from ECDC's RespiCast syndromic-indicators hub, pinned to the commit behind our results.

The hub holds each week's surveillance release, every team's submitted forecasts and the latest
data used to score them. ``checkout`` fetches only the files these scripts read (about 400 MB).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

REPO = "https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators.git"
COMMIT = "5acf646ced030353dea6dc5ab28221e5cc9fab2d"  # 21 November 2026, after the 18 Nov round
FOLDERS = ("/model-output/", "/target-data/")
OUTPUT = Path(__file__).resolve().parents[2] / "outputs" / "respicast"

TARGETS = {"ILI incidence": "ILI_incidence", "ARI incidence": "ARI_incidence"}
SOURCES = ("ERVISS", "FluID")
HORIZONS = (1, 2, 3, 4)  # weeks after the round's latest data week
FREQ = "W-SUN"  # ISO weeks, dated by their Sunday
ENSEMBLE = "respicast-hubEnsemble"
BASELINE = "respicast-quantileBaseline"
CI = "ci-proto"
TASK = ["origin_date", "target", "location", "horizon"]
LEVELS = [round(k / 10, 1) for k in range(1, 10)]  # ci-proto's quantile levels
QUANTILES = [f"q{level:g}" for level in LEVELS]
SEASONS = {"2025-26": ("2025-08-01", "2026-07-31"), "2026-27": ("2026-08-01", "2027-07-31")}


def checkout(path: Path) -> Path:
    """The hub at ``COMMIT`` in ``path``, fetched on first use."""
    git = ["git", "-C", str(path)]
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "sparse-checkout", "set", "--no-cone", *FOLDERS], check=True)
        fetch = ["fetch", "-q", "--depth", "1", "--filter=blob:none", REPO, COMMIT]
        subprocess.run([*git, *fetch], check=True)
        subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if head.stdout.strip() != COMMIT:
        raise RuntimeError(f"{path} is at {head.stdout.strip()}, not {COMMIT}")
    return path


def season(origin: pd.Timestamp) -> str:
    """The season (August to July) containing a round."""
    for name, (first, last) in SEASONS.items():
        if pd.Timestamp(first) <= origin <= pd.Timestamp(last):
            return name
    raise ValueError(f"{origin.date()} is outside {list(SEASONS)}")


def rounds(hub: Path) -> pd.DatetimeIndex:
    """Deadlines (Wednesdays) of the rounds since August 2025 with a hub ensemble."""
    files = (hub / "model-output" / ENSEMBLE).glob("*.csv")
    dates = pd.DatetimeIndex(sorted(pd.Timestamp(file.name[:10]) for file in files))
    return dates[dates >= pd.Timestamp(SEASONS["2025-26"][0])]


def release(hub: Path, target: str, origin: pd.Timestamp) -> pd.DataFrame:
    """The history for the round due on ``origin``: weekly rates, one column per country.

    The data are as published by the deadline, before any later revisions. The hub keeps a
    snapshot of each weekly data release from two sources: ERVISS for EU/EEA countries and WHO
    FluID for Switzerland and the UK nations. This takes the latest snapshot of each dated on or
    before the deadline. Rows are weeks, dated by their Sunday, up to the round's latest data week,
    which ends ten days before the deadline. Weeks a country did not report are NaN.
    """
    last = origin - pd.Timedelta(days=10)
    parts = []
    for source in SOURCES:
        folder = hub / "target-data" / source / "snapshots"
        files = sorted(folder.glob(f"*-{TARGETS[target]}.csv"))
        published = [file for file in files if pd.Timestamp(file.name[:10]) <= origin]
        if published:
            data = pd.read_csv(published[-1], parse_dates=["truth_date"])
            parts.append(data.pivot_table(index="truth_date", columns="location", values="value"))
    frame = pd.concat(parts, axis=1)
    frame = frame[frame.index <= last]
    frame = frame.reindex(pd.date_range(frame.index.min(), last, freq=FREQ))
    return frame[sorted(frame.columns)]


def truth(hub: Path) -> pd.Series:
    """What happened: the rates forecasts are scored against, by indicator, country and week.

    The hub's latest data from both sources, including every revision up to the pinned commit,
    unlike ``release``, which gives the data as they stood at a round's deadline.
    """
    folder = hub / "target-data"
    frames = [pd.read_csv(folder / f"latest-{stem}.csv") for stem in TARGETS.values()]
    frame = pd.concat(frames)
    frame["truth_date"] = pd.to_datetime(frame["truth_date"])
    return frame.set_index(["target", "location", "truth_date"])["value"]


def submissions(hub: Path, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Every submitted quantile forecast for ``origins``.

    One row per model, round, indicator, country and horizon.
    """
    frames = []
    for file in sorted((hub / "model-output").glob("*/*.csv")):
        if pd.Timestamp(file.name[:10]) in origins:
            frame = pd.read_csv(file, dtype={"output_type_id": str})
            frames.append(frame.assign(model=file.parent.name))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[(frame["output_type"] == "quantile") & frame["horizon"].isin(HORIZONS)].copy()
    frame["level"] = frame["output_type_id"].astype(float).round(3)
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    table = frame.pivot_table(
        index=["model", *TASK], columns="level", values="value", aggfunc="first"
    )
    table.columns = [f"q{level:g}" for level in table.columns]
    return table.reset_index()


def target_week(table: pd.DataFrame) -> pd.Series:
    """The week each forecast is for: ``horizon`` weeks after the round's latest data week."""
    latest = table["origin_date"] - pd.Timedelta(days=10)
    return latest + pd.to_timedelta(7 * table["horizon"], unit="D")
