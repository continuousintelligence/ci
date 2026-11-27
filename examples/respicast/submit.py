"""Make ci-proto's RespiCast submission for the round that is open now.

RespiCast takes forecasts each week until 23:59 CET on Wednesday, for the round named after that
Wednesday. This reads a checkout of the hub's main branch, forecasts with ci-proto from the hub's
latest data release exactly as forecast.py does, puts the nine quantiles on the hub's 23 levels,
checks the file against the hub's rules and writes it. The weekly workflow
(.github/workflows/respicast.yml) runs it and then opens the pull request to the hub.

    python examples/respicast/submit.py --hub HUB [--round YYYY-MM-DD] [--output FOLDER] [--dry-run]

Live runs require an open submission window. Use --dry-run to prepare historical rounds locally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from forecast import MODEL, REVISION, forecast
from hub import HORIZONS, LEVELS, OUTPUT, TARGETS, release

from continuousintelligence import load_model

NAME = "CI-ci_proto"  # team and model abbreviations, as registered with the hub
HUB_LEVELS = [0.01, 0.025, 0.05, *(round(k / 20, 2) for k in range(2, 19)), 0.95, 0.975, 0.99]
STALE_WEEKS = 8  # countries with no report in this many weeks are left out
COLUMNS = [
    "origin_date",
    "target",
    "target_end_date",
    "horizon",
    "location",
    "output_type",
    "output_type_id",
    "value",
]


def options(hub: Path, target: str) -> dict[str, set]:
    """The values the hub accepts for ``target``: rounds, countries and target weeks."""
    config = json.loads((hub / "hub-config" / "tasks.json").read_text())["rounds"][0]
    for task in config["model_tasks"]:
        ids = task["task_ids"]
        if ids["target"]["required"] == [target]:
            return {k: set(ids[k]["required"] or []) | set(ids[k]["optional"] or []) for k in ids}
    raise SystemExit(f"The hub does not list {target!r} in hub-config/tasks.json.")


def window(hub: Path) -> tuple[int, int]:
    """Days before the round date that submissions open and close."""
    due = json.loads((hub / "hub-config" / "tasks.json").read_text())["rounds"][0]
    return due["submissions_due"]["start"], due["submissions_due"]["end"]


def check_window(hub: Path, origin: pd.Timestamp, now: pd.Timestamp) -> None:
    """Reject live forecasts outside the hub's submission window."""
    start, end = window(hub)
    timezone = json.loads((hub / "hub-config" / "admin.json").read_text())["timezone"]
    today = now.tz_convert(timezone).tz_localize(None).normalize()
    opens, closes = origin + pd.Timedelta(days=start), origin + pd.Timedelta(days=end)
    if not opens <= today <= closes:
        raise SystemExit(
            f"The {origin.date()} round only accepts submissions from {opens.date()} through "
            f"{closes.date()} in {timezone}. Use --dry-run to prepare a historical round."
        )


def open_round(hub: Path, today: pd.Timestamp) -> pd.Timestamp:
    """The first round on or after ``today``: the one taking submissions now."""
    timezone = json.loads((hub / "hub-config" / "admin.json").read_text())["timezone"]
    today = today.tz_convert(timezone).tz_localize(None)
    dates = sorted(pd.Timestamp(d) for d in options(hub, next(iter(TARGETS)))["origin_date"])
    later = [d for d in dates if d >= today.normalize()]
    if not later:
        raise SystemExit(f"The hub lists no round on or after {today.date()}.")
    return later[0]


def latest_release(hub: Path, target: str) -> pd.Timestamp:
    """Date of the hub's latest ERVISS data release for ``target``."""
    files = (hub / "target-data" / "ERVISS" / "snapshots").glob(f"*-{TARGETS[target]}.csv")
    return max(pd.Timestamp(file.name[:10]) for file in files)


def extend(q9: np.ndarray) -> np.ndarray:
    """Nine quantiles ``[N, 9]`` at 10% to 90% -> the hub's 23 levels ``[N, 23]``.

    Linear in log(1 + value) against the normal score of the level, with the end segments
    continued into the tails, so the outer levels widen the way the inner ones do.
    """
    normal = NormalDist()
    x = np.array([normal.inv_cdf(level) for level in LEVELS])
    target = np.array([normal.inv_cdf(level) for level in HUB_LEVELS])
    values = np.log1p(np.maximum.accumulate(np.maximum(q9, 0.0), axis=-1))
    seg = np.clip(np.searchsorted(x, target, side="right") - 1, 0, len(x) - 2)
    left, right = values[:, seg], values[:, seg + 1]
    weight = (target - x[seg]) / (x[seg + 1] - x[seg])
    return np.maximum(np.expm1(left + weight * (right - left)), 0.0)


def submission(hub: Path, origin: pd.Timestamp, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The hub-format table for ``origin``, and each country's latest reported week."""
    rows, report = [], []
    for target in TARGETS:
        allowed = options(hub, target)
        history = release(hub, target, origin)
        quantiles = forecast(model, history)  # [locations, 9, horizons]
        for location, q9 in zip(history.columns, quantiles, strict=True):
            reported = history[location].dropna()
            last = reported.index.max()
            report.append((target, location, last, location in allowed["location"]))
            if location not in allowed["location"] or not np.isfinite(q9).all():
                continue
            if pd.isna(last) or last < history.index[-1] - pd.Timedelta(weeks=STALE_WEEKS):
                continue
            q23 = extend(q9.T)  # [horizons, 23]
            for step, horizon in enumerate(HORIZONS):
                end = origin - pd.Timedelta(days=10) + pd.Timedelta(weeks=horizon)
                base = (origin.date().isoformat(), target, end.date().isoformat(), horizon)
                rows.append((*base, location, "median", "", q9[4, step]))
                for level, value in zip(HUB_LEVELS, q23[step], strict=True):
                    rows.append((*base, location, "quantile", f"{level:g}", value))
    table = pd.DataFrame(rows, columns=COLUMNS)
    table["value"] = table["value"].astype(float).round(4)
    report = pd.DataFrame(report, columns=["target", "location", "last_reported", "hub_location"])
    report["weeks_behind"] = (origin - pd.Timedelta(days=10) - report["last_reported"]).dt.days // 7
    return table, report


def check(table: pd.DataFrame, hub: Path, origin: pd.Timestamp) -> None:
    """Stop with a message if the table breaks any of the hub's rules."""
    problems = []
    if table.empty:
        problems.append("no forecasts")
    if not (np.isfinite(table["value"]).all() and (table["value"] >= 0).all()):
        problems.append("values that are missing, infinite or negative")
    if table.duplicated(COLUMNS[:-1]).any():
        problems.append("duplicate rows")
    for target, part in table.groupby("target"):
        allowed = options(hub, target)
        for column in ("origin_date", "target_end_date", "location", "horizon"):
            values = part[column] if column != "horizon" else part[column].astype(int)
            if not set(values) <= allowed[column]:
                problems.append(f"{target}: {column} values the hub does not accept")
    keys = ["target", "location", "horizon"]
    quantiles = table[table["output_type"] == "quantile"]
    if not (quantiles.groupby(keys).size() == len(HUB_LEVELS)).all():
        problems.append("forecasts without all 23 quantile levels")
    if not quantiles.groupby(keys)["value"].apply(lambda v: (np.diff(v) >= 0).all()).all():
        problems.append("quantiles that decrease")
    if not (table.groupby(["target", "location"])["horizon"].nunique() == len(HORIZONS)).all():
        problems.append("countries without all four horizons")
    if (table["origin_date"] != origin.date().isoformat()).any():
        problems.append("rows for another round")
    if problems:
        raise SystemExit("The submission is not valid: " + "; ".join(problems) + ".")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hub", type=Path, required=True, help="checkout of the hub's main")
    parser.add_argument("--round", help="round date (a Wednesday); default: the open round")
    parser.add_argument("--output", type=Path, default=OUTPUT / "submission")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dry-run", action="store_true", help="prepare a file even if the round is closed"
    )
    args = parser.parse_args()

    today = pd.Timestamp.now(tz="UTC")
    origin = pd.Timestamp(args.round) if args.round else open_round(args.hub, today)
    if not args.dry_run:
        check_window(args.hub, origin, today)
    start, _ = window(args.hub)
    released = {target: latest_release(args.hub, target) for target in TARGETS}
    for target, date in released.items():
        if date < origin + pd.Timedelta(days=start):
            raise SystemExit(
                f"The hub's latest {target} data are from {date.date()}, before the "
                f"{origin.date()} round opened; the new release has not arrived yet."
            )
    print(f"Round {origin.date()}, data released {min(released.values()).date()}")

    model = load_model(MODEL, revision=REVISION, device=args.device)
    table, report = submission(args.hub, origin, model)
    check(table, args.hub, origin)
    if not args.dry_run:
        check_window(args.hub, origin, pd.Timestamp.now(tz="UTC"))

    relative = Path("model-output") / NAME / f"{origin.date()}-{NAME}.csv"
    path = args.output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    counts = table.groupby("target")["location"].nunique().to_dict()
    print(f"Wrote {path}: {len(table):,} rows, countries per indicator {counts}")
    behind = report[(report["weeks_behind"] > 0) | ~report["hub_location"]]
    if not behind.empty:
        print("Countries behind on reporting or outside the hub's list:")
        print(behind.sort_values(["target", "weeks_behind"]).to_string(index=False))
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            out.write(f"round={origin.date()}\nfile={relative}\n")


if __name__ == "__main__":
    main()
