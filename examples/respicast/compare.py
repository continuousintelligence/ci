"""Score ci-proto against the RespiCast ensemble and every submitted model.

Forecasts are scored with the weighted interval score (WIS, Bracher et al. 2021) on the nine
quantile levels ci-proto produces, 10% to 90%, against the latest data at the pinned commit.
Models are scored on the same forecasts as RespiCast's own evaluation: the countries and
horizons the hub asks for in each round. Each comparison divides ci-proto's total WIS by the
other model's, counting only the rounds, countries and horizons both forecast (below 1: ci-proto
had lower error), with a 90% interval from resampling rounds.

    python examples/respicast/compare.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from hub import (
    BASELINE,
    CI,
    ENSEMBLE,
    LEVELS,
    OUTPUT,
    QUANTILES,
    TASK,
    checkout,
    rounds,
    season,
    submissions,
    target_week,
    truth,
)


def wis(quantiles: np.ndarray, observed: np.ndarray, levels: list[float]) -> np.ndarray:
    """Weighted interval score from quantiles ``[N, levels]`` at symmetric levels with a median."""
    middle = len(levels) // 2
    score = 0.5 * np.abs(observed - quantiles[:, middle])
    for k in range(middle):
        alpha = 2 * levels[k]
        lower, upper = quantiles[:, k], quantiles[:, -1 - k]
        below = np.maximum(lower - observed, 0)
        above = np.maximum(observed - upper, 0)
        score = score + alpha / 2 * (upper - lower + 2 / alpha * (below + above))
    return score / (middle + 0.5)


def paired(scores: pd.DataFrame, reference: str, draws: int = 10_000) -> dict:
    """ci-proto's total WIS divided by ``reference``'s.

    Only the rounds, countries and horizons both forecast count, so neither model is scored on
    forecasts the other did not make.
    """
    ours = scores[scores["model"] == CI]
    theirs = scores.loc[scores["model"] == reference, [*TASK, "wis"]]
    both = ours.merge(theirs, on=TASK, suffixes=("", "_ref"))
    weekly = both.groupby("origin_date")[["wis", "wis_ref"]].sum()
    result = {
        "ratio": both["wis"].sum() / both["wis_ref"].sum(),
        "low": np.nan,
        "high": np.nan,
        "rounds": len(weekly),
        "forecasts": len(both),
        "locations": both["location"].nunique(),
    }
    if len(weekly) >= 3:  # too few rounds to resample otherwise
        pick = np.random.default_rng(12345).integers(0, len(weekly), (draws, len(weekly)))
        ratios = weekly["wis"].to_numpy()[pick].sum(1) / weekly["wis_ref"].to_numpy()[pick].sum(1)
        result["low"], result["high"] = np.quantile(ratios, [0.05, 0.95])
    return result


def score(hub: Path, output: Path) -> pd.DataFrame:
    """Every forecast RespiCast evaluates, with its outcome and WIS."""
    origins = rounds(hub)
    theirs = submissions(hub, origins)
    ours = pd.read_csv(output / "ci-proto.csv", parse_dates=["origin_date"]).assign(model=CI)
    table = pd.concat([theirs, ours], ignore_index=True)
    table = table[table["origin_date"].isin(origins)]
    keys = pd.MultiIndex.from_arrays([table["target"], table["location"], target_week(table)])
    table["observed"] = truth(hub).reindex(keys).to_numpy()
    table = table.dropna(subset=["observed", *QUANTILES]).copy()
    table["wis"] = wis(table[QUANTILES].to_numpy(), table["observed"].to_numpy(), LEVELS)
    table["season"] = table["origin_date"].map(season)
    tasks = table.loc[table["model"] == BASELINE, TASK]
    return table.merge(tasks, on=TASK)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hub", type=Path, default=OUTPUT / "hub", help="hub checkout")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="folder for forecasts")
    args = parser.parse_args()
    scores = score(checkout(args.hub), args.output)

    headline, versus = [], []
    for name, part in scores.groupby("season"):
        for target in ("both", "ARI incidence", "ILI incidence"):
            subset = part if target == "both" else part[part["target"] == target]
            headline.append({"season": name, "target": target, **paired(subset, ENSEMBLE)})
        for model in sorted(set(part["model"]) - {CI}):
            versus.append({"season": name, "model": model, **paired(part, model)})
    headline, versus = pd.DataFrame(headline), pd.DataFrame(versus)
    versus = versus.sort_values(["season", "ratio"])
    headline.to_csv(args.output / "headline.csv", index=False)
    versus.to_csv(args.output / "versus.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nci-proto's total WIS / the RespiCast ensemble's:")
    print(headline.round(3).to_string(index=False))
    print("\nci-proto's total WIS / each submitted model's:")
    print(versus.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
