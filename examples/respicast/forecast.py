"""Forecast every RespiCast round since August 2025 with ci-proto.

For each round and indicator, the histories of all countries go into a single call to the
model, using the data the hub had published by the round's deadline. The model has not been
fine-tuned and no settings were chosen on this data. The forecasts are saved for compare.py.

    python examples/respicast/forecast.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hub import FREQ, HORIZONS, OUTPUT, QUANTILES, TARGETS, TASK, checkout, release, rounds

from continuousintelligence import load_model

MODEL = "ContinuousIntelligence/ci-proto"
REVISION = "ee3031351a151e3f6febf8c62b53e6ea1008bef6"  # pinned Hub commit


def forecast(model, history: pd.DataFrame) -> np.ndarray:
    """Quantiles ``[locations, 9, horizons]`` from weeks x locations (NaN where missing)."""
    weeks = pd.date_range(history.index[0], periods=len(history) + len(HORIZONS), freq=FREQ)
    return model.forecast(
        torch.tensor(history.to_numpy(float).T, dtype=torch.float32),
        horizon=len(HORIZONS),
        timestamps=torch.tensor(weeks.as_unit("ns").asi8, dtype=torch.int64),
        freq=FREQ,
        series_ids=list(history.columns),
    ).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hub", type=Path, default=OUTPUT / "hub", help="hub checkout")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="folder for forecasts")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    hub = checkout(args.hub)
    model = load_model(MODEL, revision=REVISION, device=args.device)
    origins = rounds(hub)
    rows = []
    for count, origin in enumerate(origins, 1):
        for target in TARGETS:
            history = release(hub, target, origin)
            quantiles = forecast(model, history)
            for location, values in zip(history.columns, quantiles, strict=True):
                for step, horizon in enumerate(HORIZONS):
                    if np.isfinite(values[:, step]).all():
                        rows.append((origin, target, location, horizon, *values[:, step]))
        print(f"{count}/{len(origins)} rounds", end="\r", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "ci-proto.csv"
    table = pd.DataFrame(rows, columns=[*TASK, *QUANTILES])
    table.to_csv(path, index=False, float_format="%.9g")
    print(f"\nSaved {len(table):,} forecasts from {len(origins)} rounds to {path}")


if __name__ == "__main__":
    main()
