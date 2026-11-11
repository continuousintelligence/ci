"""The bundled real-data forecasting example."""

import csv
from importlib.resources import files

import torch
from torch import Tensor


def load_example() -> Tensor:
    """Return the example's observed history as a CPU float32 tensor ``[1, 120]``.

    Forecast rows are excluded. Each call returns a new tensor. Data attribution
    and source information accompany the CSV in ``_data/NOTICE``.
    """
    with files("continuousintelligence").joinpath("_data/bike_sharing.csv").open() as stream:
        values = [
            float(row["rentals"]) for row in csv.DictReader(stream) if row["split"] == "history"
        ]
    return torch.tensor([values], dtype=torch.float32)
