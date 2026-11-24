# ci-proto on Europe's respiratory forecasting hub

[RespiCast](https://respicast.ecdc.europa.eu/) is the forecasting hub ECDC runs with the ISI
Foundation and LSHTM. Every week, teams send it forecasts of influenza-like illness (ILI) and
acute respiratory infection (ARI) across Europe. The hub combines them into an ensemble, the
median of the teams' forecasts, which ECDC publishes in its weekly
[ERVISS](https://erviss.org/) report.

These scripts replay every round since August 2025 with ci-proto and score it against the
ensemble and every submitted model. We use the released model at a pinned revision. The model
has not undergone any fine-tuning and no settings were chosen on this data. In each round, the
histories of all countries go into a single call per indicator, as raw weekly rates per
100,000. Each round uses only the data the hub had published by its deadline, and forecasts are
scored against the latest data.

ci-proto's training data contains no ERVISS or RespiCast data. European ILI data up to April
2025 form part of the [fev-bench](https://github.com/autogluon/fev) test set, on which ci-proto
has been publicly evaluated. The evaluation here starts with the round of 6 August 2025.

## Results

Forecasts are scored with the weighted interval score (WIS), the score RespiCast publishes for
every model. The table gives ci-proto's total WIS divided by the RespiCast ensemble's (below 1:
ci-proto had lower error), with 90% intervals from resampling rounds:

| Season | Rounds | Forecasts | Both indicators | ARI | ILI |
|---|---:|---:|---|---|---|
| 2025-26 | 50 | 7,409 | 0.71 (0.66-0.77) | 0.64 (0.59-0.70) | 0.87 (0.79-0.97) |
| 2026-27, to 18 November | 15 | 1,924 | 0.80 (0.77-0.83) | 0.74 (0.70-0.79) | 0.92 (0.86-0.99) |

ci-proto also had lower error than each of the 21 team models in 2025-26. So far in 2026-27,
it has lower error than each of the 15 team models that have forecast more than one round and
more than one country.

## Run

```bash
python examples/respicast/forecast.py  # writes ci-proto.csv; a GPU is used if there is one
python examples/respicast/compare.py   # writes headline.csv and versus.csv
```

The first run fetches the hub at the pinned commit (about 400 MB) into
`outputs/respicast/hub`; outputs go to `outputs/respicast`. Pass `--hub` to use an existing
checkout and `--output` for another folder.

## Scoring

- WIS ([Bracher et al. 2021](https://doi.org/10.1371/journal.pcbi.1008618)) rewards a median
  close to the outcome and ranges that are narrow but still contain it. Every model is scored on
  the nine quantile levels ci-proto produces, 10% to 90%.
- Models are scored on the same forecasts as RespiCast's own evaluation: the countries and
  horizons the hub asks for in each round.
- Each comparison is a ratio of total WIS. Intervals resample rounds (10,000 draws) and need at
  least three rounds.

## Credits

Forecasts and data come from ECDC's
[European Syndromic Indicators Forecasting Hub](https://github.com/european-modelling-hubs/RespiCast-SyndromicIndicators),
run with the ISI Foundation and LSHTM, and the teams who submit to it. Surveillance data are
from ERVISS (ECDC and the WHO Regional Office for Europe) and WHO FluID, as collected by the
hub. The scripts fetch these from the hub; none are included here.
