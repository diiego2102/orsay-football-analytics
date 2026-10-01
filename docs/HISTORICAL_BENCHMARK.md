# Historical benchmark of MEC Football Analytics

## Result
On 380 Premier League matches in 2024–25, recovered MEC-1.2.1 with fixed parameters improved outcome-probability scoring over a training-average Poisson reference. It performed worse on goal-count error. This is one holdout season and does not establish general superiority, profitability or betting returns.

| Metric | MEC-1.2.1 | Training-average Poisson | Preferred direction |
|---|---:|---:|---|
| 1X2 log loss | 1.060174 | 1.084312 | Lower |
| Brier score divided by 3 | 0.210398 | 0.219324 | Lower |
| 1X2 argmax accuracy | 50.7895% | 40.7895% | Higher |
| Home goals MAE | 1.124060 | 1.071364 | Lower |
| Away goals MAE | 1.035967 | 0.934003 | Lower |

## Protocol
Training: 760 matches in 2022–23 and 2023–24. Holdout: all 380 matches in 2024–25. Models remain fixed during the holdout; rolling histories and Elo use earlier observed results as the season progresses. This represents sequential pre-match predictions, not a forecast of the entire season at its opening.

Four Poisson regressors estimate home and away goal intensity in each half. Historical rolling features and pre-match Elo inform the predictions. Intensities form a truncated scoreline distribution with fixed Dixon–Coles rho -0.05. Alpha is fixed at 0.01. No outcome calibration or decision-band tuning is fitted to the test season. The baseline uses four training-only average half-time goal counts and the same score-matrix settings.

OpenFootball date-only records were checked for duplicate fixtures and multiple same-day games for one team. Missing half-time fields occur only for 0–0 full-time games: half-time must also be 0–0. This logical derivation is counted in the manifest (23, 11 and 16 games respectively); no unknown nonzero result is imputed.

## Data and reproduction
The three JSON fixtures come from [OpenFootball football.json](https://github.com/openfootball/football.json), released under [CC0](https://github.com/openfootball/football.json/blob/master/LICENSE.md). Copies are in `data/public`; source links and SHA-256 hashes are in `reports/historical_manifest.json`. Raw data from other providers is not distributed.

```bash
python -m pip install -r requirements.txt
python examples/historical_benchmark.py
python -m unittest discover -s tests -v
```

The benchmark uses recovered v1.2.1 without altering its source. The feature selector retains numeric team-check IDs and season year; ablations and season-reset effects remain open methodological questions. No confidence interval, multi-season superiority, API ingestion, commercial service performance or RBDT-26 execution is established here.

The first local run used the runtime's bundled pip-vendored Requests module only to satisfy import-time dependencies; the benchmark made no HTTP calls. GitHub CI installs ordinary requirements and reruns the benchmark independently.
