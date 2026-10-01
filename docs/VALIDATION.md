# Validation of portfolio package 0.1.0

Run date: 2026-10-01. Recovered model: MEC-1.2.1.

## Executed checks

Five `unittest` checks passed:

1. Score matrices have finite, non-negative probabilities and sum to one.
2. Home/draw/away probabilities partition the score matrix; total-goal thresholds are ordered consistently.
3. The chosen negative Dixon–Coles coefficient increases the checked low-score draw probabilities.
4. The chronological holdout contains later matches and has no overlapping match identifiers.
5. Changing a match's result does not change its own rolling features or pre-match Elo values in the tested synthetic example.

The offline demo ran successfully on **180 synthetic matches**, using **144 for training and 36 for holdout**. Four recovered Poisson models trained, probability outputs were generated, and the probability normalisation and chronological split assertions passed. CSV predictions and synthetic evaluation metrics were written locally.

These checks verify the exercised code paths. They do not establish historical forecasting accuracy or prove the absence of leakage in all datasets and use cases.

## Execution environment

Python 3.12; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0.

The execution host lacked a separately installed `requests` package. For the offline validation only, its available pip-vendored Requests 2.34.2 was temporarily mapped to `requests` and `requests.exceptions` in memory. No HTTP calls were made, and engine source was not changed. A fresh installation should use the normal `requests` dependency in `requirements.txt` and rerun the documented commands.

## Not executed

- Original notebooks top to bottom.
- Live API extraction, account access or dataset-permission checks.
- Full historical-data training and temporal calibration pipeline.
- Parquet and Excel export paths in the full pipeline.
- RBDT-26, for which no implementation was recovered in this package.

## Preservation and publication preparation

- Recovered engine source SHA-256: `c87262d42faa19cf62a92d12df3eae17de35cc40e3733afc0be1037fb093cf46`.
- Earlier recovered engine versions were retained as imported snapshots, not reconstructed Git history.
- Notebook execution outputs and bytecode caches were excluded.
- A bounded scan found no employer-name references, common private IPv4 addresses or matching long credential literals in the prepared text files. This is a preparation check, not a security guarantee.
- Generated data and prediction files are ignored by Git and excluded from the deliverable ZIP.

## Portfolio v0.3.0

Seven tests now cover the historical fixture contract, strictly separated seasons, probabilities and feature timing. The new benchmark fits on 760 matches and evaluates on 380; see `HISTORICAL_BENCHMARK.md`, CSV and manifest. API ingestion and the full calibrated notebook remain untested. Local imports used bundled pip-vendored Requests; GitHub CI installs ordinary requirements and reruns offline execution.
