# Orsay · Football Analytics

**From match history to probabilistic decisions.**

A personal analytics project by **Diego Gallo**, combining data engineering, statistical modelling and clear communication of uncertainty. The recovered **MEC-1.2.1** engine models home and away goals in each half, then converts expected goal counts into scoreline and match-outcome probabilities.

Orsay brings together a personal football-analytics project and its current technical evolution. Model logic, data preparation and evaluation remain reviewable in code and notebooks.

## What the project demonstrates

- A raw → bronze → silver → gold data workflow.
- Historical football data ingestion, normalisation and reusable analytical tables.
- Rolling form features and pre-match Elo ratings.
- Four regularised Poisson regressions for first- and second-half goals.
- A Poisson score matrix with a Dixon–Coles adjustment for low-scoring outcomes.
- Temporal training, calibration and test partitions in the recovered pipeline.
- Evaluation functions for log loss, Brier score, goal MAE and result-band coverage.
- Excel exports for independent review of predictions and model inputs.

## The mathematical model

For each side and half, the model estimates a positive goal intensity using a Poisson regression:

$$\lambda = \exp(\beta_0 + X\beta)$$

Full-time intensity is the sum of the two half intensities. Before the low-score adjustment, a scoreline has probability:

$$P(H=h,A=a)=\operatorname{Pois}(h;\lambda_H)\operatorname{Pois}(a;\lambda_A)$$

The engine applies a Dixon–Coles correction to 0–0, 0–1, 1–0 and 1–1, then normalises the truncated score matrix. Summing its cells produces home-win, draw and away-win probabilities, alongside totals and both-teams-to-score probabilities.

These intensities are **model-projected goal counts**, not shot-location-based xG. A likely scoreline is one possible outcome, not a promise.

## Quick start

Run from the root of this repository (`orsay-football-analytics`).

```bash
python -m venv .venv
# Activate your environment, then:
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/offline_demo.py
```

The demo generates fictitious teams and matches with a fixed random seed. It trains the recovered Poisson components and produces CSV outputs in `outputs/demo/`. It makes no network requests and needs no API token. It does not reproduce the complete calibration workflow or measure performance on real football data.

To run the historical-data pipeline, install `requirements-full.txt`, set `FOOTBALL_DATA_API_TOKEN` in a local environment file and use `notebooks/01_mec1_modelo_prepartido_v121.ipynb`. Live extraction and the original notebook have not been rerun during this portfolio preparation. API access and dataset permissions must be provided locally.

See **[Running guide / Guía de ejecución](docs/RUNNING.md)** for Windows, macOS/Linux and GitHub Codespaces. Python 3.11 or later is required.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/mec1_pipeline_v121.py` | Recovered MEC-1.2.1 engine; source preserved unchanged |
| `src/mec1_pipeline_v12.py`, `src/mec1_pipeline_v11.py`, `src/mec1_pipeline.py` | Earlier recovered engines retained for comparison |
| `notebooks/` | Recovered notebooks, with execution outputs cleared |
| `examples/offline_demo.py` | Reproducible synthetic demonstration |
| `tests/` | Probability, temporal-split and feature-timing checks |
| `CHANGELOG.md` | Portfolio packaging changes, separate from model versions |
| `docs/ROADMAP.md` | Prioritised evolution of Orsay and complementary projects |

## Status and evaluation

**Implemented in recovered code:** feature engineering, Elo, Poisson models, score distributions, temporal calibration, evaluation and Excel export. **Rechecked here:** the offline synthetic demo and the tests described in `docs/VALIDATION.md`.

Historical predictive accuracy and financial returns are not asserted in this repository. A chronological benchmark against simple baselines and reproducible real-data evaluation remain the next substantive steps.

**RBDT-26** belongs to Orsay's experimental research direction. No runnable RBDT-26 implementation was recovered in this package, so it is not presented as a validated feature of MEC-1.2.1. The original Orsay work and its current technical evolution remain one personal project, with their stages described honestly.

## Resumen en español

Orsay es mi proyecto propio de analítica de fútbol. MEC-1.2.1 combina preparación de datos, forma reciente, Elo, regresión de Poisson y ajuste Dixon–Coles para producir probabilidades prepartido y comunicar incertidumbre. Esta presentación conserva el trabajo anterior y añade una demostración con datos sintéticos y controles reproducibles. RBDT-26 se mantiene como línea experimental cuya implementación debe recuperarse o desarrollarse antes de presentarla como validada.

## Data and publication

Only generated demonstration data is included. No company datasets, credentials, private infrastructure or employer source code are included. API data should stay local unless redistribution rights are established. Source code is available for portfolio review; no open-source licence has been assigned.

## Clone this project

```bash
git clone https://github.com/diiego2102/orsay-football-analytics.git
cd orsay-football-analytics
```

Follow the run commands above from this folder.
