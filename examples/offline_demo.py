"""Reproducible synthetic demo of recovered MEC components; no network calls."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mec1_pipeline_v121 as mec


def synthetic_matches(n_matches=180, seed=26):
    rng = np.random.default_rng(seed)
    rows = []
    strength = [0.85, 1.0, 1.15, 1.3, 1.05, 0.95]
    for i in range(n_matches):
        home, away = rng.choice(6, size=2, replace=False)
        hh, ah = rng.poisson(0.65 * strength[home]), rng.poisson(0.50 * strength[away])
        hs, ass = rng.poisson(0.75 * strength[home]), rng.poisson(0.60 * strength[away])
        rows.append(dict(match_id=i+1, competition_code="DEMO", competition_name="Synthetic League",
                         season_start_year=2025, utc_date=pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i),
                         matchday=i+1, status="FINISHED", home_team_id=int(home), away_team_id=int(away),
                         home_team_name=f"Demo Club {home+1}", away_team_name=f"Demo Club {away+1}",
                         ht_home_goals=int(hh), ht_away_goals=int(ah),
                         ft_home_goals=int(hh+hs), ft_away_goals=int(ah+ass)))
    return pd.DataFrame(rows)


def main():
    finished = mec.filter_finished_matches(synthetic_matches())
    long = mec.build_team_match_long(finished)
    features = mec.add_rolling_team_features(long)
    gold = mec.build_model_dataset(finished, features)
    _, numeric, categorical = mec.get_feature_columns(gold)
    train, test = mec.temporal_train_test_split(gold)
    models = mec.fit_mec1_models(train, numeric, categorical)
    predictions = mec.predict_lambdas(models, test, numeric, categorical)
    predictions = mec.add_probabilistic_outputs(predictions)
    metrics = mec.evaluate_predictions(predictions)
    probabilities = predictions[["ft_prob_home_win", "ft_prob_draw", "ft_prob_away_win"]]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities.to_numpy()).all()
    assert train.utc_date.max() < test.utc_date.min()
    output = ROOT / "outputs" / "demo"
    output.mkdir(parents=True, exist_ok=True)
    keep = ["match_id", "home_team_name", "away_team_name", "lambda_ft_home_goals", "lambda_ft_away_goals",
            "ft_prob_home_win", "ft_prob_draw", "ft_prob_away_win", "ft_top_scoreline", "predicted_result"]
    predictions[keep].to_csv(output / "synthetic_predictions.csv", index=False)
    metrics.to_csv(output / "synthetic_metrics.csv", index=False)
    print(f"Synthetic-only demo passed: {len(train)} train matches; {len(test)} holdout matches.")
    print("These outputs verify execution, not real-world forecasting performance.")


if __name__ == "__main__":
    main()
