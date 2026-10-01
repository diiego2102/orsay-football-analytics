"""Checks that target probability and temporal invariants, rather than file layout."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import mec1_pipeline_v121 as mec
from offline_demo import synthetic_matches


class ModelIntegrity(unittest.TestCase):
    def test_score_matrix_is_a_distribution(self):
        for home, away in [(0.1, 0.1), (1.4, 1.1), (4.0, 3.0)]:
            matrix = mec.poisson_score_matrix(home, away)
            self.assertTrue(np.isfinite(matrix.probability).all())
            self.assertTrue(matrix.probability.ge(0).all())
            self.assertAlmostEqual(float(matrix.probability.sum()), 1.0, places=12)

    def test_outcomes_partition_probability(self):
        summary = mec.summarize_score_matrix(mec.poisson_score_matrix(1.6, 1.0))
        self.assertAlmostEqual(sum(summary[k] for k in ["prob_home_win", "prob_draw", "prob_away_win"]), 1.0)
        self.assertGreaterEqual(summary["prob_over_15"], summary["prob_over_25"])
        self.assertGreaterEqual(summary["prob_over_25"], summary["prob_over_35"])

    def test_dixon_coles_changes_low_scores(self):
        base = mec.poisson_score_matrix(1.4, 1.1, use_dixon_coles=False).set_index("scoreline")
        adjusted = mec.poisson_score_matrix(1.4, 1.1, rho=-0.05).set_index("scoreline")
        self.assertGreater(adjusted.loc["0-0", "probability"], base.loc["0-0", "probability"])
        self.assertGreater(adjusted.loc["1-1", "probability"], base.loc["1-1", "probability"])

    def test_temporal_holdout_is_later(self):
        frame = synthetic_matches(80).sample(frac=1, random_state=7)
        train, test = mec.temporal_train_test_split(frame)
        self.assertLess(train.utc_date.max(), test.utc_date.min())
        self.assertTrue(set(train.match_id).isdisjoint(set(test.match_id)))

    def test_current_result_does_not_enter_its_own_features(self):
        original = mec.filter_finished_matches(synthetic_matches(80))
        changed = original.copy()
        row = changed.index[50]
        changed.loc[row, "ft_home_goals"] += 9
        changed.loc[row, "sh_home_goals"] += 9
        outputs = []
        for frame in [original, changed]:
            features = mec.add_rolling_team_features(mec.build_team_match_long(frame))
            outputs.append(features[features.match_id.eq(51)].sort_values("team_id").reset_index(drop=True))
        cols = [x for x in outputs[0] if x.startswith("roll")]
        pd.testing.assert_frame_equal(outputs[0][cols], outputs[1][cols])
        ratings = [mec.build_elo_features(frame).set_index("match_id").loc[51] for frame in [original, changed]]
        pd.testing.assert_series_equal(ratings[0], ratings[1])


if __name__ == "__main__":
    unittest.main()
