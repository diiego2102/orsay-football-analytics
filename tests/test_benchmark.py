import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from historical_benchmark import load_matches,season_split

class HistoricalContract(unittest.TestCase):
    def test_complete_public_fixture_and_half_time_consistency(self):
        matches,provenance=load_matches(ROOT/'data'/'public')
        self.assertEqual(len(matches),1140)
        self.assertEqual([p['halftime_derived_from_scoreless_fulltime'] for p in provenance],[23,11,16])
        self.assertTrue(matches.ht_home_goals.le(matches.ft_home_goals).all())
        self.assertTrue(matches.ht_away_goals.le(matches.ft_away_goals).all())
        train,test=season_split(matches)
        self.assertEqual((len(train),len(test)),(760,380))
        self.assertLess(train.utc_date.max(),test.utc_date.min())

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'strictly precede'):
            season_split(pd.DataFrame({'season_start_year':[2023,2024],
                'utc_date':pd.to_datetime(['2024-08-16','2024-08-16'],utc=True)}))
