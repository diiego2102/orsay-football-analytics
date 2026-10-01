"""Fixed-season, walk-forward-feature benchmark of recovered MEC-1.2.1.

Run with CC0 OpenFootball JSON files in data/public. No network or token needed.
No test-set calibration, profit claim, or RBDT implementation is involved.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import mec1_pipeline_v121 as mec

SEASONS = ('2022-23', '2023-24', '2024-25')


def load_matches(folder):
    rows = []
    provenance = []
    for season in SEASONS:
        path = folder / f'{season}.json'
        raw = path.read_bytes()
        data = json.loads(raw)
        provenance.append(dict(season=season, sha256=hashlib.sha256(raw).hexdigest(),
            halftime_derived_from_scoreless_fulltime=0,
            source=f'https://github.com/openfootball/football.json/blob/master/{season}/en.1.json'))
        if len(data['matches']) != 380:
            raise ValueError(f'{season}: expected 380 Premier League matches')
        for m in data['matches']:
            ft, ht = m['score'].get('ft'), m['score'].get('ht')
            # A completed 0-0 necessarily has a 0-0 half-time score. The source
            # omits that redundant field; derive only this logically certain case.
            if ht is None and ft == [0, 0]:
                ht = [0, 0]
                provenance[-1]['halftime_derived_from_scoreless_fulltime'] += 1
            if ft is None or ht is None or len(ft) != 2 or len(ht) != 2:
                raise ValueError('Full-time and half-time scores are required; no imputation of results')
            if any(not isinstance(g, int) or g < 0 for g in ft + ht) or any(h > f for h, f in zip(ht, ft)):
                raise ValueError('Invalid goal counts')
            rows.append(dict(match_id=len(rows)+1, competition_code='PL', competition_name='Premier League',
                season_start_year=int(season[:4]), utc_date=pd.Timestamp(m['date'], tz='UTC'),
                matchday=int(m['round'].split()[-1]), status='FINISHED', home_team_name=m['team1'],
                away_team_name=m['team2'], ft_home_goals=ft[0], ft_away_goals=ft[1],
                ht_home_goals=ht[0], ht_away_goals=ht[1]))
    frame = pd.DataFrame(rows)
    names = sorted(set(frame.home_team_name) | set(frame.away_team_name))
    ids = {name: i+1 for i, name in enumerate(names)}
    frame['home_team_id'] = frame.home_team_name.map(ids)
    frame['away_team_id'] = frame.away_team_name.map(ids)
    if frame.duplicated(['utc_date', 'home_team_name', 'away_team_name']).any():
        raise ValueError('Duplicate fixture')
    # A team cannot play twice on one calendar day: date-only rolling features
    # therefore cannot include an earlier same-day result for that team.
    both = pd.concat([frame[['utc_date', 'home_team_name']].rename(columns={'home_team_name':'team'}),
                      frame[['utc_date', 'away_team_name']].rename(columns={'away_team_name':'team'})])
    if both.duplicated(['utc_date', 'team']).any():
        raise ValueError('Multiple games for one team on the same date')
    return frame, provenance


def season_split(gold):
    train = gold[gold.season_start_year.lt(2024)].copy()
    test = gold[gold.season_start_year.eq(2024)].copy()
    if train.empty or test.empty or train.utc_date.max() >= test.utc_date.min():
        raise ValueError('Training dates must strictly precede holdout dates')
    return train, test


def benchmark(folder=ROOT/'data'/'public'):
    frame, provenance = load_matches(folder)
    finished = mec.filter_finished_matches(frame)
    features = mec.add_rolling_team_features(mec.build_team_match_long(finished))
    gold = mec.build_model_dataset(finished, features)
    train, test = season_split(gold)
    _, numeric, categorical = mec.get_feature_columns(gold)
    models = mec.fit_mec1_models(train, numeric, categorical)
    predicted = mec.predict_lambdas(models, test, numeric, categorical)
    # Both models use the same fixed rho and score truncation. No parameter is
    # selected on the holdout. Original v1.2.1 source remains byte-for-byte intact.
    predictions = mec.add_probabilistic_outputs(predicted, dixon_coles_rho=-0.05, add_decision_layers=False)
    baseline = test.copy()
    for target in mec.TARGETS:
        baseline[f'lambda_{target}'] = float(train[target].mean())
    baseline['lambda_ft_home_goals'] = baseline.lambda_ht_home_goals + baseline.lambda_sh_home_goals
    baseline['lambda_ft_away_goals'] = baseline.lambda_ht_away_goals + baseline.lambda_sh_away_goals
    baseline['lambda_total_goals'] = baseline.lambda_ft_home_goals + baseline.lambda_ft_away_goals
    baseline = mec.add_probabilistic_outputs(baseline, dixon_coles_rho=-0.05, add_decision_layers=False)
    results = []
    for label, pred in [('MEC-1.2.1 uncalibrated', predictions), ('Training-average Poisson', baseline)]:
        probabilities = pred[['ft_prob_home_win','ft_prob_draw','ft_prob_away_win']].to_numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError('Invalid probability distribution')
        metrics = mec.evaluate_predictions(pred).iloc[0].to_dict()
        results.append(dict(model=label, **{k: metrics[k] for k in ('n_matches_test','log_loss_1x2',
            'brier_score_1x2','accuracy_1x2','mae_ft_home_goals','mae_ft_away_goals')}))
    output = ROOT/'reports'
    output.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(output/'historical_metrics.csv', index=False)
    manifest = dict(source='OpenFootball football.json', data_license='CC0-1.0', files=provenance,
        training_seasons=list(SEASONS[:2]), test_season=SEASONS[-1], n_train=len(train), n_test=len(test),
        mode='Fixed model; prior observed results update rolling features and Elo during holdout',
        calibration='None; alpha=0.01 and rho=-0.05 fixed',
        brier_definition='Sum of squared H/D/A probability errors divided by 3',
        limitations=['Single league and holdout season; no statistical superiority established',
            'Same-day team uniqueness checked; timestamps have calendar-day resolution',
            'Recovered feature selection retains numeric team-check identifiers and year; ablation is future work',
            'No odds, ROI, profitability, API, or RBDT-26 validation'])
    (output/'historical_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False))
    return results


if __name__ == '__main__':
    benchmark()
