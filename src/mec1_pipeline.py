"""
MEC-1 — Motor Estadístico de Clubes v1
======================================

Pipeline inicial para predicción prepartido de fútbol de clubes:
- Ingesta histórica desde football-data.org.
- Bronze/Silver/Gold como Parquet comprimido con gzip y extensión .gzip.
- Feature engineering sin leakage temporal.
- Modelos Poisson para goles 1T y 2T.
- Matriz de marcadores, 1X2, over/under y BTTS.
- Exportación a Excel.

Requisitos:
    pip install pandas numpy requests scipy scikit-learn openpyxl pyarrow

Variable de entorno recomendada:
    FOOTBALL_DATA_API_TOKEN=<tu_token>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_project_root() -> Path:
    """PROJECT_ROOT robusto: funciona desde notebooks/ o desde la raíz."""
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name.lower() == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT


def ensure_project_dirs(project_root: Path) -> Dict[str, Path]:
    paths = {
        "raw_football_data": project_root / "data" / "raw" / "football_data",
        "raw_statsbomb": project_root / "data" / "raw" / "statsbomb",
        "bronze": project_root / "data" / "bronze",
        "silver": project_root / "data" / "silver",
        "gold": project_root / "data" / "gold",
        "outputs_excel": project_root / "outputs" / "excel",
        "outputs_charts": project_root / "outputs" / "charts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parquet_gzip_path(folder: Path, name: str) -> Path:
    """
    Devuelve ruta .gzip. El contenido es Parquet comprimido con gzip.
    Se lee con pd.read_parquet().
    """
    return folder / f"{name}.gzip"


def save_parquet_gzip(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression="gzip", index=False)
    return output_path


def read_parquet_gzip(input_path: Path) -> pd.DataFrame:
    return pd.read_parquet(input_path)


def save_json_gzip(obj: dict, output_path: Path) -> Path:
    """Solo raw JSON. Bronze/Silver/Gold se guardan en Parquet gzip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return output_path


def read_json_gzip(input_path: Path) -> dict:
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class FootballDataClient:
    api_token: Optional[str] = None
    base_url: str = "https://api.football-data.org/v4"
    sleep_seconds: float = 6.5
    timeout: int = 30

    def __post_init__(self):
        if self.api_token is None:
            self.api_token = os.getenv("FOOTBALL_DATA_API_TOKEN")
        self.session = requests.Session()
        if self.api_token:
            self.session.headers.update({"X-Auth-Token": self.api_token})

    def get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", "60"))
            print(f"[WARN] Rate limit. Esperando {retry_after}s...")
            time.sleep(retry_after)
            response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code == 403:
            raise PermissionError(
                "football-data.org devolvió 403. Revisa tu token, plan o acceso a la competición/temporada."
            )
        response.raise_for_status()
        return response.json()


DEFAULT_COMPETITIONS = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL"]


def fetch_competition_matches(
    client: FootballDataClient,
    competitions: Iterable[str],
    seasons: Iterable[int],
    raw_dir: Path,
    force: bool = False,
) -> List[Path]:
    saved_files: List[Path] = []
    for comp in competitions:
        for season in seasons:
            output = raw_dir / f"matches_{comp}_{season}.json.gz"
            if output.exists() and not force:
                print(f"[OK] Raw existe: {output.name}")
                saved_files.append(output)
                continue
            print(f"[GET] {comp} season={season}")
            payload = client.get(f"competitions/{comp}/matches", params={"season": season})
            save_json_gzip(payload, output)
            saved_files.append(output)
            time.sleep(client.sleep_seconds)
    return saved_files


def normalize_football_data_matches(raw_files: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for file in raw_files:
        payload = read_json_gzip(file)
        for m in payload.get("matches", []):
            score = m.get("score") or {}
            ft = score.get("fullTime") or {}
            ht = score.get("halfTime") or {}
            comp = m.get("competition") or {}
            season = m.get("season") or {}
            home = m.get("homeTeam") or {}
            away = m.get("awayTeam") or {}
            rows.append({
                "source": "football-data.org",
                "competition_id": comp.get("id"),
                "competition_code": comp.get("code"),
                "competition_name": comp.get("name"),
                "season_start_year": season.get("startDate", "")[:4] if season.get("startDate") else None,
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status"),
                "stage": m.get("stage"),
                "group": m.get("group"),
                "matchday": m.get("matchday"),
                "home_team_id": home.get("id"),
                "home_team_name": home.get("name"),
                "away_team_id": away.get("id"),
                "away_team_name": away.get("name"),
                "ft_home_goals": ft.get("home"),
                "ft_away_goals": ft.get("away"),
                "ht_home_goals": ht.get("home"),
                "ht_away_goals": ht.get("away"),
                "last_updated": m.get("lastUpdated"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["utc_date"] = pd.to_datetime(df["utc_date"], errors="coerce", utc=True)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
    num_cols = [
        "competition_id", "season_start_year", "match_id", "matchday", "home_team_id", "away_team_id",
        "ft_home_goals", "ft_away_goals", "ht_home_goals", "ht_away_goals"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates("match_id").sort_values(["utc_date", "match_id"]).reset_index(drop=True)


# StatsBomb Open Data queda como extractor opcional para MEC-2/MEC-1 enriquecido.
STATSBOMB_RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def fetch_statsbomb_competitions(raw_dir: Path, force: bool = False) -> Path:
    output = raw_dir / "competitions.json.gz"
    if output.exists() and not force:
        return output
    r = requests.get(f"{STATSBOMB_RAW_BASE}/competitions.json", timeout=60)
    r.raise_for_status()
    return save_json_gzip(r.json(), output)


def normalize_statsbomb_competitions(raw_file: Path) -> pd.DataFrame:
    return pd.json_normalize(read_json_gzip(raw_file))


def filter_finished_matches(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df = df[df["status"].eq("FINISHED")]
    df = df.dropna(subset=["utc_date", "home_team_id", "away_team_id", "ft_home_goals", "ft_away_goals"])
    df = df.dropna(subset=["ht_home_goals", "ht_away_goals"])
    for col in ["ft_home_goals", "ft_away_goals", "ht_home_goals", "ht_away_goals"]:
        df[col] = df[col].astype(int)
    df["sh_home_goals"] = df["ft_home_goals"] - df["ht_home_goals"]
    df["sh_away_goals"] = df["ft_away_goals"] - df["ht_away_goals"]
    return df.sort_values(["utc_date", "match_id"]).reset_index(drop=True)


def build_team_match_long(matches_finished: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in matches_finished.iterrows():
        base = {
            "match_id": m["match_id"],
            "competition_code": m["competition_code"],
            "competition_name": m["competition_name"],
            "season_start_year": m["season_start_year"],
            "utc_date": m["utc_date"],
            "matchday": m["matchday"],
        }
        sides = [
            (1, m["home_team_id"], m["home_team_name"], m["away_team_id"], m["away_team_name"],
             m["ft_home_goals"], m["ft_away_goals"], m["ht_home_goals"], m["ht_away_goals"], m["sh_home_goals"], m["sh_away_goals"]),
            (0, m["away_team_id"], m["away_team_name"], m["home_team_id"], m["home_team_name"],
             m["ft_away_goals"], m["ft_home_goals"], m["ht_away_goals"], m["ht_home_goals"], m["sh_away_goals"], m["sh_home_goals"]),
        ]
        for is_home, team_id, team_name, opp_id, opp_name, gf, ga, ht_gf, ht_ga, sh_gf, sh_ga in sides:
            points = 3 if gf > ga else 1 if gf == ga else 0
            rows.append({
                **base,
                "team_id": team_id,
                "team_name": team_name,
                "opponent_id": opp_id,
                "opponent_name": opp_name,
                "is_home": is_home,
                "gf": gf,
                "ga": ga,
                "gd": gf - ga,
                "points": points,
                "win": int(points == 3),
                "draw": int(points == 1),
                "loss": int(points == 0),
                "clean_sheet": int(ga == 0),
                "failed_to_score": int(gf == 0),
                "btts": int(gf > 0 and ga > 0),
                "over25": int(gf + ga > 2.5),
                "ht_gf": ht_gf,
                "ht_ga": ht_ga,
                "ht_gd": ht_gf - ht_ga,
                "sh_gf": sh_gf,
                "sh_ga": sh_ga,
                "sh_gd": sh_gf - sh_ga,
            })
    return pd.DataFrame(rows).sort_values(["team_id", "utc_date", "match_id"]).reset_index(drop=True)


BASE_PERFORMANCE_COLS = [
    "gf", "ga", "gd", "points", "win", "draw", "loss", "clean_sheet",
    "failed_to_score", "btts", "over25", "ht_gf", "ht_ga", "ht_gd", "sh_gf", "sh_ga", "sh_gd"
]


def add_rolling_team_features(team_long: pd.DataFrame, windows: Tuple[int, ...] = (5, 10), min_periods: int = 3) -> pd.DataFrame:
    df = team_long.copy().sort_values(["team_id", "utc_date", "match_id"]).reset_index(drop=True)
    df["prev_match_date"] = df.groupby("team_id")["utc_date"].shift(1)
    df["days_rest"] = ((df["utc_date"] - df["prev_match_date"]).dt.total_seconds() / 86400.0).clip(0, 30)
    for w in windows:
        for col in BASE_PERFORMANCE_COLS:
            df[f"roll{w}_{col}"] = df.groupby("team_id", group_keys=False)[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean()
            )
            df[f"roll{w}_ha_{col}"] = df.groupby(["team_id", "is_home"], group_keys=False)[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(2, min_periods - 1)).mean()
            )
    return df


def build_model_dataset(matches_finished: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    match_cols = [
        "match_id", "competition_code", "competition_name", "season_start_year", "utc_date", "matchday",
        "home_team_id", "home_team_name", "away_team_id", "away_team_name", "ft_home_goals", "ft_away_goals",
        "ht_home_goals", "ht_away_goals", "sh_home_goals", "sh_away_goals"
    ]
    base = matches_finished[match_cols].copy()
    feature_cols = [c for c in team_features.columns if c.startswith("roll") or c == "days_rest"]
    home_features = team_features[team_features["is_home"].eq(1)][["match_id", "team_id"] + feature_cols].rename(
        columns={c: f"home_{c}" for c in feature_cols}
    ).rename(columns={"team_id": "home_team_id_check"})
    away_features = team_features[team_features["is_home"].eq(0)][["match_id", "team_id"] + feature_cols].rename(
        columns={c: f"away_{c}" for c in feature_cols}
    ).rename(columns={"team_id": "away_team_id_check"})
    df = base.merge(home_features, on="match_id", how="left").merge(away_features, on="match_id", how="left")
    for c in feature_cols:
        df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]
    df["home_advantage"] = 1
    df["total_goals"] = df["ft_home_goals"] + df["ft_away_goals"]
    df["actual_result"] = np.select(
        [df["ft_home_goals"] > df["ft_away_goals"], df["ft_home_goals"] == df["ft_away_goals"]],
        ["H", "D"], default="A"
    )
    return df.sort_values(["utc_date", "match_id"]).reset_index(drop=True)


def get_feature_columns(model_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    exclude = {
        "match_id", "competition_name", "utc_date", "home_team_id", "home_team_name", "away_team_id", "away_team_name",
        "ft_home_goals", "ft_away_goals", "ht_home_goals", "ht_away_goals", "sh_home_goals", "sh_away_goals",
        "total_goals", "actual_result"
    }
    categorical_cols = ["competition_code"]
    numeric_cols = [c for c in model_df.columns if c not in exclude and c not in categorical_cols and pd.api.types.is_numeric_dtype(model_df[c])]
    return numeric_cols + categorical_cols, numeric_cols, categorical_cols


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.20, min_train_matches: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["utc_date", "match_id"]).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    if split_idx < min_train_matches and len(df) >= min_train_matches + 50:
        split_idx = min_train_matches
    train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    if train.empty or test.empty:
        raise ValueError("Split inválido: train o test quedó vacío. Descarga más temporadas/ligas.")
    return train, test


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)])
    return ColumnTransformer([("num", numeric_pipeline, numeric_cols), ("cat", categorical_pipeline, categorical_cols)], remainder="drop")


def build_poisson_model(numeric_cols: List[str], categorical_cols: List[str], alpha: float = 0.01) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(numeric_cols, categorical_cols)), ("model", PoissonRegressor(alpha=alpha, max_iter=2000))])


TARGETS = ["ht_home_goals", "ht_away_goals", "sh_home_goals", "sh_away_goals"]


def fit_mec1_models(train_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], alpha: float = 0.01) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}
    X = train_df[numeric_cols + categorical_cols].copy()
    for target in TARGETS:
        model = build_poisson_model(numeric_cols, categorical_cols, alpha=alpha)
        model.fit(X, train_df[target].astype(float))
        models[target] = model
    return models


def predict_lambdas(models: Dict[str, Pipeline], df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    pred = df.copy()
    X = pred[numeric_cols + categorical_cols].copy()
    for target, model in models.items():
        pred[f"lambda_{target}"] = np.clip(model.predict(X), 0.01, 6.0)
    pred["lambda_ft_home_goals"] = pred["lambda_ht_home_goals"] + pred["lambda_sh_home_goals"]
    pred["lambda_ft_away_goals"] = pred["lambda_ht_away_goals"] + pred["lambda_sh_away_goals"]
    pred["lambda_total_goals"] = pred["lambda_ft_home_goals"] + pred["lambda_ft_away_goals"]
    return pred


def poisson_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 8) -> pd.DataFrame:
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    matrix = np.outer(home_probs, away_probs)
    matrix = matrix / matrix.sum()
    rows = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            rows.append({
                "home_goals": h,
                "away_goals": a,
                "scoreline": f"{h}-{a}",
                "probability": float(matrix[h, a]),
                "result": "H" if h > a else "D" if h == a else "A",
                "total_goals": h + a,
                "btts": int(h > 0 and a > 0),
            })
    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def summarize_score_matrix(matrix: pd.DataFrame) -> Dict[str, float | str]:
    top3 = matrix.head(3)
    return {
        "prob_home_win": float(matrix.loc[matrix["result"].eq("H"), "probability"].sum()),
        "prob_draw": float(matrix.loc[matrix["result"].eq("D"), "probability"].sum()),
        "prob_away_win": float(matrix.loc[matrix["result"].eq("A"), "probability"].sum()),
        "prob_over_05": float(matrix.loc[matrix["total_goals"] > 0.5, "probability"].sum()),
        "prob_over_15": float(matrix.loc[matrix["total_goals"] > 1.5, "probability"].sum()),
        "prob_over_25": float(matrix.loc[matrix["total_goals"] > 2.5, "probability"].sum()),
        "prob_over_35": float(matrix.loc[matrix["total_goals"] > 3.5, "probability"].sum()),
        "prob_btts": float(matrix.loc[matrix["btts"].eq(1), "probability"].sum()),
        "top_scoreline": top3.iloc[0]["scoreline"],
        "top_scoreline_prob": float(top3.iloc[0]["probability"]),
        "top3_scorelines": " | ".join(top3["scoreline"].tolist()),
        "top3_scoreline_probs": " | ".join([f"{p:.3f}" for p in top3["probability"].tolist()]),
    }


def add_probabilistic_outputs(pred_df: pd.DataFrame, max_goals: int = 8) -> pd.DataFrame:
    rows = []
    for _, r in pred_df.iterrows():
        matrix_ft = poisson_score_matrix(r["lambda_ft_home_goals"], r["lambda_ft_away_goals"], max_goals=max_goals)
        matrix_ht = poisson_score_matrix(r["lambda_ht_home_goals"], r["lambda_ht_away_goals"], max_goals=max_goals)
        ft = summarize_score_matrix(matrix_ft)
        ht = summarize_score_matrix(matrix_ht)
        rows.append({"match_id": r["match_id"], **{f"ft_{k}": v for k, v in ft.items()}, **{f"ht_{k}": v for k, v in ht.items()}})
    out = pred_df.merge(pd.DataFrame(rows), on="match_id", how="left")
    out["predicted_result"] = out[["ft_prob_home_win", "ft_prob_draw", "ft_prob_away_win"]].idxmax(axis=1).map({
        "ft_prob_home_win": "H", "ft_prob_draw": "D", "ft_prob_away_win": "A"
    })
    return out


def evaluate_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    prob_actual = np.select(
        [df["actual_result"].eq("H"), df["actual_result"].eq("D"), df["actual_result"].eq("A")],
        [df["ft_prob_home_win"], df["ft_prob_draw"], df["ft_prob_away_win"]], default=np.nan
    )
    eps = 1e-12
    df["actual_scoreline"] = df["ft_home_goals"].astype(str) + "-" + df["ft_away_goals"].astype(str)
    top3_hit = np.mean([actual in str(top3).split(" | ") for actual, top3 in zip(df["actual_scoreline"], df["ft_top3_scorelines"])])
    return pd.DataFrame([{
        "n_matches_test": len(df),
        "log_loss_1x2": -np.nanmean(np.log(np.clip(prob_actual, eps, 1))),
        "brier_score_1x2": np.mean(
            (df["ft_prob_home_win"] - df["actual_result"].eq("H").astype(int)) ** 2 +
            (df["ft_prob_draw"] - df["actual_result"].eq("D").astype(int)) ** 2 +
            (df["ft_prob_away_win"] - df["actual_result"].eq("A").astype(int)) ** 2
        ) / 3,
        "accuracy_1x2": np.mean(df["actual_result"].eq(df["predicted_result"])),
        "mae_ft_home_goals": np.mean(np.abs(df["lambda_ft_home_goals"] - df["ft_home_goals"])),
        "mae_ft_away_goals": np.mean(np.abs(df["lambda_ft_away_goals"] - df["ft_away_goals"])),
        "exact_scoreline_top1_rate": np.mean(df["actual_scoreline"].eq(df["ft_top_scoreline"])),
        "exact_scoreline_top3_rate": top3_hit,
    }])


def evaluate_by_competition(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp, g in pred_df.groupby("competition_code", dropna=False):
        metrics = evaluate_predictions(g)
        metrics.insert(0, "competition_code", comp)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def snapshot_team_features(team_long: pd.DataFrame, team_id: int, as_of_date: pd.Timestamp, is_home: int, windows: Tuple[int, ...] = (5, 10)) -> Dict[str, float]:
    hist = team_long[(team_long["team_id"].eq(team_id)) & (team_long["utc_date"] < as_of_date)].sort_values("utc_date")
    hist_ha = hist[hist["is_home"].eq(is_home)]
    out = {"days_rest": np.nan if hist.empty else min(max((as_of_date - hist["utc_date"].max()).total_seconds() / 86400.0, 0), 30)}
    for w in windows:
        tail = hist.tail(w)
        tail_ha = hist_ha.tail(w)
        for col in BASE_PERFORMANCE_COLS:
            out[f"roll{w}_{col}"] = tail[col].mean() if len(tail) else np.nan
            out[f"roll{w}_ha_{col}"] = tail_ha[col].mean() if len(tail_ha) else np.nan
    return out


def build_fixture_prediction_dataset(fixtures: pd.DataFrame, team_long_finished: pd.DataFrame, windows: Tuple[int, ...] = (5, 10)) -> pd.DataFrame:
    rows = []
    for _, m in fixtures.iterrows():
        home = snapshot_team_features(team_long_finished, m["home_team_id"], m["utc_date"], 1, windows)
        away = snapshot_team_features(team_long_finished, m["away_team_id"], m["utc_date"], 0, windows)
        row = {
            "match_id": m["match_id"], "competition_code": m["competition_code"], "competition_name": m.get("competition_name"),
            "season_start_year": m.get("season_start_year"), "utc_date": m["utc_date"], "matchday": m.get("matchday"),
            "home_team_id": m["home_team_id"], "home_team_name": m["home_team_name"],
            "away_team_id": m["away_team_id"], "away_team_name": m["away_team_name"], "home_advantage": 1,
        }
        for k, v in home.items():
            row[f"home_{k}"] = v
        for k, v in away.items():
            row[f"away_{k}"] = v
        for k in home.keys():
            row[f"diff_{k}"] = home.get(k, np.nan) - away.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_scoreline_matrix_for_match(pred_row: pd.Series, half: str = "ft", max_goals: int = 8) -> pd.DataFrame:
    if half == "ft":
        lh, la = pred_row["lambda_ft_home_goals"], pred_row["lambda_ft_away_goals"]
    elif half == "ht":
        lh, la = pred_row["lambda_ht_home_goals"], pred_row["lambda_ht_away_goals"]
    else:
        raise ValueError("half debe ser 'ft' o 'ht'.")
    matrix = poisson_score_matrix(lh, la, max_goals=max_goals)
    matrix.insert(0, "half", half)
    matrix.insert(0, "away_team_name", pred_row.get("away_team_name"))
    matrix.insert(0, "home_team_name", pred_row.get("home_team_name"))
    matrix.insert(0, "match_id", pred_row["match_id"])
    return matrix


def export_mec1_excel(output_path: Path, predictions: pd.DataFrame, metrics_global: pd.DataFrame, metrics_by_comp: pd.DataFrame, model_dataset: Optional[pd.DataFrame] = None, max_matrix_matches: int = 5) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_cols_priority = [
        "match_id", "utc_date", "competition_code", "home_team_name", "away_team_name", "ft_home_goals", "ft_away_goals", "actual_result",
        "lambda_ft_home_goals", "lambda_ft_away_goals", "lambda_total_goals", "lambda_ht_home_goals", "lambda_ht_away_goals",
        "lambda_sh_home_goals", "lambda_sh_away_goals", "ft_top_scoreline", "ft_top_scoreline_prob", "ft_top3_scorelines",
        "ht_top_scoreline", "ht_top_scoreline_prob", "ht_top3_scorelines", "ft_prob_home_win", "ft_prob_draw", "ft_prob_away_win",
        "ft_prob_over_15", "ft_prob_over_25", "ft_prob_over_35", "ft_prob_btts", "predicted_result",
    ]
    pred_export = predictions[[c for c in pred_cols_priority if c in predictions.columns]].copy()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pred_export.to_excel(writer, sheet_name="Resumen_Predicciones", index=False)
        metrics_global.to_excel(writer, sheet_name="Backtesting_Global", index=False)
        metrics_by_comp.to_excel(writer, sheet_name="Backtesting_Liga", index=False)
        matrices = []
        for _, row in predictions.head(max_matrix_matches).iterrows():
            matrices.append(build_scoreline_matrix_for_match(row, half="ft"))
            matrices.append(build_scoreline_matrix_for_match(row, half="ht"))
        if matrices:
            pd.concat(matrices, ignore_index=True).to_excel(writer, sheet_name="Matriz_Marcadores", index=False)
        if model_dataset is not None:
            sample_cols = [
                c for c in model_dataset.columns
                if c in ["match_id", "utc_date", "competition_code", "home_team_name", "away_team_name", "ft_home_goals", "ft_away_goals", "ht_home_goals", "ht_away_goals", "sh_home_goals", "sh_away_goals", "actual_result"]
                or c.startswith("diff_roll5") or c.startswith("diff_roll10")
            ]
            model_dataset[sample_cols].head(5000).to_excel(writer, sheet_name="Features_Muestra", index=False)
    return output_path


def run_mec1_pipeline(
    project_root: Optional[Path] = None,
    competitions: Iterable[str] = DEFAULT_COMPETITIONS,
    seasons: Iterable[int] = (2021, 2022, 2023, 2024, 2025),
    test_size: float = 0.20,
    force_extract: bool = False,
    api_token: Optional[str] = None,
) -> Dict[str, object]:
    project_root = project_root or resolve_project_root()
    paths = ensure_project_dirs(project_root)
    print("PROJECT_ROOT:", project_root)

    client = FootballDataClient(api_token=api_token)
    raw_files = fetch_competition_matches(client, competitions, seasons, paths["raw_football_data"], force=force_extract)
    matches = normalize_football_data_matches(raw_files)
    save_parquet_gzip(matches, parquet_gzip_path(paths["bronze"], "matches_football_data"))
    print(f"[BRONZE] matches: {matches.shape}")

    matches_finished = filter_finished_matches(matches)
    team_long = build_team_match_long(matches_finished)
    team_features = add_rolling_team_features(team_long)
    save_parquet_gzip(matches_finished, parquet_gzip_path(paths["silver"], "matches_finished"))
    save_parquet_gzip(team_long, parquet_gzip_path(paths["silver"], "team_match_long"))
    save_parquet_gzip(team_features, parquet_gzip_path(paths["silver"], "team_features_rolling"))
    print(f"[SILVER] matches_finished: {matches_finished.shape}")
    print(f"[SILVER] team_features: {team_features.shape}")

    model_dataset = build_model_dataset(matches_finished, team_features)
    save_parquet_gzip(model_dataset, parquet_gzip_path(paths["gold"], "model_dataset_mec1"))
    _, numeric_cols, categorical_cols = get_feature_columns(model_dataset)
    train_df, test_df = temporal_train_test_split(model_dataset, test_size=test_size)
    print(f"[TRAIN] {train_df.shape} | [TEST] {test_df.shape}")
    print(f"[FEATURES] numeric={len(numeric_cols)} categorical={categorical_cols}")

    models = fit_mec1_models(train_df, numeric_cols, categorical_cols)
    pred_test = predict_lambdas(models, test_df, numeric_cols, categorical_cols)
    pred_test = add_probabilistic_outputs(pred_test)
    metrics_global = evaluate_predictions(pred_test)
    metrics_by_comp = evaluate_by_competition(pred_test)
    save_parquet_gzip(pred_test, parquet_gzip_path(paths["gold"], "predictions_test_mec1"))
    save_parquet_gzip(metrics_global, parquet_gzip_path(paths["gold"], "metrics_global_mec1"))
    save_parquet_gzip(metrics_by_comp, parquet_gzip_path(paths["gold"], "metrics_by_competition_mec1"))

    excel_path = export_mec1_excel(paths["outputs_excel"] / "predicciones_mec1.xlsx", pred_test, metrics_global, metrics_by_comp, model_dataset)
    print("[EXPORT]", excel_path)
    return {
        "project_root": project_root,
        "paths": paths,
        "matches": matches,
        "matches_finished": matches_finished,
        "team_long": team_long,
        "team_features": team_features,
        "model_dataset": model_dataset,
        "train_df": train_df,
        "test_df": test_df,
        "models": models,
        "pred_test": pred_test,
        "metrics_global": metrics_global,
        "metrics_by_comp": metrics_by_comp,
        "excel_path": excel_path,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
