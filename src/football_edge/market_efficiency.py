"""Paper-style football betting market-efficiency utilities.

The functions in this module support the Angelini & De Angelis-style
forecast-error experiment used in the public notebook.  They intentionally
avoid notebook state: configuration is passed in explicitly, and the
walk-forward functions use only information available before each test season.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_edge.backtest import build_research_dataset
from football_edge.config import RAW_DATA_DIR
from football_edge.data import _standardize_football_data_columns, discover_datasets


ALL_EURO_DIVISIONS = [
    "B1",   # Belgian Pro League
    "D1",   # Bundesliga
    "D2",   # Bundesliga 2
    "E0",   # English Premier League
    "E1",   # Championship
    "E2",   # League One
    "E3",   # League Two
    "EC",   # Conference
    "F1",   # Championnat / Ligue 1
    "F2",   # Ligue 2
    "G1",   # Greek Super League
    "I1",   # Serie A
    "I2",   # Serie B
    "N1",   # Eredivisie
    "P1",   # Primeira Liga
    "SC0",  # Scottish Premiership
    "SC1",  # Scottish Championship
    "SC2",  # Scottish League One
    "SC3",  # Scottish League Two
    "SP1",  # La Liga
    "SP2",  # Segunda Division
    "T1",   # Turkish Super Lig
]

PAPER_TOP_DIVISIONS = [
    "B1",   # Belgian Pro League
    "D1",   # Bundesliga
    "E0",   # English Premier League
    "F1",   # Championnat / Ligue 1
    "G1",   # Greek Super League
    "I1",   # Serie A
    "N1",   # Eredivisie
    "P1",   # Primeira Liga
    "SC0",  # Scottish Premiership
    "SP1",  # La Liga
    "T1",   # Turkish Super Lig
]

MARKET_OUTCOME_CONFIG = {
    "over_2_5": {
        "label": "Over 2.5 goals",
        "odds_key": "over_25",
        "outcome_function": lambda frame: frame["total_ft_goals"].gt(2.5).astype(int),
    },
    "under_2_5": {
        "label": "Under 2.5 goals",
        "odds_key": "under_25",
        "outcome_function": lambda frame: frame["total_ft_goals"].lt(2.5).astype(int),
    },
    "home_win": {
        "label": "Home win",
        "odds_key": "home",
        "outcome_function": lambda frame: frame["FTR"].eq("H").astype(int),
    },
    "draw": {
        "label": "Draw",
        "odds_key": "draw",
        "outcome_function": lambda frame: frame["FTR"].eq("D").astype(int),
    },
    "away_win": {
        "label": "Away win",
        "odds_key": "away",
        "outcome_function": lambda frame: frame["FTR"].eq("A").astype(int),
    },
}

MARKET_ODDS_SOURCES = {
    "average_market": {
        "odds": {"over_25": "Avg>2.5", "under_25": "Avg<2.5", "home": "AvgH", "draw": "AvgD", "away": "AvgA"},
        "margin_groups": {"ou_25": ["Avg>2.5", "Avg<2.5"], "1x2": ["AvgH", "AvgD", "AvgA"]},
        "kind": "market_aggregate",
    },
    "market_maximum": {
        "odds": {"over_25": "Max>2.5", "under_25": "Max<2.5", "home": "MaxH", "draw": "MaxD", "away": "MaxA"},
        "margin_groups": {"ou_25": ["Max>2.5", "Max<2.5"], "1x2": ["MaxH", "MaxD", "MaxA"]},
        "kind": "market_aggregate",
    },
}

MARKET_PRIMARY_SOURCES = ["average_market", "market_maximum"]

MARKET_SOURCE_COLORS = {
    "average_market": "#4C78A8",
    "market_maximum": "#F58518",
}

SUPPORTED_ESTIMATION_METHODS = {"ols_hc1", "bernoulli_wls"}


def normalize_season_key(season: str) -> str:
    """Normalize season labels such as ``2006/07`` or ``06_07``."""
    parts = re.findall(r"\d+", str(season))
    if len(parts) >= 2:
        return f"{parts[0][-2:]}_{parts[1][-2:]}"
    return str(season).replace("/", "_").replace("-", "_")


def season_to_parquet_name(season: str) -> str:
    """Return the expected all-Europe Parquet file name for a season key."""
    normalized = normalize_season_key(season)
    start, end = normalized.split("_")
    start_year = 2000 + int(start) if int(start) <= 50 else 1900 + int(start)
    end_year = 2000 + int(end) if int(end) <= 50 else 1900 + int(end)
    return f"all_euro_{start_year}_{end_year}.parquet"


def discover_parquet_seasons(parquet_dir: Path) -> list[str]:
    """Discover available all-Europe season Parquet files."""
    seasons = []
    for path in sorted(parquet_dir.glob("all_euro_*.parquet")):
        match = re.fullmatch(r"all_euro_(\d{4})_(\d{4})", path.stem)
        if match is None:
            continue
        start_year, end_year = match.groups()
        seasons.append(f"{start_year[-2:]}_{end_year[-2:]}")
    return sorted(seasons)


def prepare_matches_from_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw all-Europe Parquet rows to the repo match schema."""
    frame = _standardize_football_data_columns(frame).copy()
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Parquet data is missing required column(s): {missing}")

    frame = frame.dropna(subset=required).copy()
    frame["date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="raise")
    frame["FTHG"] = pd.to_numeric(frame["FTHG"], errors="raise")
    frame["FTAG"] = pd.to_numeric(frame["FTAG"], errors="raise")

    frame = frame.rename(
        columns={
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_ft_goals",
            "FTAG": "away_ft_goals",
        }
    )
    frame["total_ft_goals"] = frame["home_ft_goals"] + frame["away_ft_goals"]
    frame["over_2_5"] = frame["total_ft_goals"].gt(2.5).astype(int)
    return frame


def load_matches_from_season_parquets(
    parquet_dir: Path,
    *,
    seasons: list[str] | None,
    divisions: list[str] | None,
    leagues: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Load selected all-Europe season Parquet files."""
    available_seasons = discover_parquet_seasons(parquet_dir)
    selected_seasons = (
        available_seasons
        if seasons is None
        else [normalize_season_key(season) for season in seasons]
    )
    missing_files = [
        season
        for season in selected_seasons
        if not (parquet_dir / season_to_parquet_name(season)).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing selected season Parquet file(s): "
            + ", ".join(missing_files)
        )

    frames = [
        pd.read_parquet(parquet_dir / season_to_parquet_name(season))
        for season in selected_seasons
    ]
    raw = pd.concat(frames, ignore_index=True, sort=False)

    available_divisions = sorted(raw["division"].dropna().astype(str).unique())
    if divisions is None:
        selected_divisions = available_divisions
        filtered = raw.copy()
    else:
        selected_divisions = list(divisions)
        unknown_divisions = sorted(set(selected_divisions).difference(available_divisions))
        if unknown_divisions:
            raise ValueError(
                f"Unknown division code(s): {unknown_divisions}. "
                f"Available division codes: {available_divisions}"
            )
        filtered = raw.loc[raw["division"].astype(str).isin(selected_divisions)].copy()

    available_leagues = sorted(filtered["league"].dropna().astype(str).unique())
    if leagues is None:
        selected_leagues = available_leagues
    else:
        selected_leagues = list(leagues)
        unknown_leagues = sorted(set(selected_leagues).difference(available_leagues))
        if unknown_leagues:
            raise ValueError(
                f"Unknown league(s): {unknown_leagues}. "
                f"Available leagues after division filter: {available_leagues}"
            )
    filtered = filtered.loc[filtered["league"].astype(str).isin(selected_leagues)].copy()

    matches = prepare_matches_from_parquet(filtered)
    matches = matches.sort_values(
        ["league", "season", "date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    return matches, available_seasons, selected_divisions, selected_leagues


def load_matches_from_excel_fallback(
    *,
    seasons: list[str] | None,
    divisions: list[str] | None,
    leagues: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], int]:
    """Fallback loader for raw Excel/CSV files using existing repo utilities."""
    all_datasets = discover_datasets(RAW_DATA_DIR, seasons=seasons)
    available_seasons = sorted({dataset.season for dataset in all_datasets})
    available_divisions = sorted(
        {dataset.division for dataset in all_datasets if dataset.division is not None}
    )

    if divisions is None:
        division_filtered = all_datasets
        selected_divisions = available_divisions
    else:
        selected_divisions = list(divisions)
        unknown_divisions = sorted(set(selected_divisions).difference(available_divisions))
        if unknown_divisions:
            raise ValueError(
                f"Unknown division code(s): {unknown_divisions}. "
                f"Available division codes: {available_divisions}"
            )
        division_filtered = [
            dataset for dataset in all_datasets if dataset.division in selected_divisions
        ]

    available_leagues = sorted({dataset.league for dataset in division_filtered})
    if leagues is None:
        selected_leagues = available_leagues
    else:
        selected_leagues = list(leagues)
        unknown_leagues = sorted(set(selected_leagues).difference(available_leagues))
        if unknown_leagues:
            raise ValueError(
                f"Unknown league(s): {unknown_leagues}. "
                f"Available leagues after division filter: {available_leagues}"
            )

    datasets = [dataset for dataset in division_filtered if dataset.league in selected_leagues]
    matches = build_research_dataset(datasets)
    matches = matches.sort_values(
        ["league", "season", "date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)
    return matches, available_seasons, selected_divisions, selected_leagues, len(datasets)


def load_market_efficiency_matches(
    *,
    use_season_parquet: bool,
    season_parquet_dir: Path,
    seasons: list[str] | None,
    divisions: list[str] | None,
    leagues: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], int, str]:
    """Load matches for the market-efficiency notebook."""
    try:
        if use_season_parquet:
            matches, available_seasons, selected_divisions, selected_leagues = (
                load_matches_from_season_parquets(
                    season_parquet_dir,
                    seasons=seasons,
                    divisions=divisions,
                    leagues=leagues,
                )
            )
            dataset_count = matches[["season", "division"]].drop_duplicates().shape[0]
            return (
                matches,
                available_seasons,
                selected_divisions,
                selected_leagues,
                dataset_count,
                "season Parquet",
            )
        raise FileNotFoundError("USE_SEASON_PARQUET is False")
    except FileNotFoundError:
        matches, available_seasons, selected_divisions, selected_leagues, dataset_count = (
            load_matches_from_excel_fallback(
                seasons=seasons,
                divisions=divisions,
                leagues=leagues,
            )
        )
        return (
            matches,
            available_seasons,
            selected_divisions,
            selected_leagues,
            dataset_count,
            "raw Excel fallback",
        )


def build_source_panel(
    matches: pd.DataFrame,
    source_specs: dict[str, dict[str, Any]],
    *,
    market_outcome: str,
    outcome_config: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Construct source-level odds, implied probability, and profit rows."""
    outcome_spec = outcome_config[market_outcome]
    odds_key = outcome_spec["odds_key"]
    required_for_outcome = ["FTR"] if market_outcome in {"home_win", "draw", "away_win"} else []
    missing_outcome_columns = [
        column for column in required_for_outcome if column not in matches.columns
    ]
    if missing_outcome_columns:
        raise ValueError(
            f"MARKET_OUTCOME={market_outcome!r} requires missing column(s): "
            f"{missing_outcome_columns}"
        )

    frames = []
    base_columns = ["league", "season", "date", "home_team", "away_team", "total_ft_goals", "FTR"]
    base_columns = [column for column in base_columns if column in matches.columns]
    outcome = outcome_spec["outcome_function"](matches)

    for source, spec in source_specs.items():
        odds_col = spec["odds"].get(odds_key)
        if odds_col is None or odds_col not in matches.columns:
            continue

        margin_group_name = "ou_25" if odds_key in {"over_25", "under_25"} else "1x2"
        margin_columns = [
            column
            for column in spec["margin_groups"][margin_group_name]
            if column in matches.columns
        ]
        if odds_col not in margin_columns:
            margin_columns = [odds_col, *margin_columns]

        frame = matches[base_columns + list(dict.fromkeys([odds_col, *margin_columns]))].copy()
        frame["source"] = source
        frame["source_kind"] = spec["kind"]
        frame["market_outcome"] = market_outcome
        frame["outcome_label"] = outcome_spec["label"]
        frame["outcome"] = outcome.reindex(frame.index).astype(int)
        frame["outcome_odds"] = pd.to_numeric(frame[odds_col], errors="coerce")

        for column in margin_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        valid = frame["outcome_odds"].notna() & frame["outcome_odds"].gt(1)
        valid &= frame[margin_columns].notna().all(axis=1)
        valid &= frame[margin_columns].gt(1).all(axis=1)
        frame = frame.loc[valid].copy()

        margin_implied = sum(1.0 / frame[column] for column in margin_columns)
        frame["raw_implied_probability"] = 1.0 / frame["outcome_odds"]
        frame["bookmaker_margin"] = margin_implied - 1.0
        frame["forecast_error_raw"] = frame["outcome"] - frame["raw_implied_probability"]
        frame["flat_profit"] = np.where(frame["outcome"].eq(1), frame["outcome_odds"] - 1.0, -1.0)
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"No valid odds-source panels could be constructed for MARKET_OUTCOME={market_outcome!r}."
        )
    return pd.concat(frames, ignore_index=True)


def select_training_seasons(
    prior_seasons: list[str],
    *,
    training_window_seasons: int | None,
) -> list[str]:
    """Select expanding or rolling prior seasons for walk-forward calibration."""
    if training_window_seasons is None:
        return list(prior_seasons)
    if training_window_seasons <= 0:
        raise ValueError("training_window_seasons must be None or a positive integer.")
    return list(prior_seasons[-training_window_seasons:])


def fit_error_model_with_covariance(
    train: pd.DataFrame,
    *,
    min_train_matches: int,
    estimation_method: str,
) -> dict[str, object] | None:
    """Fit the paper-style forecast-error curve and covariance matrix."""
    data = train[["forecast_error_raw", "raw_implied_probability"]].dropna().copy()
    if len(data) < min_train_matches or data["raw_implied_probability"].nunique() < 5:
        return None

    p = data["raw_implied_probability"].to_numpy(float)
    x = np.column_stack([np.ones(len(data)), p])
    y = data["forecast_error_raw"].to_numpy(float)
    n, k = x.shape

    if estimation_method == "ols_hc1":
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coefficients
        xtx_inv = np.linalg.pinv(x.T @ x)
        meat = x.T @ ((residual[:, None] ** 2) * x)
        hc1_scale = n / max(n - k, 1)
        covariance = hc1_scale * xtx_inv @ meat @ xtx_inv
    elif estimation_method == "bernoulli_wls":
        bernoulli_variance = np.clip(p * (1.0 - p), 1e-8, None)
        weights = 1.0 / bernoulli_variance
        weighted_x = x * weights[:, None]
        xtwx_inv = np.linalg.pinv(x.T @ weighted_x)
        coefficients = xtwx_inv @ (weighted_x.T @ y)
        covariance = xtwx_inv
    else:
        raise ValueError(f"Unsupported estimation_method: {estimation_method}")

    return {
        "coefficients": coefficients,
        "robust_cov": covariance,
        "observations": n,
        "estimation_method": estimation_method,
    }


def run_walk_forward_paper_rule(
    panel: pd.DataFrame,
    *,
    training_window_seasons: int | None,
    min_train_seasons: int,
    min_train_matches: int,
    confidence_z: float,
    estimation_method: str,
) -> pd.DataFrame:
    """Apply the paper-style lower-confidence-bound rule walk-forward."""
    rows = []
    for (source, league), group in panel.groupby(["source", "league"], sort=True):
        group = group.sort_values(["season", "date", "home_team", "away_team"], kind="stable").copy()
        seasons = sorted(group["season"].dropna().unique())

        for season_index, test_season in enumerate(seasons):
            prior_seasons = seasons[:season_index]
            training_seasons = select_training_seasons(
                prior_seasons,
                training_window_seasons=training_window_seasons,
            )
            if len(training_seasons) < min_train_seasons:
                continue

            train = group.loc[group["season"].isin(training_seasons)]
            test = group.loc[group["season"].eq(test_season)].copy()

            if not train.empty and not test.empty and train["date"].max() >= test["date"].min():
                raise ValueError(
                    f"Walk-forward date overlap for {source} / {league} / {test_season}: "
                    f"training ends {train['date'].max()}, test starts {test['date'].min()}"
                )

            model = fit_error_model_with_covariance(
                train,
                min_train_matches=min_train_matches,
                estimation_method=estimation_method,
            )
            if model is None or test.empty:
                continue

            coefficients = model["coefficients"]
            robust_cov = model["robust_cov"]
            x_test = np.column_stack(
                [np.ones(len(test)), test["raw_implied_probability"].to_numpy(float)]
            )

            predicted_error = x_test @ coefficients
            prediction_variance = np.einsum("ij,jk,ik->i", x_test, robust_cov, x_test)
            prediction_se = np.sqrt(np.maximum(prediction_variance, 0.0))
            lower_95_error = predicted_error - confidence_z * prediction_se

            parameter_se = np.sqrt(np.maximum(np.diag(robust_cov), 0.0))
            parameter_t = np.divide(
                coefficients,
                parameter_se,
                out=np.full_like(coefficients, np.nan, dtype=float),
                where=parameter_se > 0,
            )

            test["corrected_probability"] = (
                test["raw_implied_probability"] + predicted_error
            ).clip(1e-6, 1 - 1e-6)
            test["predicted_error"] = predicted_error
            test["predicted_error_se"] = prediction_se
            test["predicted_error_lower_95"] = lower_95_error
            test["expected_value"] = test["corrected_probability"] * test["outcome_odds"] - 1.0
            test["paper_rule_selected"] = test["predicted_error_lower_95"].gt(0)
            test["model_alpha"] = float(coefficients[0])
            test["model_beta"] = float(coefficients[1])
            test["model_alpha_se"] = float(parameter_se[0])
            test["model_beta_se"] = float(parameter_se[1])
            test["model_alpha_t"] = float(parameter_t[0])
            test["model_beta_t"] = float(parameter_t[1])
            test["training_observations"] = int(model["observations"])
            test["available_prior_seasons"] = ",".join(prior_seasons)
            test["training_seasons"] = ",".join(training_seasons)
            test["training_season_count"] = len(training_seasons)
            test["training_window_setting"] = (
                "expanding"
                if training_window_seasons is None
                else f"last_{training_window_seasons}"
            )
            test["estimation_method"] = model["estimation_method"]
            rows.append(test)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_bets(bets: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Summarize flat-stake betting results."""
    columns = [
        *group_by,
        "bets",
        "total_stake",
        "profit",
        "roi_pct",
        "hit_rate_pct",
        "average_odds",
        "average_ev_pct",
        "market_outcome",
    ]
    if bets.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        bets.groupby(group_by, sort=True)
        .agg(
            bets=("outcome", "size"),
            total_stake=("outcome", "size"),
            profit=("profit", "sum"),
            hit_rate_pct=("outcome", lambda values: values.mean() * 100),
            average_odds=("outcome_odds", "mean"),
            average_ev_pct=("expected_value", lambda values: values.mean() * 100),
            market_outcome=("market_outcome", "first"),
        )
        .reset_index()
    )
    summary["roi_pct"] = np.where(
        summary["total_stake"].gt(0),
        summary["profit"] / summary["total_stake"] * 100,
        np.nan,
    )
    return summary[columns]


def build_paper_activity(paper_candidates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate walk-forward parameter and activity diagnostics."""
    if paper_candidates.empty:
        return pd.DataFrame()
    return (
        paper_candidates.groupby(["source", "league", "season"], sort=True)
        .agg(
            candidate_matches=("outcome", "size"),
            selected_bets=("paper_rule_selected", "sum"),
            average_lower_95_error=("predicted_error_lower_95", "mean"),
            model_alpha=("model_alpha", "first"),
            model_alpha_t=("model_alpha_t", "first"),
            model_beta=("model_beta", "first"),
            model_beta_t=("model_beta_t", "first"),
            training_observations=("training_observations", "first"),
            training_seasons=("training_seasons", "first"),
            training_season_count=("training_season_count", "first"),
            training_window_setting=("training_window_setting", "first"),
        )
        .reset_index()
    )


def run_paper_rule_analysis(
    market_panel: pd.DataFrame,
    *,
    training_window_seasons: int | None,
    min_train_seasons: int,
    min_train_matches: int,
    confidence_z: float,
    estimation_method: str,
    paper_rule_name: str,
) -> dict[str, pd.DataFrame]:
    """Run the complete paper-rule selection and flat-stake summaries."""
    paper_candidates = run_walk_forward_paper_rule(
        market_panel,
        training_window_seasons=training_window_seasons,
        min_train_seasons=min_train_seasons,
        min_train_matches=min_train_matches,
        confidence_z=confidence_z,
        estimation_method=estimation_method,
    )
    paper_bets = paper_candidates.loc[paper_candidates["paper_rule_selected"]].copy()
    if not paper_bets.empty:
        paper_bets["selection_rule"] = paper_rule_name
        paper_bets["profit"] = paper_bets["flat_profit"]

    return {
        "paper_candidates": paper_candidates,
        "paper_bets": paper_bets,
        "paper_summary": summarize_bets(paper_bets, ["source"]),
        "paper_season_summary": summarize_bets(paper_bets, ["source", "season"]),
        "paper_league_summary": summarize_bets(paper_bets, ["source", "league"]),
        "paper_season_league": summarize_bets(
            paper_bets, ["source", "season", "league"]
        ),
        "paper_activity": build_paper_activity(paper_candidates),
    }


def build_group_summary(bets: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize profit by arbitrary block columns."""
    columns = [
        *group_columns,
        "bets",
        "stake",
        "profit",
        "roi_pct",
        "hit_rate_pct",
        "average_odds",
    ]
    if bets.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        bets.groupby(group_columns, sort=True)
        .agg(
            bets=("outcome", "size"),
            stake=("outcome", "size"),
            profit=("profit", "sum"),
            hit_rate_pct=("outcome", lambda values: values.mean() * 100),
            average_odds=("outcome_odds", "mean"),
        )
        .reset_index()
    )
    summary["roi_pct"] = np.where(
        summary["stake"].gt(0),
        summary["profit"] / summary["stake"] * 100,
        np.nan,
    )
    return summary[columns]


def bootstrap_group_roi(
    group_summary: pd.DataFrame,
    *,
    group_label: str,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap ROI by resampling pre-aggregated blocks with replacement."""
    if group_summary.empty:
        return pd.DataFrame(columns=["group_label", "simulation", "roi_pct"])

    rng = np.random.default_rng(seed)
    n_groups = len(group_summary)
    profit = group_summary["profit"].to_numpy(float)
    stake = group_summary["stake"].to_numpy(float)
    sampled_indices = rng.integers(0, n_groups, size=(simulations, n_groups))
    sampled_profit = profit[sampled_indices].sum(axis=1)
    sampled_stake = stake[sampled_indices].sum(axis=1)
    roi = np.divide(
        sampled_profit,
        sampled_stake,
        out=np.full(simulations, np.nan),
        where=sampled_stake > 0,
    ) * 100
    return pd.DataFrame(
        {"group_label": group_label, "simulation": np.arange(simulations), "roi_pct": roi}
    )


def summarize_bootstrap(
    group_summary: pd.DataFrame,
    bootstrap_roi: pd.DataFrame,
    *,
    group_label: str,
) -> pd.DataFrame:
    """Summarize a block-bootstrap ROI distribution."""
    if group_summary.empty:
        return pd.DataFrame()

    observed_profit = group_summary["profit"].sum()
    observed_stake = group_summary["stake"].sum()
    observed_roi = observed_profit / observed_stake * 100 if observed_stake > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "group_label": group_label,
                "observed_blocks": len(group_summary),
                "observed_bets": group_summary["bets"].sum(),
                "observed_profit": observed_profit,
                "observed_stake": observed_stake,
                "observed_roi_pct": observed_roi,
                "bootstrap_median_roi_pct": bootstrap_roi["roi_pct"].median(),
                "bootstrap_p05_roi_pct": np.nanpercentile(bootstrap_roi["roi_pct"], 5),
                "bootstrap_p95_roi_pct": np.nanpercentile(bootstrap_roi["roi_pct"], 95),
                "probability_roi_positive_pct": np.nanmean(bootstrap_roi["roi_pct"] > 0)
                * 100,
            }
        ]
    )


def leave_one_group_out(group_summary: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    """Calculate total ROI after removing each block once."""
    if group_summary.empty:
        return pd.DataFrame()

    total_profit = group_summary["profit"].sum()
    total_stake = group_summary["stake"].sum()
    observed_roi = total_profit / total_stake * 100 if total_stake > 0 else np.nan
    rows = []
    for row in group_summary.itertuples(index=False):
        removed_label = " | ".join(str(getattr(row, column)) for column in label_columns)
        remaining_profit = total_profit - row.profit
        remaining_stake = total_stake - row.stake
        rows.append(
            {
                "removed_block": removed_label,
                "removed_bets": row.bets,
                "removed_profit": row.profit,
                "observed_roi_pct": observed_roi,
                "leave_one_out_roi_pct": remaining_profit / remaining_stake * 100
                if remaining_stake > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def profit_contribution(
    group_summary: pd.DataFrame,
    *,
    contribution_levels: list[float],
) -> pd.DataFrame:
    """Compute cumulative contribution from the largest winning blocks."""
    if group_summary.empty:
        return pd.DataFrame()

    sorted_blocks = group_summary.sort_values("profit", ascending=False).reset_index(drop=True)
    positive_profit = sorted_blocks.loc[sorted_blocks["profit"].gt(0), "profit"]
    total_profit = sorted_blocks["profit"].sum()
    total_positive_profit = positive_profit.sum()
    n_positive = len(positive_profit)
    rows = []
    for level in contribution_levels:
        n_take = int(np.ceil(n_positive * level)) if n_positive else 0
        top_positive_profit = positive_profit.iloc[:n_take].sum() if n_take else 0.0
        rows.append(
            {
                "top_winning_blocks_pct": level * 100,
                "winning_blocks_used": n_take,
                "total_winning_blocks": n_positive,
                "share_of_positive_profit_pct": top_positive_profit / total_positive_profit * 100
                if total_positive_profit > 0
                else np.nan,
                "share_of_total_profit_pct": top_positive_profit / total_profit * 100
                if total_profit != 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_bet_level_by_group(
    bets: pd.DataFrame,
    group_columns: list[str],
    *,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap individual bets inside each active group."""
    output_columns = [
        *group_columns,
        "bets",
        "observed_roi_pct",
        "bootstrap_median_roi_pct",
        "bootstrap_p05_roi_pct",
        "bootstrap_p95_roi_pct",
        "probability_roi_positive_pct",
    ]
    if bets.empty:
        return pd.DataFrame(columns=output_columns)

    rng = np.random.default_rng(seed)
    rows = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in bets.groupby(group_key, sort=True):
        profit = group["profit"].to_numpy(float)
        n_bets = len(profit)
        if n_bets == 0:
            continue
        sampled_indices = rng.integers(0, n_bets, size=(simulations, n_bets))
        sampled_profit = profit[sampled_indices].sum(axis=1)
        roi = sampled_profit / n_bets * 100
        observed_roi = profit.sum() / n_bets * 100

        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_values))
        row.update(
            {
                "bets": n_bets,
                "observed_roi_pct": observed_roi,
                "bootstrap_median_roi_pct": np.nanmedian(roi),
                "bootstrap_p05_roi_pct": np.nanpercentile(roi, 5),
                "bootstrap_p95_roi_pct": np.nanpercentile(roi, 95),
                "probability_roi_positive_pct": np.nanmean(roi > 0) * 100,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def build_robustness_diagnostics(
    paper_bets: pd.DataFrame,
    *,
    source: str,
    simulations: int,
    seed: int,
    contribution_levels: list[float],
) -> dict[str, pd.DataFrame]:
    """Build flat-stake robustness diagnostics for one source."""
    source_bets = paper_bets.loc[paper_bets["source"].eq(source)].copy()
    season_league_blocks = build_group_summary(source_bets, ["season", "league"])
    season_blocks = build_group_summary(source_bets, ["season"])
    season_league_bootstrap_roi = bootstrap_group_roi(
        season_league_blocks,
        group_label="season-league",
        simulations=simulations,
        seed=seed,
    )
    season_bootstrap_roi = bootstrap_group_roi(
        season_blocks,
        group_label="season",
        simulations=simulations,
        seed=seed + 1,
    )
    return {
        "market_maximum_bets": source_bets,
        "season_league_blocks": season_league_blocks,
        "season_blocks": season_blocks,
        "season_league_bootstrap_roi": season_league_bootstrap_roi,
        "season_bootstrap_roi": season_bootstrap_roi,
        "bootstrap_roi": pd.concat(
            [season_league_bootstrap_roi, season_bootstrap_roi],
            ignore_index=True,
        ),
        "season_league_bootstrap_summary": summarize_bootstrap(
            season_league_blocks,
            season_league_bootstrap_roi,
            group_label="season-league",
        ),
        "season_bootstrap_summary": summarize_bootstrap(
            season_blocks,
            season_bootstrap_roi,
            group_label="season",
        ),
        "leave_one_out_summary": leave_one_group_out(
            season_league_blocks, ["season", "league"]
        ),
        "profit_contribution_summary": profit_contribution(
            season_league_blocks,
            contribution_levels=contribution_levels,
        ),
        "season_bet_level_bootstrap_summary": bootstrap_bet_level_by_group(
            source_bets,
            ["season"],
            simulations=simulations,
            seed=seed + 2,
        ),
    }


def prepare_kelly_input(bets: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Prepare selected paper-rule bets for Kelly staking."""
    required_columns = [
        "source",
        "date",
        "season",
        "league",
        "home_team",
        "away_team",
        "corrected_probability",
        "outcome_odds",
        "expected_value",
        "flat_profit",
        "outcome",
    ]
    missing = [column for column in required_columns if column not in bets.columns]
    if missing:
        raise ValueError(f"Kelly staking requires missing column(s): {missing}")

    frame = bets.loc[bets["source"].eq(source), required_columns].copy()
    if frame.empty:
        return frame

    frame["model_probability"] = frame["corrected_probability"].clip(1e-6, 1 - 1e-6)
    frame["kelly_full_fraction"] = (
        (frame["model_probability"] * frame["outcome_odds"] - 1.0)
        / (frame["outcome_odds"] - 1.0)
    )
    frame["kelly_full_fraction"] = frame["kelly_full_fraction"].clip(lower=0.0)
    return frame.sort_values(
        ["date", "season", "league", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)


def max_drawdown_from_path(values: pd.Series) -> float:
    """Return max drawdown as a positive decimal value."""
    if values.empty:
        return np.nan
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(-drawdown.min())


def simulate_fractional_kelly_path(
    kelly_input: pd.DataFrame,
    *,
    fraction: float,
    initial_bankroll: float,
    max_bet_fraction: float,
    max_date_exposure_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a chronological fractional-Kelly bankroll path."""
    if kelly_input.empty:
        return pd.DataFrame(), pd.DataFrame()

    bankroll = float(initial_bankroll)
    bet_rows = []
    path_rows = [{"date": pd.NaT, "bankroll": bankroll, "kelly_fraction": fraction}]

    for date, date_frame in kelly_input.groupby("date", sort=True):
        date_frame = date_frame.copy()
        bankroll_before_date = bankroll
        stake_fraction = (date_frame["kelly_full_fraction"] * fraction).clip(
            lower=0.0, upper=max_bet_fraction
        )

        total_date_exposure = stake_fraction.sum()
        if total_date_exposure > max_date_exposure_fraction:
            stake_fraction = stake_fraction * (max_date_exposure_fraction / total_date_exposure)

        stake = bankroll_before_date * stake_fraction
        profit = stake * date_frame["flat_profit"].to_numpy(float)
        bankroll = bankroll_before_date + profit.sum()

        output = date_frame.copy()
        output["kelly_fraction"] = fraction
        output["stake_fraction"] = stake_fraction.to_numpy(float)
        output["stake"] = stake.to_numpy(float)
        output["kelly_profit"] = profit
        output["bankroll_before_date"] = bankroll_before_date
        output["bankroll_after_date"] = bankroll
        bet_rows.append(output)
        path_rows.append({"date": date, "bankroll": bankroll, "kelly_fraction": fraction})

    return pd.concat(bet_rows, ignore_index=True), pd.DataFrame(path_rows)


def summarize_kelly_results(
    results: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    initial_bankroll: float,
) -> pd.DataFrame:
    """Summarize Kelly staking paths."""
    if results.empty:
        return pd.DataFrame()

    rows = []
    for fraction, group in results.groupby("kelly_fraction", sort=True):
        path = paths.loc[paths["kelly_fraction"].eq(fraction)].sort_values(
            "date", na_position="first"
        )
        final_bankroll = path["bankroll"].iloc[-1]
        total_stake = group["stake"].sum()
        total_profit = group["kelly_profit"].sum()
        rows.append(
            {
                "kelly_fraction": fraction,
                "bets": len(group),
                "final_bankroll": final_bankroll,
                "bankroll_return_pct": (final_bankroll / initial_bankroll - 1.0) * 100,
                "staking_roi_pct": total_profit / total_stake * 100
                if total_stake > 0
                else np.nan,
                "total_stake": total_stake,
                "total_profit": total_profit,
                "average_stake_fraction_pct": group["stake_fraction"].mean() * 100,
                "max_stake_fraction_pct": group["stake_fraction"].max() * 100,
                "max_drawdown_pct": max_drawdown_from_path(path["bankroll"]) * 100,
                "hit_rate_pct": group["outcome"].mean() * 100,
                "average_ev_pct": group["expected_value"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def summarize_kelly_by_group(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize Kelly PnL by season, league, or both."""
    columns = [
        "kelly_fraction",
        *group_columns,
        "bets",
        "stake",
        "profit",
        "roi_pct",
        "hit_rate_pct",
        "average_stake_fraction_pct",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        results.groupby(["kelly_fraction", *group_columns], sort=True)
        .agg(
            bets=("outcome", "size"),
            stake=("stake", "sum"),
            profit=("kelly_profit", "sum"),
            hit_rate_pct=("outcome", lambda values: values.mean() * 100),
            average_stake_fraction_pct=("stake_fraction", lambda values: values.mean() * 100),
        )
        .reset_index()
    )
    summary["roi_pct"] = np.where(
        summary["stake"].gt(0),
        summary["profit"] / summary["stake"] * 100,
        np.nan,
    )
    return summary[columns]


def bootstrap_kelly_paths(
    results: pd.DataFrame,
    *,
    simulations: int,
    seed: int,
    initial_bankroll: float,
    drawdown_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bet-level bootstrap of Kelly bankroll paths."""
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    simulation_rows = []
    summary_rows = []

    for fraction, group in results.groupby("kelly_fraction", sort=True):
        stake_fraction = group["stake_fraction"].to_numpy(float)
        profit_per_unit = group["flat_profit"].to_numpy(float)
        n_bets = len(group)
        sampled_indices = rng.integers(0, n_bets, size=(simulations, n_bets))
        sampled_growth = 1.0 + stake_fraction[sampled_indices] * profit_per_unit[sampled_indices]
        bankroll_paths = initial_bankroll * np.cumprod(sampled_growth, axis=1)
        final_bankroll = bankroll_paths[:, -1]
        wealth_paths = np.column_stack([np.full(simulations, initial_bankroll), bankroll_paths])
        running_max = np.maximum.accumulate(wealth_paths, axis=1)
        drawdown_paths = wealth_paths / running_max - 1.0
        max_drawdown = -drawdown_paths.min(axis=1)
        return_pct = (final_bankroll / initial_bankroll - 1.0) * 100

        simulation_rows.append(
            pd.DataFrame(
                {
                    "kelly_fraction": fraction,
                    "simulation": np.arange(simulations),
                    "final_bankroll": final_bankroll,
                    "bankroll_return_pct": return_pct,
                    "max_drawdown_pct": max_drawdown * 100,
                }
            )
        )
        summary_rows.append(
            {
                "kelly_fraction": fraction,
                "simulations": simulations,
                "mean_return_pct": np.nanmean(return_pct),
                "median_return_pct": np.nanmedian(return_pct),
                "p05_return_pct": np.nanpercentile(return_pct, 5),
                "p95_return_pct": np.nanpercentile(return_pct, 95),
                "probability_losing_money_pct": np.nanmean(final_bankroll < initial_bankroll)
                * 100,
                "probability_drawdown_worse_than_threshold_pct": np.nanmean(
                    max_drawdown > drawdown_threshold
                )
                * 100,
            }
        )

    return pd.concat(simulation_rows, ignore_index=True), pd.DataFrame(summary_rows)


def run_kelly_analysis(
    paper_bets: pd.DataFrame,
    *,
    source: str,
    initial_bankroll: float,
    kelly_fractions: list[float],
    max_bet_fraction: float,
    max_date_exposure_fraction: float,
    bootstrap_sims: int,
    bootstrap_seed: int,
    drawdown_threshold: float,
) -> dict[str, pd.DataFrame]:
    """Run the full Kelly staking diagnostic workflow."""
    kelly_input = prepare_kelly_input(paper_bets, source=source)
    result_frames = []
    path_frames = []
    for fraction in kelly_fractions:
        result_frame, path_frame = simulate_fractional_kelly_path(
            kelly_input,
            fraction=fraction,
            initial_bankroll=initial_bankroll,
            max_bet_fraction=max_bet_fraction,
            max_date_exposure_fraction=max_date_exposure_fraction,
        )
        if not result_frame.empty:
            result_frames.append(result_frame)
            path_frames.append(path_frame)

    kelly_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    kelly_paths = pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    kelly_bootstrap_results, kelly_bootstrap_summary = bootstrap_kelly_paths(
        kelly_results,
        simulations=bootstrap_sims,
        seed=bootstrap_seed,
        initial_bankroll=initial_bankroll,
        drawdown_threshold=drawdown_threshold,
    )
    return {
        "kelly_input": kelly_input,
        "kelly_results": kelly_results,
        "kelly_paths": kelly_paths,
        "kelly_summary": summarize_kelly_results(
            kelly_results,
            kelly_paths,
            initial_bankroll=initial_bankroll,
        ),
        "kelly_season_summary": summarize_kelly_by_group(kelly_results, ["season"]),
        "kelly_league_summary": summarize_kelly_by_group(kelly_results, ["league"]),
        "kelly_bootstrap_results": kelly_bootstrap_results,
        "kelly_bootstrap_summary": kelly_bootstrap_summary,
    }
