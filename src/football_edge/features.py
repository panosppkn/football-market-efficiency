"""Leakage-safe, prior-match football features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_edge.config import ODDS_COLUMNS, OVER_25_THRESHOLD
from football_edge.data import load_matches


def _team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["match_id", "date", "HomeTeam", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "team", "FTHG": "goals", "FTAG": "goals_conceded"}
    )
    home["venue"] = "home"

    away = matches[["match_id", "date", "AwayTeam", "FTAG", "FTHG"]].rename(
        columns={"AwayTeam": "team", "FTAG": "goals", "FTHG": "goals_conceded"}
    )
    away["venue"] = "away"

    history = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "date", "match_id"], kind="stable")
        .reset_index(drop=True)
    )
    grouped = history.groupby("team", sort=False)

    # shift(1) makes every feature available before the current match.
    history["season_avg_goals"] = grouped["goals"].transform(
        lambda goals: goals.shift(1).expanding().mean()
    )
    history["last_5_avg_goals"] = grouped["goals"].transform(
        lambda goals: goals.shift(1).rolling(window=5, min_periods=1).mean()
    )
    history["season_avg_goals_conceded"] = grouped["goals_conceded"].transform(
        lambda goals: goals.shift(1).expanding().mean()
    )
    history["last_5_goals_conceded"] = grouped["goals_conceded"].transform(
        lambda goals: goals.shift(1).rolling(window=5, min_periods=1).mean()
    )
    return history


def create_goal_features(path: str | Path) -> pd.DataFrame:
    """Build match-level features using only earlier matches in this CSV."""
    matches = load_matches(path)
    available_odds = [column for column in ODDS_COLUMNS if column in matches.columns]
    for column in available_odds:
        matches[column] = pd.to_numeric(matches[column], errors="coerce")
        # Source files use zero as a missing-quote sentinel in a few rows.
        matches[column] = matches[column].where(matches[column] > 1)

    history = _team_match_history(matches)

    feature_columns = [
        "match_id",
        "season_avg_goals",
        "last_5_avg_goals",
        "season_avg_goals_conceded",
        "last_5_goals_conceded",
    ]
    home_features = history.loc[
        history["venue"].eq("home"), feature_columns
    ].rename(
        columns={
            "season_avg_goals": "home_season_avg_goals",
            "last_5_avg_goals": "home_last_5_avg_goals",
            "season_avg_goals_conceded": "home_season_avg_goals_conceded",
            "last_5_goals_conceded": "home_last_5_goals_conceded",
        }
    )
    away_features = history.loc[
        history["venue"].eq("away"), feature_columns
    ].rename(
        columns={
            "season_avg_goals": "away_season_avg_goals",
            "last_5_avg_goals": "away_last_5_avg_goals",
            "season_avg_goals_conceded": "away_season_avg_goals_conceded",
            "last_5_goals_conceded": "away_last_5_goals_conceded",
        }
    )

    match_columns = [
        "match_id",
        "date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        *available_odds,
    ]
    result = (
        matches[match_columns]
        .merge(home_features, on="match_id", validate="one_to_one")
        .merge(away_features, on="match_id", validate="one_to_one")
        .rename(
            columns={
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_ft_goals",
                "FTAG": "away_ft_goals",
            }
        )
        .drop(columns="match_id")
    )

    result["total_season_avg_goals"] = (
        result["home_season_avg_goals"] + result["away_season_avg_goals"]
    )
    result["total_last_5_avg_goals"] = (
        result["home_last_5_avg_goals"] + result["away_last_5_avg_goals"]
    )
    result["total_season_avg_goals_conceded"] = (
        result["home_season_avg_goals_conceded"]
        + result["away_season_avg_goals_conceded"]
    )
    result["total_last_5_goals_conceded"] = (
        result["home_last_5_goals_conceded"]
        + result["away_last_5_goals_conceded"]
    )
    result["total_ft_goals"] = (
        result["home_ft_goals"] + result["away_ft_goals"]
    )
    result["over_2_5"] = (
        result["total_ft_goals"] > OVER_25_THRESHOLD
    ).astype(int)

    return result
