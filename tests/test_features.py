import pandas as pd

from football_edge.features import _team_match_history


def test_features_use_prior_matches_only() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [0, 1, 2],
            "date": pd.to_datetime(
                ["2024-08-01", "2024-08-08", "2024-08-15"]
            ),
            "HomeTeam": ["A", "B", "A"],
            "AwayTeam": ["B", "A", "B"],
            "FTHG": [2, 1, 3],
            "FTAG": [0, 1, 2],
        }
    )

    history = _team_match_history(matches)
    team_a = history.loc[history["team"].eq("A")].sort_values("date")

    assert pd.isna(team_a.iloc[0]["season_avg_goals"])
    assert team_a.iloc[1]["season_avg_goals"] == 2.0
    assert team_a.iloc[2]["season_avg_goals"] == 1.5
    assert team_a.iloc[2]["last_5_avg_goals"] == 1.5


def test_rolling_feature_uses_only_five_previous_matches() -> None:
    matches = pd.DataFrame(
        {
            "match_id": range(7),
            "date": pd.date_range("2024-08-01", periods=7, freq="7D"),
            "HomeTeam": ["A"] * 7,
            "AwayTeam": [f"B{i}" for i in range(7)],
            "FTHG": range(1, 8),
            "FTAG": [0] * 7,
        }
    )

    history = _team_match_history(matches)
    team_a = history.loc[history["team"].eq("A")].sort_values("date")

    # Before match seven, the prior five scores are 2, 3, 4, 5, 6.
    assert team_a.iloc[6]["last_5_avg_goals"] == 4.0

