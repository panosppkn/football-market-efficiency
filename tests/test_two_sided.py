import numpy as np
import pandas as pd

from football_edge.two_sided import (
    build_two_sided_candidates,
    select_two_sided_bets,
    summarize_two_sided_bets,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-08-01", "2025-08-02"]),
            "league": ["League A", "League A"],
            "season": ["25_26", "25_26"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "model_probability": [0.60, 0.50],
            "over_2_5": [1, 0],
            "B365>2.5": [2.0, 2.1],
            "B365<2.5": [2.0, 2.2],
            "B365C>2.5": [1.9, 2.0],
            "B365C<2.5": [2.1, 2.1],
        }
    )


def test_two_sided_probabilities_are_complements() -> None:
    candidates = build_two_sided_candidates(
        _predictions(), sources=["bet365"]
    )
    probabilities = candidates.pivot(
        index="match_row_id", columns="bet_side", values="signal_probability"
    )

    assert np.allclose(probabilities["over"] + probabilities["under"], 1)


def test_two_sided_settlement_uses_selected_side() -> None:
    candidates = build_two_sided_candidates(
        _predictions(), sources=["bet365"]
    )
    first_match = candidates.loc[candidates["match_row_id"].eq(0)]

    over = first_match.loc[first_match["bet_side"].eq("over")].iloc[0]
    under = first_match.loc[first_match["bet_side"].eq("under")].iloc[0]
    assert over["bet_won"]
    assert np.isclose(over["profit"], 1.0)
    assert not under["bet_won"]
    assert np.isclose(under["profit"], -1.0)


def test_best_side_selects_one_higher_ev_side_and_flags_conflict() -> None:
    candidates = build_two_sided_candidates(
        _predictions(), sources=["bet365"]
    )
    selected = select_two_sided_bets(
        candidates,
        minimum_expected_value=0.03,
        policy="best_side",
    )

    assert selected.groupby(["execution_source", "match_row_id"]).size().eq(1).all()
    second = selected.loc[selected["match_row_id"].eq(1)].iloc[0]
    assert second["bet_side"] == "under"
    assert second["both_sides_qualify"]
    assert second["qualifying_sides"] == 2


def test_side_specific_policies_and_summary() -> None:
    candidates = build_two_sided_candidates(
        _predictions(), sources=["bet365"]
    )
    over = select_two_sided_bets(
        candidates, minimum_expected_value=0.0, policy="over_only"
    )
    under = select_two_sided_bets(
        candidates, minimum_expected_value=0.0, policy="under_only"
    )
    combined = pd.concat([over, under], ignore_index=True)
    summary = summarize_two_sided_bets(combined, ["bet_side"])

    assert over["bet_side"].eq("over").all()
    assert under["bet_side"].eq("under").all()
    assert set(summary["bet_side"]) == {"over", "under"}
