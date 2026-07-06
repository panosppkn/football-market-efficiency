import numpy as np
import pandas as pd

from football_edge.staking import (
    build_consensus_candidates,
    cluster_bootstrap_roi,
    full_kelly_fraction,
    simulate_kelly_bankroll,
    summarize_kelly_paths,
)


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01 15:00", "2024-08-02 15:00"]),
            "league": ["League A", "League A"],
            "season": ["24_25", "24_25"],
            "over_2_5": [1, 0],
            "Avg>2.5": [2.0, 2.0],
            "Avg<2.5": [2.0, 2.0],
            "AvgC>2.5": [2.0, 2.0],
            "Max>2.5": [2.2, 2.2],
            "MaxC>2.5": [2.3, 2.3],
            "B365>2.5": [2.2, 2.2],
            "B365C>2.5": [2.1, 2.1],
            "P>2.5": [2.2, 2.2],
            "PC>2.5": [2.1, 2.1],
            "BFE>2.5": [2.2, 2.2],
            "BFEC>2.5": [2.1, 2.1],
        }
    )


def test_consensus_candidates_use_no_vig_average_probability() -> None:
    candidates = build_consensus_candidates(_matches())

    assert set(candidates["execution_source"]) == {
        "bet365",
        "pinnacle",
        "betfair_exchange",
        "best_preclosing",
        "best_closing",
    }
    assert candidates["signal_probability"].eq(0.5).all()
    assert np.allclose(candidates["expected_value"], candidates["bet_odds"] / 2 - 1)
    assert candidates.loc[
        candidates["execution_source"].eq("best_closing"),
        "is_executable_proxy",
    ].eq(False).all()


def test_binary_kelly_fraction() -> None:
    result = full_kelly_fraction([0.55, 0.40], [2.0, 2.0])

    assert np.allclose(result, [0.10, 0.0])


def test_simultaneous_bets_do_not_compound_within_timestamp() -> None:
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01 15:00"] * 2),
            "execution_source": ["source"] * 2,
            "signal_probability": [0.60, 0.60],
            "bet_odds": [2.0, 2.0],
            "profit": [1.0, -1.0],
        }
    )
    paths = simulate_kelly_bankroll(
        bets,
        configurations={"Quarter Kelly": (0.25, 0.10)},
        initial_bankroll=100,
    )

    assert paths["bankroll_before"].eq(100).all()
    assert np.allclose(paths["stake"], [5.0, 5.0])
    assert paths["bankroll_after_batch"].eq(100).all()


def test_kelly_summary_reports_path_drawdown() -> None:
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-01", "2024-08-02"]),
            "execution_source": ["source"] * 2,
            "signal_probability": [0.60, 0.60],
            "bet_odds": [2.0, 2.0],
            "profit": [1.0, -1.0],
        }
    )
    paths = simulate_kelly_bankroll(
        bets,
        configurations={"Quarter Kelly": (0.25, 0.10)},
        initial_bankroll=100,
    )
    summary = summarize_kelly_paths(paths, initial_bankroll=100).iloc[0]

    assert np.isclose(summary["final_bankroll"], 99.75)
    assert np.isclose(summary["max_drawdown_pct"], 5.0)


def test_cluster_bootstrap_is_reproducible() -> None:
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-08-01", "2024-08-01", "2024-08-02"]
            ),
            "execution_source": ["source"] * 3,
            "profit": [1.0, -1.0, 1.0],
        }
    )

    first = cluster_bootstrap_roi(bets, n_bootstrap=100, random_seed=7)
    second = cluster_bootstrap_roi(bets, n_bootstrap=100, random_seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "date_clusters"] == 2
    assert not first.loc[0, "sufficient_clusters_for_inference"]
    assert np.isnan(first.loc[0, "roi_95_low_pct"])
