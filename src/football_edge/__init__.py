"""Tools for reproducible football-market edge research."""

from football_edge.backtest import (
    build_execution_candidates,
    bookmaker_coverage,
    coefficient_stability_summary,
    ev_bucket_summary,
    monthly_roi,
    probability_performance,
    run_pooled_monthly_recalibration_walk_forward,
    run_pooled_rolling_walk_forward,
    run_regularization_grid,
    run_walk_forward,
    run_walk_forward_with_coefficients,
    select_bookmaker_bets,
    summarize_bets,
)
from football_edge.data import discover_datasets, load_matches
from football_edge.features import create_goal_features
from football_edge.two_sided import (
    build_two_sided_candidates,
    select_two_sided_bets,
    summarize_two_sided_bets,
)

from football_edge.staking import (
    build_consensus_candidates,
    cluster_bootstrap_roi,
    full_kelly_fraction,
    simulate_kelly_bankroll,
    summarize_kelly_paths,
)

__all__ = [
    "create_goal_features",
    "summarize_two_sided_bets",
    "select_two_sided_bets",
    "build_two_sided_candidates",
    "build_consensus_candidates",
    "cluster_bootstrap_roi",
    "full_kelly_fraction",
    "simulate_kelly_bankroll",
    "summarize_kelly_paths",
    "build_execution_candidates",
    "bookmaker_coverage",
    "coefficient_stability_summary",
    "discover_datasets",
    "load_matches",
    "ev_bucket_summary",
    "monthly_roi",
    "probability_performance",
    "run_pooled_monthly_recalibration_walk_forward",
    "run_pooled_rolling_walk_forward",
    "run_regularization_grid",
    "run_walk_forward",
    "run_walk_forward_with_coefficients",
    "select_bookmaker_bets",
    "summarize_bets",
]
