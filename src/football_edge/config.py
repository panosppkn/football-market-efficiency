"""Project-wide research constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_PATTERN = "*_??_??.csv"
OVER_25_THRESHOLD = 2.5
MINIMUM_EXPECTED_VALUE = 0.03

# The model-free consensus/Kelly experiment uses a smaller buffer because
# Kelly already scales marginal estimated edges to small stakes. Keep this
# separate from the pooled-model selection threshold above.
CONSENSUS_KELLY_MINIMUM_EXPECTED_VALUE = 0.005

# Prespecified descriptive buckets. They diagnose whether higher model-implied
# value is associated with better realized outcomes; they are not optimized
# thresholds.
EV_BUCKET_EDGES = [-float("inf"), 0.00, 0.02, 0.04, 0.06, 0.10, float("inf")]
EV_BUCKET_LABELS = ["<0%", "0-2%", "2-4%", "4-6%", "6-10%", ">=10%"]

REQUIRED_COLUMNS = {
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
}

ODDS_COLUMNS = [
    "Avg>2.5",
    "Avg<2.5",
    "Max>2.5",
    "Max<2.5",
    "AvgC>2.5",
    "AvgC<2.5",
    "MaxC>2.5",
    "MaxC<2.5",
    "B365>2.5",
    "B365<2.5",
    "B365C>2.5",
    "B365C<2.5",
    "P>2.5",
    "P<2.5",
    "PC>2.5",
    "PC<2.5",
    "BFE>2.5",
    "BFE<2.5",
    "BFEC>2.5",
    "BFEC<2.5",
]

MODEL_FEATURES = [
    "market_logit",
    "home_season_avg_goals",
    "away_season_avg_goals",
    "home_last_5_avg_goals",
    "away_last_5_avg_goals",
]

# Notebook 02 holds the pooled two-season architecture fixed and varies only
# ridge strength. The grid is deliberately small and prespecified.
REGULARIZATION_GRID = {
    "L2 = 1 (baseline)": 1.0,
    "L2 = 10": 10.0,
    "L2 = 100": 100.0,
}

PRICE_SCENARIOS = {
    "average_preclosing": ("Avg>2.5", "AvgC>2.5"),
    "best_preclosing": ("Max>2.5", "MaxC>2.5"),
}

# Optional execution-price sensitivity. A closing quote has no later price in
# the dataset against which conventional CLV can be measured.
CLOSING_PRICE_SCENARIOS = {
    "best_closing": "MaxC>2.5",
}

# Football-Data does not specify an applicable Betfair Exchange commission rate.
# The primary analysis therefore reports gross quoted odds with zero commission.
# An account-specific rate can be introduced later as a sensitivity analysis.
BOOKMAKER_SCENARIOS = {
    "bet365": {
        "price_column": "B365>2.5",
        "closing_column": "B365C>2.5",
        "commission": 0.00,
    },
    "pinnacle": {
        "price_column": "P>2.5",
        "closing_column": "PC>2.5",
        "commission": 0.00,
    },
    "betfair_exchange": {
        "price_column": "BFE>2.5",
        "closing_column": "BFEC>2.5",
        "commission": 0.00,
    },
}
