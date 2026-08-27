# Notebook guide

This folder intentionally contains one public research notebook:

1. `01_market_efficiency_decision_rule.ipynb`
   - Main research notebook.
   - Tests a market-efficiency decision rule inspired by Angelini and De Angelis
     (2019), using raw odds-implied probabilities and expanding-window
     walk-forward parameter estimation.
   - Reports flat-stake results, market-maximum price-dispersion diagnostics,
     fractional-Kelly staking, and bootstrap robustness checks.

Reusable code for this notebook lives primarily in:

- `src/football_edge/data.py`;
- `src/football_edge/market_efficiency.py`;
- `src/football_edge/plotting.py`;
- `src/football_edge/config.py`.
