# Notebook guide

This folder contains a small public notebook sequence:

1. `01_market_efficiency_decision_rule.ipynb`
   - Main research notebook.
   - Tests a market-efficiency decision rule inspired by Angelini and De Angelis
     (2019), using raw odds-implied probabilities and expanding-window
     walk-forward parameter estimation.
   - Reports flat-stake results, market-maximum price-dispersion diagnostics,
     fractional-Kelly staking, and bootstrap robustness checks.

2. `02_forward_market_maximum_benchmark.ipynb`
   - Applies the frozen home-win rule to all completed current-season matches.
   - Assumes every qualifying bet is fully placed at Football-Data `MaxH`.
   - Reports the reference-price benchmark separately from venue-specific execution feasibility.

3. `03_forward_polymarket_data_collection.ipynb`
   - Applies the frozen `market_maximum` home-win rule to one completed calendar month.
   - Resolves exact Polymarket markets and retrieves eligible Dune transaction observations.
   - Writes local league-month audit files for later forward execution analysis; it does not report P&L.

4. `04_forward_polymarket_execution.ipynb`
   - Loads all finalized league-month files for the configured forward season.
   - Simulates chronological fills and settlements under frozen fee, premium, participation, staking, and exposure assumptions.
   - Reports combined-portfolio and standalone league-sleeve bankroll and risk diagnostics.

For the latest consolidated 2026/27 forward results, see the [forward holdout report](../reports/current_season_holdout/README.md).

Reusable code for these notebooks lives primarily in:

- `src/football_edge/data.py`;
- `src/football_edge/market_efficiency.py`;
- `src/football_edge/forward_benchmark.py`;
- `src/football_edge/forward_polymarket.py`;
- `src/football_edge/forward_execution.py`;
- `src/football_edge/plotting.py`;
- `src/football_edge/config.py`.
