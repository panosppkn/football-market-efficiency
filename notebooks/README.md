# Notebook guide

These notebooks are presentation-layer research reports. The reusable data
loading, feature engineering, modeling, backtesting, and plotting logic lives in
`src/football_edge/`.

Recommended reading order:

1. `01_main_pooled_l2_100_edge_analysis.ipynb`
   - Main selected-model analysis.
   - Uses one pooled all-league market-anchored logistic model.
   - Trains on the previous two seasons before each test season.
   - Uses the strongest ridge setting from notebook 02 (`L2 = 100`).

2. `02_pooled_regularization_comparison.ipynb`
   - Model-selection and robustness notebook.
   - Compares fixed L2 penalties of 1, 10, and 100 using the same pooled
     rolling validation design.

3. `03_ev_bucket_diagnostics.ipynb`
   - Tests whether the selected pooled model's estimated EV ranks candidates
     sensibly across fixed buckets.
   - Uses fixed EV buckets to avoid threshold hunting.

4. `04_league_specific_baseline.ipynb`
   - Baseline and research-evolution notebook.
   - Fits separate league-level market-anchored models.
   - Kept to document why the project moved toward pooled seasonal modeling.

5. `05_consensus_kelly_staking.ipynb`
   - Independent, model-free benchmark using the no-vig average market
     probability rather than the pooled logistic model.
   - Tests flat stakes and prespecified fractional Kelly rules at Bet365,
     Pinnacle, Betfair Exchange, the pre-closing maximum, and the explicitly
     non-executable closing-maximum sensitivity.
   - Uses a dedicated 0.5% consensus-Kelly EV threshold, separate from the
     pooled model's 3% threshold.
   - Reports calibration, date-cluster bootstrap uncertainty, bankroll
     drawdown, turnover, concentration, and league-season stability.

The notebooks intentionally keep conclusions cautious. The dataset does not
contain reliable exact quote timestamps, so the backtests should be interpreted
as historical research diagnostics rather than production-ready betting
results.
