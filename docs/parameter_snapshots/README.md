# Parameter snapshots

This folder stores frozen market-efficiency parameter snapshots for prospective monitoring.

Each snapshot should contain only information known at the training cutoff. It should not include future realized ROI, profit, bet counts, or hit rates.

## Current market-maximum snapshot

The table below freezes the current `market_maximum` decision-rule parameters for the `home_win` market, estimated using all available data up to and including season `25_26`. The goal is to fix these parameters before future outcomes are known, then monitor through the season how the pre-specified rule performs.

The same values are stored in machine-readable form here:

[`market_maximum_home_win_asof_2025_26.csv`](market_maximum_home_win_asof_2025_26.csv)

| as_of_training_end_season | market_outcome | source | league | alpha | beta | accepted_probability_min | accepted_probability_max | accepted_odds_min | accepted_odds_max |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 25_26 | home_win | market_maximum | Greek Super League | -0.042914 | 0.087709 | 0.675 | 0.990 | 1.010 | 1.481 |
| 25_26 | home_win | market_maximum | La Liga | -0.014866 | 0.055447 | 0.454 | 0.990 | 1.010 | 2.203 |
| 25_26 | home_win | market_maximum | Primeira Liga | -0.046892 | 0.102088 | 0.595 | 0.990 | 1.010 | 1.681 |
| 25_26 | home_win | market_maximum | Serie A | -0.038014 | 0.073193 | 0.820 | 0.990 | 1.010 | 1.220 |

Snapshot metadata:

```text
training_window = expanding_through_as_of_season
confidence_level = 95%
estimation_method = ols_hc1
```

## Interpretation

A future bet is flagged when its raw implied probability,

```text
p = 1 / odds
```

falls inside the accepted probability range for the corresponding league and market outcome.

Equivalently, the quoted decimal odds should fall inside the accepted odds range. For example, in Greek Super League home-win bets, the current `market_maximum` rule accepts odds approximately between `1.010` and `1.481`.

These parameters define a fixed forward-monitoring rule. Future performance should be evaluated against the frozen ranges above rather than re-estimating parameters after new outcomes are known.
