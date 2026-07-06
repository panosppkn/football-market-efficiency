# Data

The raw CSV files are sourced from
[Football-Data.co.uk](https://www.football-data.co.uk/). They contain match
results, match statistics, and pre-closing/closing bookmaker odds for four
European leagues across the 2021/22-2025/26 seasons.

Raw files are retained unchanged under `data/raw/`. Their naming convention is:

```text
<league>_<YY>_<YY>.csv
```

Examples:

- `Premier_League_24_25.csv`
- `La_Liga_24_25.csv`
- `Bundesliga_24_25.csv`
- `Seria_A_24_25.csv`

See [`data_dictionary.md`](data_dictionary.md) for the source field definitions.

## Download instructions

The raw CSV files are intentionally excluded from Git and must be downloaded
before running the notebooks:

1. Open the
   [Football-Data historical download page](https://www.football-data.co.uk/downloadm.php).
2. Download the Bundesliga, La Liga, Premier League, and Serie A CSV files for
   seasons 2021/22 through 2025/26.
3. Create `data/raw/` if it does not already exist.
4. Place the unchanged files in that directory and rename them according to
   the convention above.

The notebooks will raise `FileNotFoundError` when no matching files are
available. Exact reproduction requires the same source-file versions; record
the download date and preferably a SHA-256 checksum for every file.

The repository's MIT License applies to the analysis code, not to these source
datasets. Review the provider's terms before publishing or redistributing them.

The 2025/26 files should be treated according to their provenance and
collection date. If they contain forecasts, simulated results, or data
unavailable at the time of analysis, exclude them from claims about realized
historical performance.

## Pinnacle 2025/26 caution

Football-Data reports that Pinnacle odds became unreliable after 23 July 2025
because of public-API problems. Treat Pinnacle 2025/26 observations and
cross-season comparisons with caution.
