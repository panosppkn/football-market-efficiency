# Data

This project uses historical football results and betting odds from
[Football-Data.co.uk](https://www.football-data.co.uk/).

The public notebook is designed around the Football-Data European season files,
where each workbook contains multiple league sheets for one season. Example
filenames:

```text
all-euro-data-2006-2007.xls
all-euro-data-2017-2018.xlsx
```

Raw data files are not redistributed in this repository. Place downloaded
source files under:

```text
data/raw/
```

For the saved public run, the analysis uses the available 2006/07-2025/26
season files and the 11 first-division European leagues selected in the main
notebook. The notebook can also be configured to use a smaller season or league
subset.

## Faster local Parquet cache

Excel loading can be slow, so the repository includes a reproducible conversion
script that creates one Parquet file per season:

```bash
python scripts/build_all_euro_season_parquets.py
```

The generated files are written to:

```text
data/processed/all_euro_by_season/
```

The Parquet files are a local cache only. They preserve the original
Football-Data columns as closely as possible, add metadata columns such as
`season`, `source_file`, `source_sheet`, `division`, and `league`, and do not
normalize odds, probabilities, or outcomes.

Both raw files and generated Parquet files are excluded from Git.

## Dependencies

Reading historical Excel files requires the dependencies installed by the
project:

- `xlrd` for `.xls` files;
- `openpyxl` for `.xlsx` files;
- `pyarrow` for Parquet output.

Install them with:

```bash
python -m pip install -e ".[dev]"
```

## Reproducibility recommendation

Football-Data source files may be revised over time. For exact reproducibility,
record the download date, source URL, and preferably a SHA-256 checksum for
every raw file used in a saved run.

## Field definitions

See [`data_dictionary.md`](data_dictionary.md) for Football-Data field
definitions and historical column-name conventions.

The repository's MIT License applies to the analysis code, not to the source
datasets. Review the provider's terms before publishing or redistributing any
data.
