"""Convert Football-Data all-Europe season workbooks to per-season Parquet files.

The conversion is intentionally close to the raw Excel files:

- every season workbook becomes one Parquet file;
- all relevant league sheets are concatenated;
- original Football-Data columns are preserved;
- metadata columns are added to identify season, workbook, sheet/division, and league;
- fully empty rows/columns and duplicated header rows are removed and reported.

This script does not engineer features, normalize probabilities, or change outcomes.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from football_edge.data import LEAGUE_CODE_TO_NAME


RAW_PATTERNS = ("all-euro-data-*.xls", "all-euro-data-*.xlsx")
FILENAME_RE = re.compile(r"all-euro(?:-data)?-(\d{4})-(\d{4})\.(xls|xlsx)$", re.IGNORECASE)

CORE_LEAGUE_COLUMNS = {"Date", "HomeTeam", "AwayTeam"}
IMPORTANT_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "AvgH",
    "AvgD",
    "AvgA",
    "MaxH",
    "MaxD",
    "MaxA",
    "Avg>2.5",
    "Avg<2.5",
    "Max>2.5",
    "Max<2.5",
    "BbAvH",
    "BbAvD",
    "BbAvA",
    "BbMxH",
    "BbMxD",
    "BbMxA",
    "BbAv>2.5",
    "BbAv<2.5",
    "BbMx>2.5",
    "BbMx<2.5",
]
METADATA_COLUMNS = ["season", "source_file", "source_sheet", "division", "league"]


@dataclass(frozen=True)
class SeasonWorkbook:
    path: Path
    start_year: str
    end_year: str

    @property
    def season(self) -> str:
        return f"{self.start_year[-2:]}_{self.end_year[-2:]}"

    @property
    def output_name(self) -> str:
        return f"all_euro_{self.start_year}_{self.end_year}.parquet"


def parse_workbook_path(path: Path) -> SeasonWorkbook | None:
    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        return None
    start_year, end_year, _suffix = match.groups()
    return SeasonWorkbook(path=path, start_year=start_year, end_year=end_year)


def discover_workbooks(raw_dir: Path, seasons: list[str] | None = None) -> list[SeasonWorkbook]:
    selected = {normalize_season_key(season) for season in seasons} if seasons else None
    workbooks: list[SeasonWorkbook] = []
    for pattern in RAW_PATTERNS:
        for path in sorted(raw_dir.glob(pattern)):
            workbook = parse_workbook_path(path)
            if workbook is None:
                continue
            if selected is not None and workbook.season not in selected:
                continue
            workbooks.append(workbook)
    return sorted(workbooks, key=lambda item: (item.start_year, item.end_year, item.path.name))


def normalize_season_key(season: str) -> str:
    parts = re.findall(r"\d+", str(season))
    if len(parts) >= 2:
        return f"{parts[0][-2:]}_{parts[1][-2:]}"
    return str(season).replace("/", "_").replace("-", "_")


def excel_engine(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        return "xlrd"
    if suffix == ".xlsx":
        return "openpyxl"
    raise ValueError(f"Unsupported Excel suffix: {path.suffix}")


def is_duplicate_header_row(row: pd.Series, columns: pd.Index) -> bool:
    non_missing = row.dropna()
    if non_missing.empty:
        return False
    matches = 0
    checked = 0
    for column, value in row.items():
        if pd.isna(value):
            continue
        checked += 1
        if str(value).strip() == str(column).strip():
            matches += 1
    return checked >= 3 and matches >= max(3, int(0.6 * checked))


def clean_sheet(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    original_rows, original_columns = frame.shape
    output = frame.copy()

    empty_rows = int(output.isna().all(axis=1).sum())
    output = output.dropna(axis=0, how="all")

    empty_columns = int(output.isna().all(axis=0).sum())
    output = output.dropna(axis=1, how="all")

    duplicate_header_mask = output.apply(lambda row: is_duplicate_header_row(row, output.columns), axis=1)
    duplicate_header_rows = int(duplicate_header_mask.sum())
    if duplicate_header_rows:
        output = output.loc[~duplicate_header_mask].copy()

    return output.reset_index(drop=True), {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "empty_rows_removed": empty_rows,
        "empty_columns_removed": empty_columns,
        "duplicate_header_rows_removed": duplicate_header_rows,
    }


def is_relevant_league_sheet(frame: pd.DataFrame) -> bool:
    return CORE_LEAGUE_COLUMNS.issubset(set(map(str, frame.columns)))


def date_parse_success_rate(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
    non_missing = values.notna()
    if int(non_missing.sum()) == 0:
        return 0.0
    return float(parsed.loc[non_missing].notna().mean() * 100)


def duplicated_match_count(frame: pd.DataFrame) -> int:
    required = ["Date", "HomeTeam", "AwayTeam"]
    if not set(required).issubset(frame.columns):
        return 0
    return int(frame.duplicated(subset=required, keep=False).sum())

def prepare_for_parquet(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cast only Arrow-incompatible mixed-type columns to string.

    Football-Data changed formats over time. Some historical Asian-handicap
    columns contain both numeric values and strings such as "-0.5,-1". Parquet
    requires one physical type per column, so these specific columns are stored
    as strings to preserve their original values.
    """
    output = frame.copy()
    cast_columns: list[str] = []
    for column in output.columns:
        try:
            pa.array(output[column], from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError):
            output[column] = output[column].astype("string")
            cast_columns.append(str(column))
    return output, cast_columns



def convert_workbook(workbook: SeasonWorkbook, output_dir: Path) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    engine = excel_engine(workbook.path)
    excel = pd.ExcelFile(workbook.path, engine=engine)
    season_frames: list[pd.DataFrame] = []
    sheet_rows: list[dict] = []
    skipped_rows: list[dict] = []

    for sheet_name in excel.sheet_names:
        raw_sheet = pd.read_excel(excel, sheet_name=sheet_name, dtype=object)
        cleaned, cleaning = clean_sheet(raw_sheet)
        sheet_label = str(sheet_name)
        division = sheet_label
        league = LEAGUE_CODE_TO_NAME.get(division, division)

        if cleaned.empty:
            skipped_rows.append(
                {
                    "season": workbook.season,
                    "source_file": workbook.path.name,
                    "source_sheet": sheet_label,
                    "reason": "empty sheet after removing fully empty rows/columns",
                }
            )
            continue

        if not is_relevant_league_sheet(cleaned):
            skipped_rows.append(
                {
                    "season": workbook.season,
                    "source_file": workbook.path.name,
                    "source_sheet": sheet_label,
                    "reason": "missing core league columns: Date, HomeTeam, AwayTeam",
                }
            )
            continue

        cleaned.insert(0, "league", league)
        cleaned.insert(0, "division", division)
        cleaned.insert(0, "source_sheet", sheet_label)
        cleaned.insert(0, "source_file", workbook.path.name)
        cleaned.insert(0, "season", workbook.season)
        season_frames.append(cleaned)

        sheet_rows.append(
            {
                "season": workbook.season,
                "source_file": workbook.path.name,
                "source_sheet": sheet_label,
                "division": division,
                "league": league,
                "rows": len(cleaned),
                "columns": cleaned.shape[1],
                "date_parse_success_pct": date_parse_success_rate(cleaned["Date"]),
                "duplicated_match_rows": duplicated_match_count(cleaned),
                **cleaning,
            }
        )

    if season_frames:
        season_frame = pd.concat(season_frames, ignore_index=True, sort=False)
    else:
        season_frame = pd.DataFrame(columns=METADATA_COLUMNS)

    output_path = output_dir / workbook.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_frame, cast_columns = prepare_for_parquet(season_frame)
    parquet_frame.to_parquet(output_path, index=False)

    if cast_columns:
        for row in sheet_rows:
            row["columns_cast_to_string_for_parquet"] = ", ".join(cast_columns)
    else:
        for row in sheet_rows:
            row["columns_cast_to_string_for_parquet"] = ""

    return season_frame, sheet_rows, skipped_rows, cast_columns


def build_validation_reports(output_dir: Path, season_summaries: list[dict], sheet_rows: list[dict], skipped_rows: list[dict], season_columns: list[dict]) -> None:
    validation_dir = output_dir / "_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    season_summary = pd.DataFrame(season_summaries)
    sheet_summary = pd.DataFrame(sheet_rows)
    skipped_summary = pd.DataFrame(
        skipped_rows,
        columns=["season", "source_file", "source_sheet", "reason"],
    )
    columns_by_season = pd.DataFrame(season_columns)

    if not columns_by_season.empty:
        season_to_columns = {
            season: set(group["column"])
            for season, group in columns_by_season.groupby("season", sort=True)
        }
        all_columns = sorted(set().union(*season_to_columns.values()))
        column_diff_rows = []
        for season, present in season_to_columns.items():
            missing = sorted(set(all_columns).difference(present))
            extra_vs_all = sorted(present.difference(set.intersection(*season_to_columns.values())))
            column_diff_rows.append(
                {
                    "season": season,
                    "columns_missing_from_this_season": ", ".join(missing),
                    "columns_not_common_to_all_seasons": ", ".join(extra_vs_all),
                }
            )
        column_differences = pd.DataFrame(column_diff_rows)
    else:
        column_differences = pd.DataFrame(columns=["season", "columns_missing_from_this_season", "columns_not_common_to_all_seasons"])

    season_summary.to_csv(validation_dir / "season_conversion_summary.csv", index=False)
    sheet_summary.to_csv(validation_dir / "sheet_conversion_summary.csv", index=False)
    skipped_summary.to_csv(validation_dir / "skipped_sheets.csv", index=False)
    columns_by_season.to_csv(validation_dir / "columns_by_season.csv", index=False)
    column_differences.to_csv(validation_dir / "column_differences_by_season.csv", index=False)

    print("\nSeason conversion summary")
    if season_summary.empty:
        print("No seasons converted.")
    else:
        print(season_summary.to_string(index=False))

    print("\nValidation reports written to:", validation_dir)
    print("- season_conversion_summary.csv")
    print("- sheet_conversion_summary.csv")
    print("- skipped_sheets.csv")
    print("- columns_by_season.csv")
    print("- column_differences_by_season.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/all_euro_by_season"))
    parser.add_argument("--seasons", nargs="*", default=None, help="Optional seasons such as 20_21 or 2020/2021.")
    args = parser.parse_args()

    workbooks = discover_workbooks(args.raw_dir, seasons=args.seasons)
    if not workbooks:
        raise FileNotFoundError(f"No all-euro-data Excel files found in {args.raw_dir}")

    season_summaries: list[dict] = []
    all_sheet_rows: list[dict] = []
    all_skipped_rows: list[dict] = []
    season_columns: list[dict] = []

    for workbook in workbooks:
        print(f"Converting {workbook.path.name} ...")
        season_frame, sheet_rows, skipped_rows, cast_columns = convert_workbook(workbook, args.output_dir)
        output_path = args.output_dir / workbook.output_name
        important_missing = [column for column in IMPORTANT_COLUMNS if column not in season_frame.columns]
        season_summaries.append(
            {
                "season": workbook.season,
                "raw_file": workbook.path.name,
                "output_parquet_file": str(output_path),
                "sheets_read": len(sheet_rows),
                "rows": len(season_frame),
                "columns": season_frame.shape[1],
                "skipped_sheets": "; ".join(row["source_sheet"] for row in skipped_rows) if skipped_rows else "",
                "important_missing_columns": ", ".join(important_missing),
                "columns_cast_to_string_for_parquet": ", ".join(cast_columns),
            }
        )
        all_sheet_rows.extend(sheet_rows)
        all_skipped_rows.extend(skipped_rows)
        season_columns.extend(
            {"season": workbook.season, "raw_file": workbook.path.name, "column": column}
            for column in season_frame.columns
        )

    build_validation_reports(args.output_dir, season_summaries, all_sheet_rows, all_skipped_rows, season_columns)


if __name__ == "__main__":
    main()
