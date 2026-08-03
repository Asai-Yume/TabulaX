#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


OVERLAP_FOLDER_PATTERN = re.compile(r".+_f(?:025|050|075|100)$", re.IGNORECASE)
ID_LIKE_NAMES = {
    "id",
    "index",
    "row_id",
    "rowid",
    "record_id",
    "recordid",
}


@dataclass(frozen=True)
class ConversionResult:
    dataset: str
    source_column: str
    target_column: str
    gt_pairs: int
    status: str


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV with BOM-tolerant UTF-8 and preserve all values as strings."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")

        headers = [str(name).strip() for name in reader.fieldnames]
        rows: list[dict[str, str]] = []

        for row_number, row in enumerate(reader, start=2):
            normalized: dict[str, str] = {}

            for header in headers:
                value = row.get(header)
                if value is None:
                    raise ValueError(
                        f"{path}, row {row_number}: missing value for column {header!r}."
                    )
                normalized[header] = str(value)

            rows.append(normalized)

    return headers, rows


def is_id_like(column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return normalized in ID_LIKE_NAMES or normalized.endswith("_id")


def infer_side_column(
    *,
    table_headers: Sequence[str],
    gt_headers: Sequence[str],
    suffix: str,
    side_name: str,
    override: str | None,
) -> tuple[str, str]:
    """
    Return (table_column, gt_column) for one side.

    A candidate GT column must:
      1. end in the requested suffix, such as "_l";
      2. have a base name that exists in the corresponding table;
      3. not be ID-like, unless it is the only possible candidate.
    """
    if override:
        gt_column = f"{override}{suffix}"

        if override not in table_headers:
            raise ValueError(
                f"Requested {side_name} column {override!r} is not present in "
                f"{side_name}.csv. Available columns: {list(table_headers)}"
            )

        if gt_column not in gt_headers:
            raise ValueError(
                f"Expected GT column {gt_column!r} for requested "
                f"{side_name} column {override!r}. Available GT columns: "
                f"{list(gt_headers)}"
            )

        return override, gt_column

    candidates: list[tuple[str, str]] = []

    for gt_column in gt_headers:
        if not gt_column.endswith(suffix):
            continue

        base_name = gt_column[: -len(suffix)]
        if base_name in table_headers:
            candidates.append((base_name, gt_column))

    non_id_candidates = [
        candidate for candidate in candidates if not is_id_like(candidate[0])
    ]

    if len(non_id_candidates) == 1:
        return non_id_candidates[0]

    if len(non_id_candidates) > 1:
        names = [candidate[0] for candidate in non_id_candidates]
        raise ValueError(
            f"Ambiguous {side_name} join column. Candidates inferred from GT: "
            f"{names}. Use --source-column or --target-column to specify it."
        )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise ValueError(
            f"Could not infer a {side_name} join column. "
            f"{side_name}.csv columns: {list(table_headers)}; "
            f"GT columns: {list(gt_headers)}"
        )

    names = [candidate[0] for candidate in candidates]
    raise ValueError(
        f"Ambiguous {side_name} join column. Candidates: {names}. "
        f"Use --source-column or --target-column to specify it."
    )


def validate_gt_membership(
    *,
    dataset_name: str,
    gt_rows: Sequence[dict[str, str]],
    source_gt_column: str,
    target_gt_column: str,
    left_rows: Sequence[dict[str, str]],
    right_rows: Sequence[dict[str, str]],
    source_column: str,
    target_column: str,
) -> None:
    """Verify that every GT value occurs in the corresponding full table."""
    left_values = {row[source_column] for row in left_rows}
    right_values = {row[target_column] for row in right_rows}

    missing_left: list[str] = []
    missing_right: list[str] = []

    for row in gt_rows:
        source_value = row[source_gt_column]
        target_value = row[target_gt_column]

        if source_value not in left_values:
            missing_left.append(source_value)

        if target_value not in right_values:
            missing_right.append(target_value)

    if missing_left or missing_right:
        messages: list[str] = []

        if missing_left:
            messages.append(
                f"{len(missing_left)} GT source values are absent from "
                f"left.csv[{source_column!r}]. Examples: {missing_left[:5]}"
            )

        if missing_right:
            messages.append(
                f"{len(missing_right)} GT target values are absent from "
                f"right.csv[{target_column!r}]. Examples: {missing_right[:5]}"
            )

        raise ValueError(f"{dataset_name}: " + " ".join(messages))


def write_rows_file(
    output_path: Path,
    source_column: str,
    target_column: str,
) -> None:
    output_path.write_text(
        f"{source_column}:{target_column}\nsource\n",
        encoding="utf-8",
    )


def write_ground_truth_file(
    output_path: Path,
    gt_rows: Sequence[dict[str, str]],
    source_gt_column: str,
    target_gt_column: str,
    source_column: str,
    target_column: str,
) -> None:
    source_header = f"source-{source_column}"
    target_header = f"target-{target_column}"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[source_header, target_header],
        )
        writer.writeheader()

        for row in gt_rows:
            writer.writerow(
                {
                    source_header: row[source_gt_column],
                    target_header: row[target_gt_column],
                }
            )


def convert_dataset(
    dataset_dir: Path,
    *,
    source_column_override: str | None,
    target_column_override: str | None,
    overwrite: bool,
    dry_run: bool,
) -> ConversionResult:
    left_path = dataset_dir / "left.csv"
    right_path = dataset_dir / "right.csv"
    gt_path = dataset_dir / "gt.csv"
    rows_output_path = dataset_dir / "rows.txt"
    gt_output_path = dataset_dir / "ground truth.csv"

    missing_inputs = [
        path.name
        for path in (left_path, right_path, gt_path)
        if not path.is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            f"{dataset_dir.name}: missing required file(s): "
            f"{', '.join(missing_inputs)}"
        )

    existing_outputs = [
        path.name
        for path in (rows_output_path, gt_output_path)
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            f"{dataset_dir.name}: output file(s) already exist: "
            f"{', '.join(existing_outputs)}. Re-run with --overwrite."
        )

    left_headers, left_rows = read_csv(left_path)
    right_headers, right_rows = read_csv(right_path)
    gt_headers, gt_rows = read_csv(gt_path)

    if not gt_rows:
        raise ValueError(f"{dataset_dir.name}: gt.csv contains no pairs.")

    source_column, source_gt_column = infer_side_column(
        table_headers=left_headers,
        gt_headers=gt_headers,
        suffix="_l",
        side_name="left",
        override=source_column_override,
    )
    target_column, target_gt_column = infer_side_column(
        table_headers=right_headers,
        gt_headers=gt_headers,
        suffix="_r",
        side_name="right",
        override=target_column_override,
    )

    validate_gt_membership(
        dataset_name=dataset_dir.name,
        gt_rows=gt_rows,
        source_gt_column=source_gt_column,
        target_gt_column=target_gt_column,
        left_rows=left_rows,
        right_rows=right_rows,
        source_column=source_column,
        target_column=target_column,
    )

    if not dry_run:
        write_rows_file(
            rows_output_path,
            source_column,
            target_column,
        )
        write_ground_truth_file(
            gt_output_path,
            gt_rows,
            source_gt_column,
            target_gt_column,
            source_column,
            target_column,
        )

    return ConversionResult(
        dataset=dataset_dir.name,
        source_column=source_column,
        target_column=target_column,
        gt_pairs=len(gt_rows),
        status="validated" if dry_run else "written",
    )


def discover_dataset_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")

    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and OVERLAP_FOLDER_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: path.name.lower(),
    )


def write_summary_csv(
    summary_path: Path,
    results: Iterable[ConversionResult],
) -> None:
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "source_column",
                "target_column",
                "gt_pairs",
                "status",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "dataset": result.dataset,
                    "source_column": result.source_column,
                    "target_column": result.target_column,
                    "gt_pairs": result.gt_pairs,
                    "status": result.status,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create rows.txt and ground truth.csv for every AutoFJ overlap "
            "dataset folder."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Root folder containing dataset directories such as "
            "Amphibian_f025 and Artwork_f100."
        ),
    )
    parser.add_argument(
        "--source-column",
        help=(
            "Optional source-column override. Use only when every selected "
            "dataset uses the same source column."
        ),
    )
    parser.add_argument(
        "--target-column",
        help=(
            "Optional target-column override. Use only when every selected "
            "dataset uses the same target column."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rows.txt and ground truth.csv files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report conversions without writing output files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue converting other datasets after one dataset fails.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help=(
            "Optional summary CSV path. Defaults to "
            "<root>/tabulax_conversion_summary.csv when not in dry-run mode."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    dataset_dirs = discover_dataset_dirs(root)

    if not dataset_dirs:
        print(
            "No folders ending in _f025, _f050, _f075, or _f100 "
            f"were found under {root}.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(dataset_dirs)} overlap dataset folder(s) under {root}")

    results: list[ConversionResult] = []
    failures = 0

    for dataset_dir in dataset_dirs:
        try:
            result = convert_dataset(
                dataset_dir,
                source_column_override=args.source_column,
                target_column_override=args.target_column,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            results.append(result)
            print(
                f"[OK] {result.dataset}: "
                f"{result.source_column} -> {result.target_column}, "
                f"GT pairs={result.gt_pairs}, status={result.status}"
            )
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {dataset_dir.name}: {exc}", file=sys.stderr)

            if not args.continue_on_error:
                print(
                    "Stopped after the first error. Use --continue-on-error "
                    "to process the remaining folders.",
                    file=sys.stderr,
                )
                return 1

    if not args.dry_run:
        summary_path = (
            args.summary.resolve()
            if args.summary
            else root / "tabulax_conversion_summary.csv"
        )
        write_summary_csv(summary_path, results)
        print(f"Summary written to: {summary_path}")

    print(
        f"Finished: successful={len(results)}, failed={failures}, "
        f"total={len(dataset_dirs)}"
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
