#!/usr/bin/env python3
"""Verify internal consistency of the reports/*.csv outputs `analyze` wrote."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from .build_vi_github_analysis import (
    ADJUSTED_SETTING_FOR_CLASSIFICATION,
    ALL_OUTPUTS,
    CALL_SITE_EVIDENCE_FIELDS,
    FILENAME_FOR_OUTPUT,
    NA,
    NEEDS_REVIEW_QUEUE_FIELDS,
    OVERVIEW_FIELDS,
    RANDOM_REPO_REVIEW_FIELDS,
    REPO_LEVEL_SUMMARY_FIELDS,
)

COLUMNS_FOR_OUTPUT = {
    "call_site_evidence": CALL_SITE_EVIDENCE_FIELDS,
    "default_advi_repos": ["repo_link", "filename", "line_number"],
    "adjusted_advi_repos": ["repo_link", "filename", "line_number", "adjusted_setting"],
    "repo_level_summary": REPO_LEVEL_SUMMARY_FIELDS,
    "overview": OVERVIEW_FIELDS,
    "needs_review_queue": NEEDS_REVIEW_QUEUE_FIELDS,
    "random_repo_review_actions": RANDOM_REPO_REVIEW_FIELDS,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def verify_files(out_dir: Path, requested_outputs: set[str], errors: list[str]) -> None:
    for name in requested_outputs:
        path = out_dir / FILENAME_FOR_OUTPUT[name]
        if not path.exists():
            fail(errors, f"Missing requested output: {FILENAME_FOR_OUTPUT[name]}")


def verify_columns(out_dir: Path, name: str, errors: list[str]) -> None:
    expected = COLUMNS_FOR_OUTPUT.get(name)
    if expected is None:
        return
    path = out_dir / FILENAME_FOR_OUTPUT[name]
    with path.open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    if fieldnames != expected:
        fail(errors, f"{FILENAME_FOR_OUTPUT[name]} has unexpected columns: {fieldnames}")


def verify_call_site_evidence(out_dir: Path, errors: list[str]) -> list[dict[str, str]]:
    rows = read_csv(out_dir / FILENAME_FOR_OUTPUT["call_site_evidence"])
    na_fields = ["call_kind", "actual_call_snippet", "extracted_function", "extracted_arguments", "classification_reason"]
    for row in rows:
        expected_id = f"{row['repo_full_name']}:{row['commit_sha']}:{row['path']}:{row['line_number']}"
        if row["call_site_id"] != expected_id:
            fail(errors, f"Bad call_site_id: {row['call_site_id']} != {expected_id}")
        if row["manual_review_status"] not in {"yes", "no"}:
            fail(errors, f"Bad manual_review_status for {row['call_site_id']}: {row['manual_review_status']}")
        if row["why_in_query_results"] != "true_positive":
            for field in na_fields:
                if row[field] != NA:
                    fail(errors, f"Excluded row has non-NA {field}: {row['call_site_id']}")
    return rows


def verify_default_and_adjusted(out_dir: Path, requested_outputs: set[str], call_rows: list[dict[str, str]], errors: list[str]) -> None:
    if "default_advi_repos" in requested_outputs:
        default_rows = read_csv(out_dir / FILENAME_FOR_OUTPUT["default_advi_repos"])
        expected = Counter(
            (f"https://github.com/{row['repo_full_name']}", row["path"], row["line_number"])
            for row in call_rows
            if row["final_classification"] == "default"
        )
        actual = Counter((row["repo_link"], row["filename"], row["line_number"]) for row in default_rows)
        if actual != expected:
            fail(
                errors,
                f"default_advi_repos.csv ({sum(actual.values())} rows) does not match the 'default' "
                f"call sites in call_site_evidence.csv ({sum(expected.values())} rows)",
            )

    if "adjusted_advi_repos" in requested_outputs:
        adjusted_rows = read_csv(out_dir / FILENAME_FOR_OUTPUT["adjusted_advi_repos"])
        expected = Counter(
            (f"https://github.com/{row['repo_full_name']}", row["path"], row["line_number"], setting)
            for row in call_rows
            if (setting := ADJUSTED_SETTING_FOR_CLASSIFICATION.get(row["final_classification"])) is not None
        )
        actual = Counter(
            (row["repo_link"], row["filename"], row["line_number"], row["adjusted_setting"]) for row in adjusted_rows
        )
        if actual != expected:
            fail(
                errors,
                f"adjusted_advi_repos.csv ({sum(actual.values())} rows) does not match the adjusted "
                f"call sites in call_site_evidence.csv ({sum(expected.values())} rows)",
            )


def verify_repo_summary(out_dir: Path, errors: list[str]) -> None:
    for row in read_csv(out_dir / FILENAME_FOR_OUTPUT["repo_level_summary"]):
        adjusts_any = row["adjusts_anything"] == "True"
        if adjusts_any != (row["adjusts_optimizer"] == "True" or row["adjusts_mc"] == "True"):
            fail(errors, f"adjusts_anything mismatch for {row['repo_full_name']}")


def verify_random_repo_review(out_dir: Path, errors: list[str]) -> None:
    valid_decisions = {"correct", "incorrect", "needs_followup", "skip"}
    for row in read_csv(out_dir / FILENAME_FOR_OUTPUT["random_repo_review_actions"]):
        if row["decision"] not in valid_decisions:
            fail(errors, f"Bad random repo review decision for {row['repo_full_name']}: {row['decision']}")


def verify_analysis(out_dir: Path, requested_outputs: set[str] | None = None) -> list[str]:
    """Check the reports actually written in `out_dir` for internal
    consistency. `requested_outputs` should be the output names `analyze`
    was asked to write (a subset of build_vi_github_analysis.ALL_OUTPUTS);
    defaults to all of them if not given."""
    requested_outputs = requested_outputs if requested_outputs is not None else set(ALL_OUTPUTS)
    errors: list[str] = []
    verify_files(out_dir, requested_outputs, errors)
    if errors:
        return errors

    for name in sorted(requested_outputs):
        verify_columns(out_dir, name, errors)

    call_rows: list[dict[str, str]] = []
    if "call_site_evidence" in requested_outputs:
        call_rows = verify_call_site_evidence(out_dir, errors)
    if call_rows:
        verify_default_and_adjusted(out_dir, requested_outputs, call_rows, errors)
    if "repo_level_summary" in requested_outputs:
        verify_repo_summary(out_dir, errors)
    if "random_repo_review_actions" in requested_outputs:
        verify_random_repo_review(out_dir, errors)
    return errors


def detected_outputs(out_dir: Path) -> set[str]:
    """The outputs present in `out_dir`, for verifying a directory without
    knowing which flags produced it."""
    return {name for name in ALL_OUTPUTS if (out_dir / FILENAME_FOR_OUTPUT[name]).exists()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", dest="reports_dir", type=Path, required=True)
    args = parser.parse_args()

    present = detected_outputs(args.reports_dir)
    if not present:
        print(f"No recognized report files found in {args.reports_dir}")
        sys.exit(1)
    errors = verify_analysis(args.reports_dir, requested_outputs=present)

    if errors:
        print("Verification failed:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("Verification passed.")


if __name__ == "__main__":
    main()
