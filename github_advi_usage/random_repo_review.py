#!/usr/bin/env python3
"""Randomly review already-classified repositories.

This is a quality-control review tool for repo-level classifications. It samples
from `repo_level_summary.csv`, shows the repo classification and supporting
call-site evidence, and records review decisions without changing the analysis
classification rules. If a sampled repo is wrong, use the regular manual-review
workflow to update the underlying call-site classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from pathlib import Path

from .build_vi_github_analysis import RANDOM_REPO_REVIEW_FIELDS, build_random_repo_review_actions, write_csv
from .db import utc_now


CHOICES = {
    "1": ("correct", ""),
    "2": ("incorrect", "default"),
    "3": ("incorrect", "adjusted_mc"),
    "4": ("incorrect", "adjusted_optimizer"),
    "5": ("incorrect", "adjusted_both"),
    "6": ("incorrect", "no_vi_fit"),
    "7": ("needs_followup", ""),
    "8": ("skip", ""),
    "0": ("stop", ""),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vi_github_random_repo_review_actions (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_full_name TEXT NOT NULL,
            repo_url TEXT,
            packages_used TEXT,
            repo_adjusts_anything TEXT,
            repo_adjusts_optimizer TEXT,
            repo_adjusts_mc TEXT,
            num_true_positive_call_sites INTEGER,
            sampled_call_site_ids TEXT NOT NULL,
            decision TEXT NOT NULL,
            suggested_correct_classification TEXT,
            notes TEXT,
            seed TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def export_actions(conn: sqlite3.Connection, reports_dir: Path) -> None:
    write_csv(
        reports_dir / "random_repo_review_actions.csv",
        build_random_repo_review_actions(conn),
        RANDOM_REPO_REVIEW_FIELDS,
    )


def reviewed_repos(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT repo_full_name FROM vi_github_random_repo_review_actions WHERE decision != 'skip'"
        )
    }


def repo_label(row: dict[str, str]) -> str:
    if row["adjusts_optimizer"] == "True" and row["adjusts_mc"] == "True":
        return "adjusted_both"
    if row["adjusts_optimizer"] == "True":
        return "adjusted_optimizer"
    if row["adjusts_mc"] == "True":
        return "adjusted_mc"
    return "default"


def filter_repos(rows: list[dict[str, str]], args: argparse.Namespace, already_reviewed: set[str]) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if not args.include_reviewed and row["repo_full_name"] in already_reviewed:
            continue
        if args.package and args.package not in row["packages_used"].split(";"):
            continue
        label = repo_label(row)
        if args.only_default and label != "default":
            continue
        if args.only_adjusted and label not in {"adjusted_mc", "adjusted_optimizer", "adjusted_both"}:
            continue
        if args.exclude_stan and row["packages_used"] == "stan":
            continue
        filtered.append(row)
    return filtered


def print_choices() -> None:
    print(
        """
Choose a review decision:
  1. correct as classified
  2. incorrect: should be default
  3. incorrect: should be adjusted_mc
  4. incorrect: should be adjusted_optimizer
  5. incorrect: should be adjusted_both
  6. incorrect: should be excluded/no_vi_fit
  7. needs follow-up
  8. skip this sampled repo
  0. stop and save progress
"""
    )


def sample_call_sites(row: dict[str, str], rng: random.Random, max_call_sites: int) -> list[str]:
    call_site_ids = [item for item in row["supporting_call_site_ids"].split(";") if item]
    if len(call_site_ids) <= max_call_sites:
        return call_site_ids
    return rng.sample(call_site_ids, max_call_sites)


def print_repo(
    row: dict[str, str],
    call_rows_by_id: dict[str, dict[str, str]],
    sampled_call_site_ids: list[str],
    index: int,
    total: int,
    remaining_pool: int,
) -> None:
    print("\n" + "=" * 88)
    print(f"Random repo review {index} of {total} | remaining eligible pool: {remaining_pool}")
    print(f"repo: {row['repo_full_name']}")
    print(f"url: {row['repo_url']}")
    print(f"packages: {row['packages_used']}")
    print(f"repo classification: {repo_label(row)}")
    print(
        "adjusts_anything={adjusts_anything} | adjusts_optimizer={adjusts_optimizer} | adjusts_mc={adjusts_mc}".format(
            **row
        )
    )
    print(f"num_true_positive_call_sites: {row['num_true_positive_call_sites']}")
    print("\nSampled supporting call sites:")
    for call_site_id in sampled_call_site_ids:
        call = call_rows_by_id.get(call_site_id)
        print(f"- {call_site_id}")
        if not call:
            continue
        print(f"  link: {call['github_permalink']}")
        print(f"  call classification: {call['final_classification']} | {call['package']} / {call['call_kind']}")
        print(f"  snippet: {call['actual_call_snippet']}")
        print(f"  reason: {call['classification_reason']}")


def write_action(
    conn: sqlite3.Connection,
    row: dict[str, str],
    sampled_call_site_ids: list[str],
    decision: str,
    suggested_correct_classification: str,
    notes: str,
    seed: str,
) -> None:
    conn.execute(
        """
        INSERT INTO vi_github_random_repo_review_actions(
            repo_full_name, repo_url, packages_used, repo_adjusts_anything,
            repo_adjusts_optimizer, repo_adjusts_mc, num_true_positive_call_sites,
            sampled_call_site_ids, decision, suggested_correct_classification,
            notes, seed, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["repo_full_name"],
            row["repo_url"],
            row["packages_used"],
            row["adjusts_anything"],
            row["adjusts_optimizer"],
            row["adjusts_mc"],
            int(row["num_true_positive_call_sites"] or 0),
            json.dumps(sampled_call_site_ids),
            decision,
            suggested_correct_classification,
            notes,
            seed,
            utc_now(),
        ),
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", default="")
    parser.add_argument("--max-call-sites", type=int, default=3)
    parser.add_argument("--package", choices=["pymc", "pyro", "numpyro", "tensorflow_probability", "stan"])
    parser.add_argument("--only-default", action="store_true")
    parser.add_argument("--only-adjusted", action="store_true")
    parser.add_argument("--exclude-stan", action="store_true")
    parser.add_argument("--include-reviewed", action="store_true")
    args = parser.parse_args()

    if args.only_default and args.only_adjusted:
        raise SystemExit("Use at most one of --only-default and --only-adjusted.")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ensure_table(conn)
    seed = args.seed or utc_now()
    rng = random.Random(seed)

    repo_rows = read_csv(args.reports_dir / "repo_level_summary.csv")
    call_rows = read_csv(args.reports_dir / "call_site_evidence.csv")
    call_rows_by_id = {row["call_site_id"]: row for row in call_rows}
    pool = filter_repos(repo_rows, args, reviewed_repos(conn))
    rng.shuffle(pool)

    if not pool:
        print("No eligible repos remain for this random-review filter.")
        export_actions(conn, args.reports_dir)
        conn.close()
        return

    total = min(args.sample_size, len(pool))
    reviewed_count = 0
    try:
        for row in pool[:total]:
            sampled_call_site_ids = sample_call_sites(row, rng, args.max_call_sites)
            print_repo(row, call_rows_by_id, sampled_call_site_ids, reviewed_count + 1, total, len(pool) - reviewed_count)
            print_choices()
            while True:
                choice = input("Enter choice number: ").strip()
                if choice in CHOICES:
                    break
                print("Unrecognized choice; please enter one of the listed numbers.")
            decision, suggested = CHOICES[choice]
            if decision == "stop":
                print("Stopping. Previous random-review decisions have been saved.")
                break
            notes = input("Optional note: ").strip()
            write_action(conn, row, sampled_call_site_ids, decision, suggested, notes, str(seed))
            reviewed_count += 1
    finally:
        export_actions(conn, args.reports_dir)
        conn.close()


if __name__ == "__main__":
    main()
