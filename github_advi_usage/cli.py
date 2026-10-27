"""Unified CLI: `fetch` (capability 1) and `analyze` (capability 2), plus
`all` to run both in sequence. See the repo README for the full picture."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from . import build_vi_github_analysis, verify_vi_github_analysis
from .api import ApiClient
from .config import Settings, settings
from .db import connect, init_db
from .github_advi import (
    EXHAUSTIVE_GITHUB_ADVI_QUERIES,
    classify_github_advi_calls,
    fetch_github_advi_report_tables,
    scan_all_pending_github_advi_files,
    search_github_advi_code_exhaustive,
)


def add_fetch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", action="append", help="GitHub code-search query. May be repeated.")
    parser.add_argument("--search-interval", type=float, default=6.2)
    parser.add_argument("--max-size", type=int, default=1_000_000)
    parser.add_argument("--min-size-window", type=int, default=1)
    parser.add_argument("--scan-batch-size", type=int, default=100)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--skip-scan", action="store_true")


def add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, help="Limit how many new call sites get classified.")
    parser.add_argument("--refresh", action="store_true", help="Reclassify call sites that already have a classification.")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Defaults to <data_dir>/reports.")
    parser.add_argument("--skip-verify", action="store_true")

    output_group = parser.add_argument_group(
        "output selection",
        "With none of these, writes the core set: call_site_evidence, default_advi_repos, "
        "adjusted_advi_repos, repo_level_summary, overview. --comprehensive writes everything.",
    )
    output_group.add_argument("--default-only", action="store_true", help="Write only default_advi_repos.csv.")
    output_group.add_argument("--adjusted-only", action="store_true", help="Write only adjusted_advi_repos.csv.")
    output_group.add_argument("--overview", action="store_true", help="Write only overview.csv.")
    output_group.add_argument("--call-sites-only", action="store_true", help="Write only call_site_evidence.csv.")
    output_group.add_argument("--repo-summary-only", action="store_true", help="Write only repo_level_summary.csv.")
    output_group.add_argument(
        "--include-review-queue",
        action="store_true",
        help="Also write needs_review_queue.csv, unresolved_parameter_queue.csv, and "
        "parameter_resolution_evidence.csv (the working files manual-review/deep-resolve use).",
    )
    output_group.add_argument(
        "--comprehensive",
        action="store_true",
        help="Write every report, including the review-queue and audit files.",
    )
    output_group.add_argument(
        "--keep-existing",
        action="store_true",
        help="Don't delete reports-dir files that this run didn't request (default: it does, so "
        "reports-dir always matches exactly what you asked for).",
    )


def run_fetch(conn: sqlite3.Connection, client: ApiClient, args: argparse.Namespace) -> tuple[int, int]:
    """Capability (1): search + scan GitHub to (re)build the raw database.

    Exhaustive GitHub Code Search for every query, then fetches and parses
    the matched files into github_advi_calls rows. Requires GITHUB_TOKEN. No
    CSVs are written here -- that's `analyze`'s job, run against the
    database this leaves behind.
    """
    queries = args.query or EXHAUSTIVE_GITHUB_ADVI_QUERIES
    hit_count = search_github_advi_code_exhaustive(
        conn,
        client,
        queries=queries,
        interval=args.search_interval,
        max_size=args.max_size,
        min_size_window=args.min_size_window,
    )
    scan_count = 0
    if not args.skip_scan:
        scan_count = scan_all_pending_github_advi_files(
            conn,
            client,
            batch_size=args.scan_batch_size,
            interval=args.scan_interval,
        )
    print(f"Saved {hit_count} GitHub code-search file hits from exhaustive search")
    print(f"Scanned {scan_count} GitHub files")
    return hit_count, scan_count


def run_analyze(conn: sqlite3.Connection, cfg: Settings, args: argparse.Namespace) -> dict:
    """Capability (2): classify a saved database and write the requested
    reports/*.csv outputs. No network access: classification is rule-based
    over data already in the database."""
    reports_dir = args.reports_dir or (cfg.data_dir / "reports")

    count = classify_github_advi_calls(conn, limit=args.limit, refresh=args.refresh)
    print(f"Classified {count} GitHub ADVI/VI call rows")

    report_tables = fetch_github_advi_report_tables(conn)
    written = build_vi_github_analysis.build_analysis(
        conn,
        report_tables,
        reports_dir,
        default_only=args.default_only,
        adjusted_only=args.adjusted_only,
        overview=args.overview,
        comprehensive=args.comprehensive,
        call_sites_only=args.call_sites_only,
        repo_summary_only=args.repo_summary_only,
        include_review_queue=args.include_review_queue,
        keep_existing=args.keep_existing,
    )
    for name, rows in sorted(written.items()):
        print(f"- {name}.csv: {len(rows)} row(s)")

    if not args.skip_verify:
        errors = verify_vi_github_analysis.verify_analysis(reports_dir, requested_outputs=set(written))
        if errors:
            print("Verification failed:")
            for error in errors:
                print(f"- {error}")
            raise SystemExit(1)
        print("Verified output consistency.")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce (fetch) and/or analyze a GitHub ADVI/VI usage database."
    )
    parser.add_argument("--db", help="SQLite database path. Defaults to data/github-vi-software.sqlite.")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_parser = sub.add_parser("fetch", help="Capability (1): reproduce the sqlite database from GitHub.")
    add_fetch_arguments(fetch_parser)

    analyze_parser = sub.add_parser("analyze", help="Capability (2): analyze a saved database and write CSV outputs.")
    add_analyze_arguments(analyze_parser)

    all_parser = sub.add_parser("all", help="Run fetch, then analyze.")
    add_fetch_arguments(all_parser)
    add_analyze_arguments(all_parser)

    args = parser.parse_args()
    cfg = settings(Path(args.db) if args.db else None)
    conn = connect(cfg.db_path)
    init_db(conn)

    if args.command in ("fetch", "all"):
        client = ApiClient(cfg)
        run_fetch(conn, client, args)

    if args.command in ("analyze", "all"):
        run_analyze(conn, cfg, args)


if __name__ == "__main__":
    main()
