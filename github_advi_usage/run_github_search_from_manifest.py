#!/usr/bin/env python3
"""Run the GitHub VI search with the queries in a saved query manifest
(e.g. github_query_manifest.csv)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .api import ApiClient
from .cli import run_fetch
from .config import settings
from .db import connect, init_db


def read_queries(manifest: Path) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    queries = [row["query_string"] for row in rows if row.get("query_string")]
    if not queries:
        raise SystemExit(f"No query_string values found in {manifest}")
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--search-interval", type=float, default=6.2)
    parser.add_argument("--max-size", type=int, default=1_000_000)
    parser.add_argument("--min-size-window", type=int, default=1)
    parser.add_argument("--scan-batch-size", type=int, default=100)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--skip-scan", action="store_true")
    args = parser.parse_args()

    args.query = read_queries(args.manifest)
    print(f"Running {len(args.query)} GitHub queries from {args.manifest}")

    cfg = settings(args.db)
    conn = connect(cfg.db_path)
    init_db(conn)
    client = ApiClient(cfg)
    run_fetch(conn, client, args)


if __name__ == "__main__":
    main()
