"""Classify a Python source file offline, with exactly the pipeline's rules.

Runs the pipeline's own call extraction, classification, and report-building
steps on an in-memory database, so the labels match what `analyze` would
produce for the same file. Used by example_query_and_classification.ipynb.
"""

from __future__ import annotations

import sqlite3

from .build_vi_github_analysis import build_call_site_evidence, build_needs_review_queue
from .db import init_db
from .github_advi import (
    classify_github_advi_calls,
    fetch_github_advi_report_tables,
    find_advi_calls,
    github_file_id,
    replace_github_advi_calls,
    save_github_search_item,
)


def classify_source(
    source: str,
    *,
    repo_full_name: str = "example/repo",
    path: str = "model.py",
    commit_sha: str = "HEAD",
) -> dict[str, list[dict[str, object]]]:
    """Return the call_site_evidence.csv and needs_review_queue.csv rows that
    `analyze` would write for `source` saved at `repo_full_name`/`path`.

    The path matters: calls in test, tutorial, or vendored paths are
    classified differently from the same call in project code.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    save_github_search_item(
        conn,
        {
            "repository": {"full_name": repo_full_name},
            "path": path,
            "html_url": f"https://github.com/{repo_full_name}/blob/{commit_sha}/{path}",
        },
    )
    replace_github_advi_calls(conn, github_file_id(repo_full_name, path), find_advi_calls(source))
    classify_github_advi_calls(conn)
    tables = fetch_github_advi_report_tables(conn)
    conn.close()
    return {
        "call_site_evidence": build_call_site_evidence(tables["calls"], tables["relevant"], set()),
        "needs_review_queue": build_needs_review_queue(tables["calls"]),
    }
