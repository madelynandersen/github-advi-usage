#!/usr/bin/env python3
"""Interactive command-line review for VI GitHub analysis rows.

The review loop reads the current generated review queues, presents one item at
a time, records the user's decision in SQLite, and optionally regenerates all
reports/analysis CSVs after each saved decision.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

from . import build_vi_github_analysis
from .db import connect, init_db, utc_now
from .github_advi import classify_github_advi_calls, fetch_github_advi_report_tables


CHOICES = {
    "1": ("default", "true_positive"),
    "2": ("adjusted_mc", "true_positive"),
    "3": ("adjusted_optimizer", "true_positive"),
    "4": ("adjusted_both", "true_positive"),
    "5": ("no_vi_fit", "false_positive"),
    "6": ("method_repo_code_itself", "false_positive"),
    "7": ("import_only", "false_positive"),
    "8": ("comment_only", "false_positive"),
    "9": ("skip", "skip"),
    "10": ("delete", "false_positive"),
    "0": ("stop", "stop"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_review_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS vi_github_manual_review_actions (
            call_site_id TEXT PRIMARY KEY,
            call_id INTEGER,
            queue TEXT NOT NULL,
            action TEXT NOT NULL,
            final_classification TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def skipped_call_site_ids(conn: sqlite3.Connection) -> set[str]:
    ensure_review_table(conn)
    return {
        row[0]
        for row in conn.execute(
            "SELECT call_site_id FROM vi_github_manual_review_actions WHERE action='skip'"
        )
    }


def parse_call_site_id(call_site_id: str) -> tuple[str, str, str, str]:
    repo, commit_sha, rest = call_site_id.split(":", 2)
    path, line_number = rest.rsplit(":", 1)
    return repo, commit_sha, path, line_number


def lookup_call(conn: sqlite3.Connection, call_site_id: str) -> sqlite3.Row | None:
    repo, _commit_sha, path, line_number = parse_call_site_id(call_site_id)
    return conn.execute(
        """
        SELECT c.*, k.usage_status, k.is_false_positive, k.category, k.reason
        FROM github_advi_calls c
        LEFT JOIN github_advi_call_classifications k ON k.call_id = c.id
        WHERE c.repo_full_name=? AND c.path=? AND c.line_number=?
        ORDER BY c.id
        LIMIT 1
        """,
        (repo, path, int(line_number)),
    ).fetchone()


def write_action(
    conn: sqlite3.Connection,
    *,
    call_site_id: str,
    call_id: int | None,
    queue: str,
    action: str,
    final_classification: str | None,
    notes: str,
) -> None:
    now = utc_now()
    existing = conn.execute(
        "SELECT created_at FROM vi_github_manual_review_actions WHERE call_site_id=?",
        (call_site_id,),
    ).fetchone()
    created_at = existing[0] if existing else now
    conn.execute(
        """
        INSERT OR REPLACE INTO vi_github_manual_review_actions(
            call_site_id, call_id, queue, action, final_classification, notes,
            created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (call_site_id, call_id, queue, action, final_classification, notes, created_at, now),
    )
    conn.commit()


def set_classification(
    conn: sqlite3.Connection,
    call_id: int,
    *,
    usage_status: str,
    is_false_positive: bool,
    category: str,
    reason: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO github_advi_call_classifications(
            call_id, usage_status, is_false_positive, category, confidence,
            reason, classifier_version, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call_id,
            usage_status,
            1 if is_false_positive else 0,
            category,
            1.0,
            reason,
            "manual_review_cli",
            utc_now(),
        ),
    )
    conn.commit()


def upsert_override(
    conn: sqlite3.Connection,
    call: sqlite3.Row,
    *,
    optimizer: str | None,
    loss: str | None,
    sample_or_mc: str | None,
    reason: str,
) -> None:
    if not any(value not in (None, "") for value in [optimizer, loss, sample_or_mc]):
        return
    now = utc_now()
    existing = conn.execute(
        """
        SELECT created_at FROM github_advi_parameter_overrides
        WHERE repo_full_name=? AND path=? AND line_number=?
        """,
        (call["repo_full_name"], call["path"], call["line_number"]),
    ).fetchone()
    created_at = existing[0] if existing else now
    conn.execute(
        """
        INSERT OR REPLACE INTO github_advi_parameter_overrides(
            repo_full_name, path, line_number,
            override_optimizer_expression, override_loss_expression,
            override_sample_or_mc_expression, reason, source, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call["repo_full_name"],
            call["path"],
            call["line_number"],
            optimizer or None,
            loss or None,
            sample_or_mc or None,
            reason,
            "manual_review_cli",
            created_at,
            now,
        ),
    )
    conn.commit()


def default_values_for_package(package: str) -> tuple[str, str, str]:
    if package == "pymc":
        return "adagrad_window", "", "None"
    if package == "tensorflow_probability":
        return "Adam", "", "1"
    if package in {"pyro", "numpyro", "pyro_or_numpyro"}:
        return "Adam", "Trace_ELBO()", "1"
    if package == "stan":
        return "", "", "1"
    return "", "", ""


def prompt_with_default(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def collect_override_values(package: str, final_classification: str) -> tuple[str, str, str]:
    default_optimizer, default_loss, default_sample = default_values_for_package(package)
    optimizer = ""
    loss = ""
    sample = ""

    if final_classification == "default":
        optimizer = prompt_with_default("Optimizer/default expression", default_optimizer)
        loss = prompt_with_default("Loss/default expression", default_loss)
        sample = prompt_with_default("MC/sample/default expression", default_sample)
    elif final_classification == "adjusted_mc":
        optimizer = prompt_with_default("Optimizer/default expression", default_optimizer)
        loss = prompt_with_default("Loss expression if useful", default_loss)
        sample = prompt_with_default("Adjusted MC/sample value", "")
    elif final_classification == "adjusted_optimizer":
        optimizer = prompt_with_default("Adjusted optimizer expression", "")
        loss = prompt_with_default("Loss/default expression", default_loss)
        sample = prompt_with_default("MC/sample/default expression", default_sample)
    elif final_classification == "adjusted_both":
        optimizer = prompt_with_default("Adjusted optimizer expression", "")
        loss = prompt_with_default("Loss expression if useful", default_loss)
        sample = prompt_with_default("Adjusted MC/sample value", "")
    return optimizer, loss, sample


def load_entries(
    reports_dir: Path,
    queue: str,
    include_skipped: bool,
    conn: sqlite3.Connection,
    *,
    also_exclude: set[str] = frozenset(),
) -> list[dict[str, object]]:
    """Pending review entries, excluding skipped call sites and `also_exclude`
    (call sites already handled this session, which matters when the queue
    CSVs are not rebuilt after each decision). The "unresolved" queue is the
    unresolved rows of parameter_resolution_evidence.csv."""
    skipped = skipped_call_site_ids(conn) if not include_skipped else set()
    excluded = skipped | also_exclude
    entries: list[dict[str, object]] = []

    if queue in {"both", "unresolved"}:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(reports_dir / "parameter_resolution_evidence.csv"):
            if row.get("resolution_status") == "unresolved":
                grouped[row["call_site_id"]].append(row)
        for call_site_id, rows in grouped.items():
            if call_site_id not in excluded:
                entries.append({"queue": "unresolved", "call_site_id": call_site_id, "rows": rows})

    if queue in {"both", "needs_review"}:
        for row in read_csv(reports_dir / "needs_review_queue.csv"):
            if row["call_site_id"] not in excluded:
                entries.append({"queue": "needs_review", "call_site_id": row["call_site_id"], "row": row})

    return entries


def print_entry(entry: dict[str, object], call: sqlite3.Row | None, index: int, total: int) -> None:
    print("\n" + "=" * 88)
    print(f"Review item {index} of {total} | queue={entry['queue']}")
    print(f"call_site_id: {entry['call_site_id']}")
    if call:
        print(f"repo: {call['repo_full_name']}")
        print(f"path: {call['path']}:{call['line_number']}")
        print(f"link: {call['html_url']}#L{call['line_number']}")
        print(f"package/call_kind: {call['package']} / {call['call_kind']}")
        print(f"function/method: {call['function_name']} / {call['method'] or ''}")
        print(f"current status: {call['usage_status'] or 'unclassified'} | {call['category'] or ''}")
        print(f"reason: {call['reason'] or ''}")
        print("snippet:")
        print((call["source_snippet"] or "").strip())
        print(f"keyword_arguments: {call['keyword_arguments']}")
        print(f"positional_arguments: {call['positional_arguments']}")
    if entry["queue"] == "unresolved":
        print("\nUnresolved parameter evidence:")
        for row in entry["rows"]:  # type: ignore[index]
            print(f"- {row['parameter_name']}: raw={row['raw_expression_at_call']} | link={row['evidence_permalink']}")
    else:
        row = entry["row"]  # type: ignore[assignment]
        print("\nNeeds-review details:")
        print(f"what_needs_review: {row['what_needs_review']}")
        print(f"current_reason: {row['current_reason']}")


def print_choices() -> None:
    print(
        """
Choose a label:
  1. default
  2. adjusted_mc
  3. adjusted_optimizer
  4. adjusted_both
  5. no_vi_fit
  6. method_repo_code_itself
  7. import_only
  8. comment_only
  9. skip this entry
 10. delete/remove this entry from analysis
  0. stop and save progress
"""
    )


def refresh_outputs(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    """Re-classify and rebuild the review-queue CSVs from the current db
    state, so manual review always works off up-to-date data."""
    if args.no_rebuild_each:
        return
    classify_github_advi_calls(conn, limit=None, refresh=False)
    report_tables = fetch_github_advi_report_tables(conn)
    build_vi_github_analysis.build_analysis(
        conn, report_tables, args.reports_dir, include_review_queue=True, keep_existing=True
    )


def apply_decision(
    conn: sqlite3.Connection,
    entry: dict[str, object],
    call: sqlite3.Row,
    final_classification: str,
    kind: str,
) -> None:
    call_id = int(call["id"])
    notes = prompt_with_default("Optional note", "")
    if kind == "skip":
        write_action(
            conn,
            call_site_id=str(entry["call_site_id"]),
            call_id=call_id,
            queue=str(entry["queue"]),
            action="skip",
            final_classification=None,
            notes=notes,
        )
        return

    if kind == "false_positive":
        category_by_label = {
            "no_vi_fit": "manual_no_vi_fit",
            "method_repo_code_itself": "upstream_library_self_reference",
            "import_only": "manual_import_only",
            "comment_only": "manual_comment_only",
            "delete": "manual_removed",
        }
        category = category_by_label.get(final_classification, "manual_no_vi_fit")
        reason = notes or f"Manual review CLI marked as {final_classification}."
        set_classification(
            conn,
            call_id,
            usage_status="likely_false_positive",
            is_false_positive=True,
            category=category,
            reason=reason,
        )
        write_action(
            conn,
            call_site_id=str(entry["call_site_id"]),
            call_id=call_id,
            queue=str(entry["queue"]),
            action="delete" if final_classification == "delete" else "classify",
            final_classification="no_vi_fit",
            notes=reason,
        )
        return

    reason = notes or f"Manual review CLI marked as {final_classification}."
    set_classification(
        conn,
        call_id,
        usage_status="likely_true_positive",
        is_false_positive=False,
        category="manual_user_review",
        reason=reason,
    )
    optimizer, loss, sample = collect_override_values(call["package"], final_classification)
    upsert_override(conn, call, optimizer=optimizer, loss=loss, sample_or_mc=sample, reason=reason)
    write_action(
        conn,
        call_site_id=str(entry["call_site_id"]),
        call_id=call_id,
        queue=str(entry["queue"]),
        action="classify",
        final_classification=final_classification,
        notes=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--queue", choices=["both", "needs_review", "unresolved"], default="both")
    parser.add_argument("--include-skipped", action="store_true")
    parser.add_argument("--no-rebuild-each", action="store_true", help="Save DB updates but rebuild reports only when the script exits.")
    args = parser.parse_args()

    conn = connect(args.db)
    init_db(conn)
    ensure_review_table(conn)

    handled_this_session: set[str] = set()
    try:
        refresh_outputs(conn, args)
        while True:
            entries = load_entries(
                args.reports_dir, args.queue, args.include_skipped, conn, also_exclude=handled_this_session
            )
            if not entries:
                print("No review entries remain for this queue.")
                break
            entry = entries[0]
            call = lookup_call(conn, str(entry["call_site_id"]))
            print_entry(entry, call, 1, len(entries))
            print_choices()
            choice = input("Enter choice number: ").strip()
            if choice not in CHOICES:
                print("Unrecognized choice; please enter one of the listed numbers.")
                continue
            final_classification, kind = CHOICES[choice]
            if kind == "stop":
                print("Stopping. Previous reviews have been saved.")
                break
            if call is None:
                print("Could not find this call in the database; skipping for now.")
                continue
            apply_decision(conn, entry, call, final_classification, kind)
            handled_this_session.add(str(entry["call_site_id"]))
            refresh_outputs(conn, args)
    finally:
        if args.no_rebuild_each:
            refresh_outputs(conn, argparse.Namespace(**{**vars(args), "no_rebuild_each": False}))
        conn.close()


if __name__ == "__main__":
    main()
