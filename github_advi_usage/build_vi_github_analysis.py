#!/usr/bin/env python3
"""Build the GitHub ADVI/VI analysis CSVs.

Takes the row tables queried by github_advi.fetch_github_advi_report_tables()
and writes the requested reports/*.csv outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NA = "NA"

CORE_OUTPUTS = ("call_site_evidence", "default_advi_repos", "adjusted_advi_repos", "repo_level_summary", "overview")
QUEUE_OUTPUTS = ("needs_review_queue", "unresolved_parameter_queue", "parameter_resolution_evidence")
AUDIT_OUTPUTS = ("duplicate_repo_clusters", "random_repo_review_actions")
ALL_OUTPUTS = CORE_OUTPUTS + QUEUE_OUTPUTS + AUDIT_OUTPUTS

FILENAME_FOR_OUTPUT = {
    "call_site_evidence": "call_site_evidence.csv",
    "default_advi_repos": "default_advi_repos.csv",
    "adjusted_advi_repos": "adjusted_advi_repos.csv",
    "repo_level_summary": "repo_level_summary.csv",
    "overview": "overview.csv",
    "needs_review_queue": "needs_review_queue.csv",
    "unresolved_parameter_queue": "unresolved_parameter_queue.csv",
    "parameter_resolution_evidence": "parameter_resolution_evidence.csv",
    "duplicate_repo_clusters": "duplicate_repo_clusters.csv",
    "random_repo_review_actions": "random_repo_review_actions.csv",
}

ADJUSTED_SETTING_FOR_CLASSIFICATION = {
    "adjusted_optimizer": "optimizer",
    "adjusted_mc": "mc",
    "adjusted_both": "both",
}

RANDOM_REPO_REVIEW_FIELDS = [
    "review_id",
    "repo_full_name",
    "repo_url",
    "packages_used",
    "repo_adjusts_anything",
    "repo_adjusts_optimizer",
    "repo_adjusts_mc",
    "num_true_positive_call_sites",
    "sampled_call_site_ids",
    "decision",
    "suggested_correct_classification",
    "notes",
    "seed",
    "created_at",
]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def commit_from_url(url: str) -> str:
    match = re.search(r"/blob/([^/]+)/", url or "")
    return match.group(1) if match else NA


def source_link(url: str, line_number: object) -> str:
    if not url:
        return NA
    return url if "#L" in url else f"{url}#L{line_number}"


def call_site_id(repo: str, url: str, path: str, line_number: object) -> str:
    return f"{repo}:{commit_from_url(url)}:{path}:{line_number}"


def parse_json(value: object, default: object) -> object:
    if value in (None, "", NA):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def why_for_false_positive(row: dict[str, Any]) -> str:
    category = row.get("classification_category", "") or ""
    snippet = (row.get("source_snippet", "") or "").strip()
    repo = row.get("repo_full_name", "") or ""

    if category == "upstream_library_self_reference" or repo in {
        "pymc-devs/pymc",
        "pyro-ppl/pyro",
        "pyro-ppl/numpyro",
        "tensorflow/probability",
        "stan-dev/cmdstanpy",
        "stan-dev/pystan",
    }:
        return "method_repo_code_itself"
    if snippet.startswith("#") or snippet.startswith('"""') or snippet.startswith("'''"):
        return "comment_only"
    if re.search(r"^(from\s+\S+\s+import|import\s+\S+)", snippet) and "(" not in snippet:
        return "import_only"
    return "no_vi_fit"


def final_from_relevant(row: dict[str, Any]) -> str:
    mode = row.get("relevant_parameter_mode", "")
    if mode == "implicit_default_for_relevant_parameters":
        return "default"
    if mode in {"unknown_optimizer_or_loss_setup", "unknown_sample_or_mc_setup"}:
        return "needs_review"

    params = set(parse_json(row.get("relevant_parameters"), []))
    optimizer_keys = {"optimizer_choice", "obj_optimizer", "optimizer"}
    mc_keys = {"sample_or_mc", "obj_n_mc", "sample_size", "loss_function", "grad_samples"}
    adjusts_optimizer = bool(params & optimizer_keys)
    adjusts_mc = bool(params & mc_keys)

    if adjusts_optimizer and adjusts_mc:
        return "adjusted_both"
    if adjusts_optimizer:
        return "adjusted_optimizer"
    if adjusts_mc:
        return "adjusted_mc"
    return "default"


def classification_reason(row: dict[str, Any], final_classification: str) -> str:
    if final_classification == "default":
        return "True-positive VI fit call with default relevant optimizer and MC/sample-count behavior."
    if final_classification == "adjusted_optimizer":
        return f"Adjusted optimizer: {row.get('optimizer_expression') or 'optimizer expression recorded in extracted arguments'}."
    if final_classification == "adjusted_mc":
        value = row.get("sample_or_mc_expression") or row.get("loss_expression") or "sample/loss expression recorded in extracted arguments"
        return f"Adjusted MC/sample-count or ELBO particle behavior: {value}."
    if final_classification == "adjusted_both":
        return (
            f"Adjusted optimizer ({row.get('optimizer_expression')}) and "
            f"MC/sample-count ({row.get('sample_or_mc_expression') or row.get('loss_expression')})."
        )
    if final_classification == "needs_review":
        return "Relevant optimizer or ELBO/loss setup remains unresolved and should be manually reviewed."
    return NA


def extracted_arguments(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "keyword_arguments": parse_json(row.get("keyword_arguments"), {}),
            "positional_arguments": parse_json(row.get("positional_arguments"), []),
            "optimizer_expression": row.get("optimizer_expression", ""),
            "loss_expression": row.get("loss_expression", ""),
            "sample_or_mc_expression": row.get("sample_or_mc_expression", ""),
        },
        ensure_ascii=False,
    )


CALL_SITE_EVIDENCE_FIELDS = [
    "call_site_id",
    "final_classification",
    "package",
    "call_kind",
    "actual_call_snippet",
    "repo_full_name",
    "commit_sha",
    "path",
    "line_number",
    "github_permalink",
    "extracted_function",
    "extracted_arguments",
    "why_in_query_results",
    "manual_review_status",
    "classification_reason",
]


def build_call_site_evidence(
    calls: list[dict[str, Any]],
    relevant: list[dict[str, Any]],
    override_keys: set[tuple[str, str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in relevant:
        final = final_from_relevant(row)
        key = (row["repo_full_name"], row["path"], str(row["line_number"]))
        rows.append(
            {
                "call_site_id": call_site_id(row["repo_full_name"], row["html_url"], row["path"], row["line_number"]),
                "final_classification": final,
                "package": row.get("package", ""),
                "call_kind": row.get("call_kind", ""),
                "actual_call_snippet": row.get("source_snippet", ""),
                "repo_full_name": row.get("repo_full_name", ""),
                "commit_sha": commit_from_url(row.get("html_url", "")),
                "path": row.get("path", ""),
                "line_number": row.get("line_number", ""),
                "github_permalink": source_link(row.get("html_url", ""), row.get("line_number", "")),
                "extracted_function": row.get("function_name", ""),
                "extracted_arguments": extracted_arguments(row),
                "why_in_query_results": "true_positive",
                "manual_review_status": "yes" if key in override_keys else "no",
                "classification_reason": classification_reason(row, final),
            }
        )

    for row in calls:
        if row.get("usage_status") == "likely_false_positive" or row.get("is_false_positive") in (1, "1"):
            key = (row["repo_full_name"], row["path"], str(row["line_number"]))
            rows.append(
                {
                    "call_site_id": call_site_id(row["repo_full_name"], row["html_url"], row["path"], row["line_number"]),
                    "final_classification": "no_vi_fit",
                    "package": row.get("package", ""),
                    "call_kind": NA,
                    "actual_call_snippet": NA,
                    "repo_full_name": row.get("repo_full_name", ""),
                    "commit_sha": commit_from_url(row.get("html_url", "")),
                    "path": row.get("path", ""),
                    "line_number": row.get("line_number", ""),
                    "github_permalink": source_link(row.get("html_url", ""), row.get("line_number", "")),
                    "extracted_function": NA,
                    "extracted_arguments": NA,
                    "why_in_query_results": why_for_false_positive(row),
                    "manual_review_status": "yes" if key in override_keys or "manual" in (row.get("classification_category", "") or "") else "no",
                    "classification_reason": NA,
                }
            )
    return rows


def build_default_advi_repos(call_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """One row per call site using the package's default optimizer AND
    default MC/sample-count behavior (final_classification == 'default')."""
    rows = [
        {
            "repo_link": f"https://github.com/{row['repo_full_name']}",
            "filename": row["path"],
            "line_number": row["line_number"],
        }
        for row in call_rows
        if row["final_classification"] == "default"
    ]
    rows.sort(key=lambda r: (str(r["repo_link"]).lower(), str(r["filename"]).lower(), int(r["line_number"])))
    return rows


def build_adjusted_advi_repos(call_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """One row per call site where the optimizer and/or MC/sample-count was
    explicitly changed from the package default. `adjusted_setting` says
    which: "optimizer", "mc", or "both"."""
    rows = []
    for row in call_rows:
        setting = ADJUSTED_SETTING_FOR_CLASSIFICATION.get(str(row["final_classification"]))
        if setting is None:
            continue
        rows.append(
            {
                "repo_link": f"https://github.com/{row['repo_full_name']}",
                "filename": row["path"],
                "line_number": row["line_number"],
                "adjusted_setting": setting,
            }
        )
    rows.sort(key=lambda r: (str(r["repo_link"]).lower(), str(r["filename"]).lower(), int(r["line_number"])))
    return rows


def build_needs_review_queue(calls: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for row in calls:
        if row.get("usage_status") == "needs_review":
            rows.append(
                {
                    "call_site_id": call_site_id(row["repo_full_name"], row["html_url"], row["path"], row["line_number"]),
                    "repo_identifier": f"{row.get('repo_full_name', '')}:{row.get('path', '')}",
                    "line_number": row.get("line_number", ""),
                    "github_permalink": source_link(row.get("html_url", ""), row.get("line_number", "")),
                    "package": row.get("package", ""),
                    "current_call_kind": row.get("call_kind", ""),
                    "current_function_name": row.get("function_name", ""),
                    "current_method": row.get("method", ""),
                    "what_needs_review": row.get("classification_category", "") or "needs_review",
                    "current_reason": row.get("classification_reason", ""),
                    "source_snippet": row.get("source_snippet", ""),
                }
            )
    return rows


NEEDS_REVIEW_QUEUE_FIELDS = [
    "call_site_id",
    "repo_identifier",
    "line_number",
    "github_permalink",
    "package",
    "current_call_kind",
    "current_function_name",
    "current_method",
    "what_needs_review",
    "current_reason",
    "source_snippet",
]

UNRESOLVED_PARAMETER_QUEUE_FIELDS = [
    "repo_identifier",
    "code_line_number",
    "github_url",
    "package",
    "call_kind",
    "unresolved_reason",
    "current_classification_mode",
    "current_relevant_parameters",
    "optimizer_relevant_status",
    "loss_relevant_status",
    "optimizer_expression",
    "loss_expression",
    "sample_or_mc_expression",
    "source_snippet",
]

PARAMETER_RESOLUTION_EVIDENCE_FIELDS = [
    "call_site_id",
    "parameter_name",
    "raw_expression_at_call",
    "resolved_value",
    "resolution_status",
    "evidence_permalink",
    "evidence_line_number",
    "resolution_reason",
]


def build_parameter_resolution_evidence(
    overrides: list[dict[str, Any]],
    unknown_parameter_rows: list[dict[str, Any]],
    relevant: list[dict[str, Any]],
) -> list[dict[str, object]]:
    relevant_by_key = {(row["repo_full_name"], row["path"], str(row["line_number"])): row for row in relevant}

    rows: list[dict[str, object]] = []
    for override in overrides:
        key = (override["repo_full_name"], override["path"], str(override["line_number"]))
        relevant_row = relevant_by_key.get(key, {})
        url = relevant_row.get("html_url", "")
        cid = call_site_id(str(override["repo_full_name"]), url, str(override["path"]), override["line_number"])
        specs = [
            ("optimizer", relevant_row.get("optimizer_expression", ""), override.get("override_optimizer_expression")),
            ("loss", relevant_row.get("loss_expression", ""), override.get("override_loss_expression")),
            ("sample_or_mc", relevant_row.get("sample_or_mc_expression", ""), override.get("override_sample_or_mc_expression")),
        ]
        for parameter_name, raw_expression, resolved_value in specs:
            if resolved_value not in (None, ""):
                rows.append(
                    {
                        "call_site_id": cid,
                        "parameter_name": parameter_name,
                        "raw_expression_at_call": raw_expression or NA,
                        "resolved_value": resolved_value,
                        "resolution_status": "resolved_manual",
                        "evidence_permalink": source_link(url, override["line_number"]) if url else NA,
                        "evidence_line_number": override["line_number"],
                        "resolution_reason": override.get("reason", ""),
                    }
                )

    for unknown in unknown_parameter_rows:
        repo, path = unknown["repo_identifier"].split(":", 1) if ":" in unknown["repo_identifier"] else (unknown["repo_identifier"], "")
        url = unknown.get("github_url", "")
        cid = call_site_id(repo, url, path, unknown.get("code_line_number", ""))
        expressions = [
            ("optimizer", unknown.get("optimizer_expression", "")),
            ("loss", unknown.get("loss_expression", "")),
            ("sample_or_mc", unknown.get("sample_or_mc_expression", "")),
        ]
        added = False
        for parameter_name, raw_expression in expressions:
            if raw_expression:
                added = True
                rows.append(
                    {
                        "call_site_id": cid,
                        "parameter_name": parameter_name,
                        "raw_expression_at_call": raw_expression,
                        "resolved_value": NA,
                        "resolution_status": "unresolved",
                        "evidence_permalink": source_link(url, unknown.get("code_line_number", "")),
                        "evidence_line_number": unknown.get("code_line_number", ""),
                        "resolution_reason": unknown.get("unresolved_reason", "classification_unknown"),
                    }
                )
        if not added:
            rows.append(
                {
                    "call_site_id": cid,
                    "parameter_name": "optimizer_or_loss_setup",
                    "raw_expression_at_call": NA,
                    "resolved_value": NA,
                    "resolution_status": "unresolved",
                    "evidence_permalink": source_link(url, unknown.get("code_line_number", "")),
                    "evidence_line_number": unknown.get("code_line_number", ""),
                    "resolution_reason": unknown.get("unresolved_reason", "classification_unknown"),
                }
            )
    return rows


def build_repo_level_summary(
    relevant: list[dict[str, Any]],
    pymc_family_rows: list[dict[str, Any]],
    repo_urls: dict[str, str],
) -> list[dict[str, object]]:
    pymc_family_by_repo = {row["repo_full_name"]: row.get("pymc_advi_family_scope", "") for row in pymc_family_rows}

    repo_data: dict[str, dict[str, object]] = defaultdict(
        lambda: {"packages": set(), "call_ids": [], "adjusts_optimizer": False, "adjusts_mc": False, "stan_algorithms": set()}
    )
    for row in relevant:
        final = final_from_relevant(row)
        if final == "needs_review":
            continue
        repo = row["repo_full_name"]
        data = repo_data[repo]
        data["packages"].add(row["package"])
        data["call_ids"].append(call_site_id(repo, row["html_url"], row["path"], row["line_number"]))
        if final in {"adjusted_optimizer", "adjusted_both"}:
            data["adjusts_optimizer"] = True
        if final in {"adjusted_mc", "adjusted_both"}:
            data["adjusts_mc"] = True
        if row["package"] == "stan":
            text = " ".join([row.get("method_or_loss", ""), row.get("keyword_arguments", ""), row.get("source_snippet", "")]).lower()
            if "fullrank" in text:
                data["stan_algorithms"].add("fullrank")
            elif "meanfield" in text:
                data["stan_algorithms"].add("meanfield")
            elif "algorithm" in text:
                data["stan_algorithms"].add("configurable_or_variable")
            else:
                data["stan_algorithms"].add("unspecified_or_default")

    rows = []
    for repo, data in sorted(repo_data.items()):
        packages = sorted(data["packages"])
        adjusts_optimizer = bool(data["adjusts_optimizer"])
        adjusts_mc = bool(data["adjusts_mc"])
        rows.append(
            {
                "repo_full_name": repo,
                "repo_url": repo_urls.get(repo, f"https://github.com/{repo}"),
                "packages_used": ";".join(packages),
                "adjusts_anything": adjusts_optimizer or adjusts_mc,
                "adjusts_optimizer": adjusts_optimizer,
                "adjusts_mc": adjusts_mc,
                "uses_default_optimizer": not adjusts_optimizer,
                "uses_default_mc": not adjusts_mc,
                "pymc_advi_family": pymc_family_by_repo.get(repo, NA),
                "stan_variational_algorithm": ";".join(sorted(data["stan_algorithms"])) if data["stan_algorithms"] else NA,
                "num_true_positive_call_sites": len(data["call_ids"]),
                "supporting_call_site_ids": ";".join(data["call_ids"]),
            }
        )
    return rows


REPO_LEVEL_SUMMARY_FIELDS = [
    "repo_full_name",
    "repo_url",
    "packages_used",
    "adjusts_anything",
    "adjusts_optimizer",
    "adjusts_mc",
    "uses_default_optimizer",
    "uses_default_mc",
    "pymc_advi_family",
    "stan_variational_algorithm",
    "num_true_positive_call_sites",
    "supporting_call_site_ids",
]


def build_overview(
    call_rows: list[dict[str, object]],
    default_rows: list[dict[str, object]],
    adjusted_rows: list[dict[str, object]],
    needs_review_rows: list[dict[str, object]],
    repo_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """One row per metric: the headline counts a reader needs to interpret
    default_advi_repos.csv/adjusted_advi_repos.csv without recomputing
    anything themselves."""
    adjusted_breakdown = Counter(row["adjusted_setting"] for row in adjusted_rows)
    distinct_repos = {row["repo_full_name"] for row in call_rows}
    default_repos = {row["repo_full_name"] for row in call_rows if row["final_classification"] == "default"}
    adjusted_repos = {row["repo_full_name"] for row in call_rows if row["final_classification"] in ADJUSTED_SETTING_FOR_CLASSIFICATION}

    rows = [
        {"metric": "total_call_sites_classified", "value": len(call_rows)},
        {"metric": "true_positive_call_sites", "value": sum(1 for row in call_rows if row["why_in_query_results"] == "true_positive")},
        {"metric": "excluded_no_vi_fit_call_sites", "value": sum(1 for row in call_rows if row["why_in_query_results"] != "true_positive")},
        {"metric": "default_call_sites", "value": len(default_rows)},
        {"metric": "adjusted_call_sites", "value": len(adjusted_rows)},
        {"metric": "adjusted_call_sites_optimizer_only", "value": adjusted_breakdown.get("optimizer", 0)},
        {"metric": "adjusted_call_sites_mc_only", "value": adjusted_breakdown.get("mc", 0)},
        {"metric": "adjusted_call_sites_both", "value": adjusted_breakdown.get("both", 0)},
        {"metric": "needs_review_call_sites", "value": len(needs_review_rows)},
        {"metric": "distinct_repositories_scanned", "value": len(distinct_repos)},
        {"metric": "repositories_with_a_default_call_site", "value": len(default_repos)},
        {"metric": "repositories_with_an_adjusted_call_site", "value": len(adjusted_repos)},
        {"metric": "repo_level_summary_rows", "value": len(repo_rows)},
    ]
    return rows


OVERVIEW_FIELDS = ["metric", "value"]


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        is not None
    )


def load_overrides(conn: sqlite3.Connection) -> list[dict[str, object]]:
    return [dict(row) for row in conn.execute("SELECT * FROM github_advi_parameter_overrides")]


def build_random_repo_review_actions(conn: sqlite3.Connection) -> list[dict[str, object]]:
    if not table_exists(conn, "vi_github_random_repo_review_actions"):
        return []
    return [
        dict(row)
        for row in conn.execute(
            """
            SELECT review_id, repo_full_name, repo_url, packages_used,
                   repo_adjusts_anything, repo_adjusts_optimizer, repo_adjusts_mc,
                   num_true_positive_call_sites, sampled_call_site_ids, decision,
                   suggested_correct_classification, notes, seed, created_at
            FROM vi_github_random_repo_review_actions
            ORDER BY review_id
            """
        )
    ]


DUPLICATE_REPO_CLUSTERS_FIELDS = ["repo_full_name", "canonical_repo_full_name", "is_canonical"]


def resolve_requested_outputs(*, default_only: bool, adjusted_only: bool, overview: bool, comprehensive: bool,
                               call_sites_only: bool, repo_summary_only: bool, include_review_queue: bool) -> set[str]:
    if comprehensive:
        return set(ALL_OUTPUTS)
    narrowed = {
        "default_advi_repos": default_only,
        "adjusted_advi_repos": adjusted_only,
        "overview": overview,
        "call_site_evidence": call_sites_only,
        "repo_level_summary": repo_summary_only,
    }
    requested = {name for name, flagged in narrowed.items() if flagged}
    if not requested:
        requested = set(CORE_OUTPUTS)
    if include_review_queue:
        requested |= set(QUEUE_OUTPUTS)
    return requested


def build_analysis(
    conn: sqlite3.Connection,
    report_tables: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    *,
    default_only: bool = False,
    adjusted_only: bool = False,
    overview: bool = False,
    comprehensive: bool = False,
    call_sites_only: bool = False,
    repo_summary_only: bool = False,
    include_review_queue: bool = False,
    keep_existing: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Compute the output tables and write the requested ones to `out_dir`.
    `report_tables` is the result of github_advi.fetch_github_advi_report_tables()."""
    requested = resolve_requested_outputs(
        default_only=default_only,
        adjusted_only=adjusted_only,
        overview=overview,
        comprehensive=comprehensive,
        call_sites_only=call_sites_only,
        repo_summary_only=repo_summary_only,
        include_review_queue=include_review_queue,
    )

    calls = report_tables["calls"]
    relevant = report_tables["relevant"]
    overrides = load_overrides(conn)
    override_keys = {(str(row["repo_full_name"]), str(row["path"]), str(row["line_number"])) for row in overrides}

    call_rows = build_call_site_evidence(calls, relevant, override_keys)
    default_rows = build_default_advi_repos(call_rows)
    adjusted_rows = build_adjusted_advi_repos(call_rows)
    needs_review_rows = build_needs_review_queue(calls)
    repo_urls = {row["repo_full_name"]: f"https://github.com/{row['repo_full_name']}" for row in relevant}
    repo_rows = build_repo_level_summary(relevant, report_tables["pymc_family"], repo_urls)
    overview_rows = build_overview(call_rows, default_rows, adjusted_rows, needs_review_rows, repo_rows)
    parameter_rows = build_parameter_resolution_evidence(overrides, report_tables["unknown_parameters"], relevant)
    unresolved_queue_rows = [
        {field: row.get(field, NA) for field in UNRESOLVED_PARAMETER_QUEUE_FIELDS}
        for row in report_tables["unknown_parameters"]
    ]
    duplicate_rows = []
    random_review_rows = []
    if "duplicate_repo_clusters" in requested:
        from .github_advi import fetch_duplicate_repo_cluster_rows

        duplicate_rows = fetch_duplicate_repo_cluster_rows(conn)
    if "random_repo_review_actions" in requested:
        random_review_rows = build_random_repo_review_actions(conn)

    tables: dict[str, tuple[list[dict[str, object]], list[str]]] = {
        "call_site_evidence": (call_rows, CALL_SITE_EVIDENCE_FIELDS),
        "default_advi_repos": (default_rows, ["repo_link", "filename", "line_number"]),
        "adjusted_advi_repos": (adjusted_rows, ["repo_link", "filename", "line_number", "adjusted_setting"]),
        "repo_level_summary": (repo_rows, REPO_LEVEL_SUMMARY_FIELDS),
        "overview": (overview_rows, OVERVIEW_FIELDS),
        "needs_review_queue": (needs_review_rows, NEEDS_REVIEW_QUEUE_FIELDS),
        "unresolved_parameter_queue": (unresolved_queue_rows, UNRESOLVED_PARAMETER_QUEUE_FIELDS),
        "parameter_resolution_evidence": (parameter_rows, PARAMETER_RESOLUTION_EVIDENCE_FIELDS),
        "duplicate_repo_clusters": (duplicate_rows, DUPLICATE_REPO_CLUSTERS_FIELDS),
        "random_repo_review_actions": (random_review_rows, RANDOM_REPO_REVIEW_FIELDS),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if not keep_existing:
        for name in ALL_OUTPUTS:
            if name not in requested:
                stale = out_dir / FILENAME_FOR_OUTPUT[name]
                stale.unlink(missing_ok=True)

    written: dict[str, list[dict[str, object]]] = {}
    for name in requested:
        rows, fields = tables[name]
        write_csv(out_dir / FILENAME_FOR_OUTPUT[name], rows, fields)
        written[name] = rows
    return written


def main() -> None:
    from .config import settings
    from .db import connect, init_db
    from .github_advi import fetch_github_advi_report_tables

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Path to the github-advi-usage SQLite database.")
    parser.add_argument("--reports-dir", type=Path, required=True, help="Directory to write reports/*.csv into.")
    parser.add_argument("--default-only", action="store_true")
    parser.add_argument("--adjusted-only", action="store_true")
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--comprehensive", action="store_true")
    parser.add_argument("--call-sites-only", action="store_true")
    parser.add_argument("--repo-summary-only", action="store_true")
    parser.add_argument("--include-review-queue", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    cfg = settings(args.db)
    conn = connect(cfg.db_path)
    init_db(conn)
    report_tables = fetch_github_advi_report_tables(conn)
    written = build_analysis(
        conn,
        report_tables,
        args.reports_dir,
        default_only=args.default_only,
        adjusted_only=args.adjusted_only,
        overview=args.overview,
        comprehensive=args.comprehensive,
        call_sites_only=args.call_sites_only,
        repo_summary_only=args.repo_summary_only,
        include_review_queue=args.include_review_queue,
        keep_existing=args.keep_existing,
    )
    print(f"Wrote {len(written)} report(s) to {args.reports_dir}")


if __name__ == "__main__":
    main()
