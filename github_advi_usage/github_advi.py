from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import multiprocessing
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from requests import HTTPError, RequestException

from .api import ApiClient
from .db import log_event, utc_now


QUERY_MANIFEST = Path(__file__).with_name("github_query_manifest.csv")


def manifest_queries(manifest: Path = QUERY_MANIFEST) -> list[str]:
    """The distinct base queries in a query manifest, in manifest order.

    Rows for `size:` windows record sub-queries that an earlier run split off;
    the search splits oversized queries by size itself, so those rows map back
    to their base query.
    """
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    queries = [re.sub(r" size:\d+\.\.\d+$", "", row["query_string"]) for row in rows if row.get("query_string")]
    if not queries:
        raise ValueError(f"No query_string values found in {manifest}")
    return list(dict.fromkeys(queries))


@dataclass(frozen=True)
class AdvICall:
    line_number: int
    package: str
    call_kind: str
    function_name: str
    method: str | None
    changed_parameters: list[str]
    keyword_arguments: dict[str, str]
    positional_arguments: list[str]
    source_snippet: str


@dataclass(frozen=True)
class ViVariable:
    package: str
    method: str | None


@dataclass(frozen=True)
class GithubCallClassification:
    call_id: int
    usage_status: str
    is_false_positive: bool
    category: str
    confidence: float
    reason: str


GITHUB_CALL_CLASSIFIER_VERSION = "2026-05-06-v1-path-and-call-rules"

OFFICIAL_PACKAGE_REPOS = {
    "pymc-devs/pymc",
    "pymc-devs/pymc3",
    "pyro-ppl/pyro",
    "pyro-ppl/numpyro",
    "tensorflow/probability",
    "stan-dev/cmdstanpy",
    "stan-dev/pystan",
}

VENDORED_PATH_MARKERS = (
    "/site-packages/",
    "/dist-packages/",
    "/node_modules/",
    "/.venv/",
    "/venv/",
    "/env/",
    "/myenv/",
    "/virtualenv/",
    "/python-3.",
    "/lib/python",
    "/lib/site-packages/",
    "/lib64/python",
    "/.tox/",
    "/vendor/",
    "/.vendor/",
    "/third_party/",
)

CODE_CORPUS_MARKERS = (
    "/downloaded_files/",
    "/data_collection/pr_files/",
    "/original_datasets/the_stack",
    "/the_stack_organized/",
    "/top-pypi-sdists",
    "/code-generation-benchmark/",
)

UPSTREAM_PACKAGE_PATH_MARKERS = (
    "/tensorflow_probability/",
    "/cmdstanpy/",
    "/pymc/",
    "/pymc3/",
    "/pyro/",
    "/numpyro/",
)

TUTORIAL_PATH_MARKERS = (
    "/example",
    "/examples/",
    "/tutorial",
    "/tutorials/",
    "/notebook",
    "/notebooks/",
    "/demo",
    "/demos/",
    "/course",
    "/courses/",
    "/workshop",
)

TEST_PATH_MARKERS = (
    "/test/",
    "/tests/",
    "/testing/",
    "test_",
    "_test.py",
)

BULK_MIRROR_ACCOUNT_PREFIXES = (
    "orgTestCodacy11KRepos110MB",
)
BULK_MIRROR_ACCOUNT_PREFIXES_CASEFOLDED = tuple(prefix.casefold() for prefix in BULK_MIRROR_ACCOUNT_PREFIXES)

VENDORED_RESEARCH_TOOL_REPOS = {
    "wonyeol/smoothness-analysis",
    "wonyeol/static-analysis-for-support-match",
}


def search_github_advi_code_exhaustive(
    conn: sqlite3.Connection,
    client: ApiClient,
    *,
    queries: list[str] | None = None,
    interval: float = 6.2,
    max_results_per_query: int = 1000,
    max_size: int = 1_000_000,
    min_size_window: int = 1,
) -> int:
    if not client.settings.github_token:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub code search")
    total = 0
    for query in queries or manifest_queries():
        total += search_github_advi_code_exhaustive_query(
            conn,
            client,
            query=query,
            interval=interval,
            max_results_per_query=max_results_per_query,
            max_size=max_size,
            min_size_window=min_size_window,
        )
    return total


def search_github_advi_code_exhaustive_query(
    conn: sqlite3.Connection,
    client: ApiClient,
    *,
    query: str,
    interval: float,
    max_results_per_query: int,
    max_size: int,
    min_size_window: int,
) -> int:
    try:
        first_page = github_code_search_page(client, query, page=1, per_page=100)
    except Exception as exc:
        log_event(conn, "github_search_advi_exhaustive", str(exc), level="error", payload={"query": query})
        print(f"GitHub query failed for {query!r}: {exc}")
        return 0
    total_count = int(first_page.get("total_count") or 0)
    if total_count <= max_results_per_query:
        saved = save_github_code_search_all_pages(
            conn,
            client,
            query=query,
            first_page=first_page,
            total_count=total_count,
            interval=interval,
            max_results_per_query=max_results_per_query,
        )
        log_event(
            conn,
            "github_search_advi_exhaustive",
            f"Completed GitHub code query: {query}",
            payload={"query": query, "total_count": total_count, "saved_items": saved},
        )
        print(f"{query}: GitHub total={total_count}, saved page items={saved}")
        if interval:
            time.sleep(interval)
        return saved
    print(f"{query}: GitHub total={total_count}, splitting by file size")
    return search_github_advi_code_size_window(
        conn,
        client,
        base_query=query,
        size_start=0,
        size_end=max_size,
        min_size_window=min_size_window,
        interval=interval,
        max_results_per_query=max_results_per_query,
    )


def search_github_advi_code_size_window(
    conn: sqlite3.Connection,
    client: ApiClient,
    *,
    base_query: str,
    size_start: int,
    size_end: int,
    min_size_window: int,
    interval: float,
    max_results_per_query: int,
) -> int:
    query = f"{base_query} size:{size_start}..{size_end}"
    try:
        first_page = github_code_search_page(client, query, page=1, per_page=100)
    except Exception as exc:
        log_event(
            conn,
            "github_search_advi_exhaustive",
            str(exc),
            level="error",
            payload={"query": query, "size_start": size_start, "size_end": size_end},
        )
        print(f"GitHub query failed for {query!r}: {exc}")
        return 0
    total_count = int(first_page.get("total_count") or 0)
    width = size_end - size_start + 1
    if total_count > max_results_per_query and width > min_size_window:
        midpoint = size_start + width // 2 - 1
        return search_github_advi_code_size_window(
            conn,
            client,
            base_query=base_query,
            size_start=size_start,
            size_end=midpoint,
            min_size_window=min_size_window,
            interval=interval,
            max_results_per_query=max_results_per_query,
        ) + search_github_advi_code_size_window(
            conn,
            client,
            base_query=base_query,
            size_start=midpoint + 1,
            size_end=size_end,
            min_size_window=min_size_window,
            interval=interval,
            max_results_per_query=max_results_per_query,
        )
    saved = save_github_code_search_all_pages(
        conn,
        client,
        query=query,
        first_page=first_page,
        total_count=total_count,
        interval=interval,
        max_results_per_query=max_results_per_query,
    )
    log_event(
        conn,
        "github_search_advi_exhaustive",
        f"Completed GitHub code query size window: {query}",
        payload={"query": query, "total_count": total_count, "saved_items": saved},
    )
    print(f"{query}: GitHub total={total_count}, saved page items={saved}")
    if interval:
        time.sleep(interval)
    return saved


def github_code_search_page(client: ApiClient, query: str, *, page: int, per_page: int) -> dict[str, Any]:
    return github_get_with_rate_limit(
        client,
        "search/code",
        {"q": query, "per_page": per_page, "page": page},
    )


def save_github_code_search_page(conn: sqlite3.Connection, page: dict[str, Any]) -> int:
    saved = 0
    for item in page.get("items") or []:
        save_github_search_item(conn, item)
        saved += 1
    return saved


def save_github_code_search_all_pages(
    conn: sqlite3.Connection,
    client: ApiClient,
    *,
    query: str,
    first_page: dict[str, Any],
    total_count: int,
    interval: float,
    max_results_per_query: int,
) -> int:
    saved = save_github_code_search_page(conn, first_page)
    pages = min(10, (min(total_count, max_results_per_query) + 99) // 100)
    for page in range(2, pages + 1):
        if interval:
            time.sleep(interval)
        saved += save_github_code_search_page(conn, github_code_search_page(client, query, page=page, per_page=100))
    return saved


def scan_all_pending_github_advi_files(
    conn: sqlite3.Connection,
    client: ApiClient,
    *,
    batch_size: int = 100,
    interval: float = 1.0,
) -> int:
    total = 0
    while True:
        count = scan_github_advi_files(conn, client, limit=batch_size, interval=interval)
        total += count
        if count == 0:
            return total


SCAN_FILE_TIMEOUT_SECONDS = 20


class ScanTimeoutError(Exception):
    pass


def _find_advi_calls_worker(content: str, queue: "multiprocessing.Queue") -> None:
    try:
        queue.put(("ok", find_advi_calls(content)))
    except Exception as exc:  # noqa: BLE001 - report any worker failure back to the parent
        queue.put(("error", str(exc)))


def find_advi_calls_with_timeout(content: str, timeout_seconds: int) -> list[AdvICall]:
    ctx = multiprocessing.get_context("spawn")
    queue: "multiprocessing.Queue" = ctx.Queue()
    process = ctx.Process(target=_find_advi_calls_worker, args=(content, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        raise ScanTimeoutError(f"Parsing exceeded {timeout_seconds}s (worker process hard-killed)")
    if queue.empty():
        raise ScanTimeoutError("Worker process exited without a result (crashed or killed)")
    status, payload = queue.get()
    if status == "error":
        raise RuntimeError(payload)
    return payload


def record_scan_timeout(data_dir: Path, row: sqlite3.Row, exc: Exception) -> Path:
    error_list_path = data_dir / "reports" / "github_advi_scan_timeouts.csv"
    error_list_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not error_list_path.exists()
    with error_list_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["timestamp", "file_id", "repo_full_name", "path", "reason"])
        writer.writerow([utc_now(), row["file_id"], row["repo_full_name"], row["path"], str(exc)])
    return error_list_path


def scan_github_advi_files(
    conn: sqlite3.Connection,
    client: ApiClient,
    *,
    limit: int | None = None,
    interval: float = 1.0,
) -> int:
    rows = conn.execute(
        """
        SELECT file_id, repo_full_name, path
        FROM github_files
        WHERE scan_status IN ('pending', 'failed')
        ORDER BY updated_at DESC
        LIMIT COALESCE(?, -1)
        """,
        (limit,),
    ).fetchall()
    scanned = 0
    for row in rows:
        try:
            content = fetch_github_file_text(client, row["repo_full_name"], row["path"])
            calls = find_advi_calls_with_timeout(content, SCAN_FILE_TIMEOUT_SECONDS)
            replace_github_advi_calls(conn, row["file_id"], calls)
            set_github_file_status(conn, row["file_id"], "scanned")
            scanned += 1
            print(f"Scanned {row['repo_full_name']}:{row['path']} ({len(calls)} ADVI-style call(s))")
        except ScanTimeoutError as exc:
            set_github_file_status(conn, row["file_id"], "scanned")
            error_list_path = record_scan_timeout(client.settings.data_dir, row, exc)
            log_event(
                conn,
                "github_scan_advi_timeout",
                str(exc),
                entity_id=row["file_id"],
                level="error",
                payload={"repo": row["repo_full_name"], "path": row["path"]},
            )
            scanned += 1
            print(
                f"TIMEOUT scanning {row['repo_full_name']}:{row['path']} after {SCAN_FILE_TIMEOUT_SECONDS}s "
                f"— marked scanned, logged to {error_list_path}"
            )
        except Exception as exc:
            set_github_file_status(conn, row["file_id"], "failed")
            log_event(
                conn,
                "github_scan_advi",
                str(exc),
                entity_id=row["file_id"],
                level="error",
                payload={"repo": row["repo_full_name"], "path": row["path"]},
            )
            print(f"GitHub scan failed for {row['repo_full_name']}:{row['path']}: {exc}")
        if interval:
            time.sleep(interval)
    return scanned


def fetch_github_file_text(client: ApiClient, repo_full_name: str, path: str) -> str:
    data = github_get_with_rate_limit(client, f"repos/{repo_full_name}/contents/{path}")
    encoding = data.get("encoding")
    content = data.get("content") or ""
    if encoding != "base64":
        raise ValueError(f"Unsupported GitHub content encoding: {encoding}")
    return base64.b64decode(content).decode("utf-8", errors="replace")


def github_get_with_rate_limit(client: ApiClient, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    for attempt in range(4):
        try:
            return client.github_get(path, params)
        except HTTPError as exc:
            response = exc.response
            if response is None or response.status_code not in {403, 429, 500, 502, 503, 504}:
                raise
            reset = response.headers.get("x-ratelimit-reset")
            if reset and response.status_code in {403, 429}:
                delay = max(5, min(120, int(reset) - int(time.time()) + 2))
            else:
                delay = min(60, 2**attempt)
            time.sleep(delay)
        except RequestException:
            time.sleep(min(60, 2**attempt))
    return client.github_get(path, params)


def save_github_search_item(conn: sqlite3.Connection, item: dict[str, Any]) -> None:
    repo = item.get("repository") or {}
    full_name = repo.get("full_name")
    path = item.get("path")
    if not full_name or not path:
        return
    now = utc_now()
    conn.execute(
        """
        INSERT INTO github_repositories(
            full_name, html_url, description, language, stars, forks, pushed_at,
            raw_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(full_name) DO UPDATE SET
            html_url=excluded.html_url,
            description=excluded.description,
            language=excluded.language,
            stars=excluded.stars,
            forks=excluded.forks,
            pushed_at=excluded.pushed_at,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """,
        (
            full_name,
            repo.get("html_url") or f"https://github.com/{full_name}",
            repo.get("description"),
            repo.get("language"),
            repo.get("stargazers_count"),
            repo.get("forks_count"),
            repo.get("pushed_at"),
            json.dumps(repo),
            now,
            now,
        ),
    )
    file_id = github_file_id(full_name, path)
    conn.execute(
        """
        INSERT INTO github_files(
            file_id, repo_full_name, path, html_url, raw_url, sha, raw_json,
            scan_status, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        ON CONFLICT(repo_full_name, path) DO UPDATE SET
            html_url=excluded.html_url,
            raw_url=excluded.raw_url,
            sha=excluded.sha,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """,
        (
            file_id,
            full_name,
            path,
            item.get("html_url") or f"https://github.com/{full_name}/blob/HEAD/{path}",
            item.get("url"),
            item.get("sha"),
            json.dumps(item),
            now,
            now,
        ),
    )
    conn.commit()


def github_file_id(repo_full_name: str, path: str) -> str:
    return "GHFILE-" + hashlib.sha256(f"{repo_full_name}:{path}".encode("utf-8")).hexdigest()[:24]


def set_github_file_status(conn: sqlite3.Connection, file_id: str, status: str) -> None:
    conn.execute("UPDATE github_files SET scan_status=?, updated_at=? WHERE file_id=?", (status, utc_now(), file_id))
    conn.commit()


def replace_github_advi_calls(conn: sqlite3.Connection, file_id: str, calls: Iterable[AdvICall]) -> None:
    row = conn.execute(
        "SELECT repo_full_name, path, html_url FROM github_files WHERE file_id=?",
        (file_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown GitHub file: {file_id}")
    conn.execute(
        """
        DELETE FROM github_advi_call_classifications
        WHERE call_id IN (SELECT id FROM github_advi_calls WHERE file_id=?)
        """,
        (file_id,),
    )
    conn.execute(
        """
        DELETE FROM github_advi_settings_companions
        WHERE call_id IN (SELECT id FROM github_advi_calls WHERE file_id=?)
           OR settings_call_id IN (SELECT id FROM github_advi_calls WHERE file_id=?)
        """,
        (file_id, file_id),
    )
    conn.execute("DELETE FROM github_advi_calls WHERE file_id=?", (file_id,))
    for call in calls:
        conn.execute(
            """
            INSERT INTO github_advi_calls(
                file_id, repo_full_name, path, html_url, line_number, package,
                call_kind, function_name, method, changed_parameters,
                keyword_arguments, positional_arguments, source_snippet, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                row["repo_full_name"],
                row["path"],
                row["html_url"],
                call.line_number,
                call.package,
                call.call_kind,
                call.function_name,
                call.method,
                json.dumps(call.changed_parameters),
                json.dumps(call.keyword_arguments),
                json.dumps(call.positional_arguments),
                call.source_snippet,
                utc_now(),
            ),
        )
    conn.commit()


# The call_kind(s) per package that carry the optimizer/loss/sample-size (or
# Stan's grad_samples) arguments. Other true-positive calls for that package
# borrow these settings from a companion call in the same file, else the same repo.
PRIMARY_SETTINGS_CALL_KINDS_BY_PACKAGE: dict[str, tuple[str, ...]] = {
    "pymc": ("fit", "advi_fit_method"),
    "tensorflow_probability": ("fit_surrogate_posterior",),
    "stan": ("stan_variational_method",),
    "pyro": ("svi_autoguide",),
    "numpyro": ("svi_autoguide",),
    "pyro_or_numpyro": ("svi_autoguide",),
}


def save_settings_companion(conn: sqlite3.Connection, call_id: int, settings_call_id: int | None, scope: str) -> None:
    conn.execute(
        """
        INSERT INTO github_advi_settings_companions(call_id, settings_call_id, resolution_scope, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(call_id) DO UPDATE SET
            settings_call_id=excluded.settings_call_id,
            resolution_scope=excluded.resolution_scope,
            created_at=excluded.created_at
        """,
        (call_id, settings_call_id, scope, utc_now()),
    )


def resolve_settings_companions(conn: sqlite3.Connection) -> dict[str, int]:
    """For every likely_true_positive call, find the call_id that actually carries
    its resolvable settings: itself, if its call_kind is the package's primary
    settings-bearing kind; else a companion of that kind in the same file; else one
    anywhere else in the repo; else none. Persisted so the parameter-analysis SQL
    can borrow settings across call sites instead of only reading a call's own
    (often empty) keyword_arguments."""
    rows = conn.execute(
        """
        SELECT c.id, c.repo_full_name, c.path, c.package, c.call_kind
        FROM github_advi_calls c
        JOIN github_advi_call_classifications k ON k.call_id = c.id
        WHERE k.usage_status='likely_true_positive'
        """
    ).fetchall()
    counts = {"self": 0, "same_file": 0, "same_repo": 0, "none": 0}
    for row in rows:
        search_packages = PACKAGE_MATCH_GROUPS.get(row["package"], (row["package"],))
        primary_kinds: set[str] = set()
        for pkg in search_packages:
            primary_kinds.update(PRIMARY_SETTINGS_CALL_KINDS_BY_PACKAGE.get(pkg, ()))
        if not primary_kinds:
            save_settings_companion(conn, row["id"], None, "none")
            counts["none"] += 1
            continue
        if row["call_kind"] in primary_kinds:
            save_settings_companion(conn, row["id"], row["id"], "self")
            counts["self"] += 1
            continue
        placeholders = ",".join("?" for _ in primary_kinds)
        package_placeholders = ",".join("?" for _ in search_packages)
        same_file = conn.execute(
            f"""
            SELECT fc.id
            FROM github_advi_calls fc
            JOIN github_advi_call_classifications fk ON fk.call_id = fc.id
            WHERE fc.repo_full_name=? AND fc.path=?
              AND fc.package IN ({package_placeholders})
              AND fc.call_kind IN ({placeholders})
              AND fk.usage_status='likely_true_positive'
            LIMIT 1
            """,
            (row["repo_full_name"], row["path"], *search_packages, *primary_kinds),
        ).fetchone()
        if same_file:
            save_settings_companion(conn, row["id"], same_file["id"], "same_file")
            counts["same_file"] += 1
            continue
        same_repo = conn.execute(
            f"""
            SELECT fc.id
            FROM github_advi_calls fc
            JOIN github_advi_call_classifications fk ON fk.call_id = fc.id
            WHERE fc.repo_full_name=?
              AND fc.package IN ({package_placeholders})
              AND fc.call_kind IN ({placeholders})
              AND fk.usage_status='likely_true_positive'
            LIMIT 1
            """,
            (row["repo_full_name"], *search_packages, *primary_kinds),
        ).fetchone()
        if same_repo:
            save_settings_companion(conn, row["id"], same_repo["id"], "same_repo")
            counts["same_repo"] += 1
            continue
        save_settings_companion(conn, row["id"], None, "none")
        counts["none"] += 1
    conn.commit()
    return counts


def classify_github_advi_calls(conn: sqlite3.Connection, *, limit: int | None = None, refresh: bool = False) -> int:
    where = "" if refresh else "WHERE c.id NOT IN (SELECT call_id FROM github_advi_call_classifications)"
    rows = conn.execute(
        f"""
        SELECT c.*
        FROM github_advi_calls c
        {where}
        ORDER BY c.repo_full_name, c.path, c.line_number
        LIMIT COALESCE(?, -1)
        """,
        (limit,),
    ).fetchall()
    count = 0
    for row in rows:
        save_github_advi_call_classification(conn, classify_github_advi_call(dict(row)))
        count += 1
    return count


FITTING_CALL_KINDS = (
    "fit",
    "advi_fit_method",
    "svi_run_method",
    "svi_step_method",
    "stan_variational_method",
    "fit_surrogate_posterior",
)


PACKAGE_MATCH_GROUPS: dict[str, tuple[str, ...]] = {
    "pyro": ("pyro", "pyro_or_numpyro"),
    "numpyro": ("numpyro", "pyro_or_numpyro"),
    "pyro_or_numpyro": ("pyro", "numpyro", "pyro_or_numpyro"),
    "pymc": ("pymc",),
    "tensorflow_probability": ("tensorflow_probability",),
    "stan": ("stan",),
}


def cluster_duplicate_needs_review_repos(
    conn: sqlite3.Connection, *, overlap_threshold: float = 0.6, min_shared_paths: int = 2
) -> dict[str, str]:
    """Group repos whose needs_review call paths substantially overlap
    (re-uploads of the same content, which GitHub's fork flag does not catch)
    and map each repo to the most-starred repo in its group."""
    rows = conn.execute(
        """
        SELECT DISTINCT c.repo_full_name, c.path
        FROM github_advi_calls c
        JOIN github_advi_call_classifications k ON k.call_id = c.id
        WHERE k.usage_status='needs_review'
        """
    ).fetchall()
    paths_by_repo: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        paths_by_repo[row["repo_full_name"]].add(row["path"])

    repos_by_path: dict[str, set[str]] = defaultdict(set)
    for repo, paths in paths_by_repo.items():
        for path in paths:
            repos_by_path[path].add(repo)

    parent: dict[str, str] = {repo: repo for repo in paths_by_repo}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pairs_checked: set[tuple[str, str]] = set()
    for path, repos in repos_by_path.items():
        if len(repos) < 2:
            continue
        repo_list = sorted(repos)
        for i in range(len(repo_list)):
            for j in range(i + 1, len(repo_list)):
                pair = (repo_list[i], repo_list[j])
                if pair in pairs_checked:
                    continue
                pairs_checked.add(pair)
                a_paths, b_paths = paths_by_repo[pair[0]], paths_by_repo[pair[1]]
                intersection = len(a_paths & b_paths)
                union_size = len(a_paths | b_paths)
                smaller_size = min(len(a_paths), len(b_paths))
                jaccard_ratio = intersection / union_size if union_size else 0.0
                containment_ratio = intersection / smaller_size if smaller_size else 0.0
                # Containment catches partial copies that are missing some files.
                if (
                    jaccard_ratio >= overlap_threshold or containment_ratio >= 0.8
                ) and intersection >= min_shared_paths:
                    union(pair[0], pair[1])

    stars = {
        row["full_name"]: row["stars"] or 0
        for row in conn.execute("SELECT full_name, stars FROM github_repositories").fetchall()
    }
    clusters: dict[str, list[str]] = defaultdict(list)
    for repo in paths_by_repo:
        clusters[find(repo)].append(repo)

    canonical_for: dict[str, str] = {}
    for members in clusters.values():
        canonical = max(members, key=lambda r: (stars.get(r, 0), r))
        for member in members:
            canonical_for[member] = canonical

    return canonical_for


def classify_github_advi_call(row: dict[str, Any]) -> GithubCallClassification:
    repo = (row.get("repo_full_name") or "").casefold()
    path = normalize_repo_path(row.get("path") or "")
    package = (row.get("package") or "").casefold()
    call_kind = (row.get("call_kind") or "").casefold()
    function_name = (row.get("function_name") or "").casefold()

    if row.get("package") == "unknown" or call_kind == "textual_match":
        return github_call_classification(
            row,
            "likely_false_positive",
            True,
            "textual_or_unparsed_match",
            0.6,
            "The file failed to parse as Python, so this came from a plain-text fallback search "
            "(can match plain comments/strings) rather than a real call site.",
        )
    if call_kind in FITTING_CALL_KINDS:
        return github_call_classification(
            row,
            "likely_true_positive",
            False,
            "confirmed_real_fit_call",
            0.75,
            "A complete fit/step/variational call exists here. Testing or vendoring ADVI is still real "
            "usage of the software, so this outranks path- or repo-based exclusions.",
        )
    if repo.split("/", 1)[0] in BULK_MIRROR_ACCOUNT_PREFIXES_CASEFOLDED:
        return github_call_classification(
            row,
            "likely_false_positive",
            True,
            "bulk_mirror_account",
            0.9,
            "The repository owner is a known bulk code-corpus-mirroring account, not an independent project.",
        )
    if row.get("repo_full_name") in VENDORED_RESEARCH_TOOL_REPOS:
        return github_call_classification(
            row,
            "likely_false_positive",
            True,
            "vendored_upstream_examples_as_test_fixtures",
            0.85,
            "This repository reuses the upstream package's own example programs as test fixtures for a "
            "program-analysis tool, rather than applying VI to a domain problem.",
        )
    if any(marker in path for marker in CODE_CORPUS_MARKERS):
        return github_call_classification(
            row,
            "likely_false_positive",
            True,
            "code_corpus_snapshot",
            0.92,
            "The path looks like a harvested code corpus or pull-request snapshot, not the repository's own working code.",
        )
    if is_vendored_dependency_path(path):
        return github_call_classification(
            row,
            "likely_false_positive",
            True,
            "vendored_dependency",
            0.96,
            "The path is inside a virtual environment, site-packages, vendored dependency, or copied dependency tree.",
        )
    if repo in OFFICIAL_PACKAGE_REPOS:
        return github_call_classification(
            row,
            "likely_false_positive",
            True,
            "upstream_library_self_reference",
            0.9,
            "The repository is the upstream package itself, so examples/tests/internal code are not external package usage.",
        )
    if looks_like_copied_upstream_package(path, package):
        return github_call_classification(
            row,
            "likely_false_positive",
            True,
            "copied_upstream_package_source",
            0.88,
            "The path looks like copied upstream package source or tests rather than project code using the package.",
        )
    if is_test_path(path):
        return github_call_classification(
            row,
            "needs_review",
            False,
            "test_or_benchmark_usage",
            0.64,
            "The call is in tests or benchmark code; it may be valid usage but may not represent the main analysis workflow.",
        )
    if is_tutorial_path(path):
        return github_call_classification(
            row,
            "needs_review",
            False,
            "tutorial_or_example_usage",
            0.62,
            "The call is in an example, demo, tutorial, notebook, or course path; it is usage, but likely pedagogical.",
        )
    if call_kind.endswith("_constructor") and call_kind != "surrogate_posterior_constructor":
        return github_call_classification(
            row,
            "needs_review",
            False,
            "constructor_only_or_supporting_call",
            0.58,
            "Constructor calls are useful context, but a separate fit/run/step/variational call is stronger evidence.",
        )
    if package == "stan" and function_name.endswith((".stanmodel", ".cmdstanmodel")):
        return github_call_classification(
            row,
            "needs_review",
            False,
            "stan_model_constructor_only",
            0.58,
            "The Stan model constructor is supporting evidence; model.variational(...) is the fitting call.",
        )
    return github_call_classification(
        row,
        "likely_true_positive",
        False,
        "project_code_usage",
        0.82,
        "The call appears in project code rather than vendored dependencies or upstream package internals.",
    )


def github_call_classification(
    row: dict[str, Any],
    usage_status: str,
    is_false_positive: bool,
    category: str,
    confidence: float,
    reason: str,
) -> GithubCallClassification:
    return GithubCallClassification(
        call_id=int(row["id"]),
        usage_status=usage_status,
        is_false_positive=is_false_positive,
        category=category,
        confidence=confidence,
        reason=reason,
    )


def save_github_advi_call_classification(conn: sqlite3.Connection, result: GithubCallClassification) -> None:
    conn.execute(
        """
        INSERT INTO github_advi_call_classifications(
            call_id, usage_status, is_false_positive, category, confidence,
            reason, classifier_version, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(call_id) DO UPDATE SET
            usage_status=excluded.usage_status,
            is_false_positive=excluded.is_false_positive,
            category=excluded.category,
            confidence=excluded.confidence,
            reason=excluded.reason,
            classifier_version=excluded.classifier_version,
            created_at=excluded.created_at
        """,
        (
            result.call_id,
            result.usage_status,
            int(result.is_false_positive),
            result.category,
            result.confidence,
            result.reason,
            GITHUB_CALL_CLASSIFIER_VERSION,
            utc_now(),
        ),
    )
    conn.commit()


def normalize_repo_path(path: str) -> str:
    return "/" + path.replace("\\", "/").casefold().strip("/")


def is_vendored_dependency_path(path: str) -> bool:
    return any(marker in path for marker in VENDORED_PATH_MARKERS)


def looks_like_copied_upstream_package(path: str, package: str) -> bool:
    if package == "tensorflow_probability" and "/tensorflow_probability/" in path:
        return True
    if package == "stan" and "/cmdstanpy/" in path:
        return True
    if package == "pymc" and ("/pymc/tests/" in path or "/pymc3/tests/" in path or "/pymc3/pymc3/" in path):
        return True
    if package == "pyro" and ("/pyro/tests/" in path or "/pyro/pyro/" in path):
        return True
    if package == "numpyro" and ("/numpyro/test/" in path or "/numpyro/numpyro/" in path):
        return True
    return any(marker in path for marker in UPSTREAM_PACKAGE_PATH_MARKERS) and (
        "/tests/" in path or "/test/" in path or "/site-packages/" in path
    )


def is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return any(marker in path for marker in TEST_PATH_MARKERS) or name.startswith("test_") or name.endswith("_test.py")


def is_tutorial_path(path: str) -> bool:
    return any(marker in path for marker in TUTORIAL_PATH_MARKERS)


def find_advi_calls(source: str) -> list[AdvICall]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return find_textual_advi_calls(source)
    lines = source.splitlines()
    import_aliases = collect_import_aliases(tree)
    vi_variables = collect_vi_variables(tree, import_aliases)
    expression_aliases = collect_expression_aliases(tree)
    parent_links = collect_parent_links(tree)
    calls: list[AdvICall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # File-level assignments (e.g. `if optimizer is None: optimizer = Adam(...)`)
        # take precedence over literal parameter defaults such as `optimizer=None`.
        scoped_aliases = dict(collect_enclosing_parameter_defaults(node, parent_links))
        scoped_aliases.update(expression_aliases)
        call = classify_advi_call(node, lines, import_aliases, vi_variables, scoped_aliases)
        if call:
            calls.append(call)
    return calls


def collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def collect_vi_variables(tree: ast.AST, import_aliases: dict[str, str]) -> dict[str, ViVariable]:
    variables: dict[str, ViVariable] = {}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        if not isinstance(value, ast.Call):
            continue
        resolved_name = resolve_call_name(value.func, import_aliases)
        keyword_arguments = {kw.arg or "**": expression_text(kw.value) for kw in value.keywords}
        positional_arguments = [expression_text(arg) for arg in value.args]
        method = infer_method(value, resolved_name.lower(), keyword_arguments, positional_arguments)
        package, call_kind = infer_package_and_kind(
            resolved_name.lower(),
            method,
            keyword_arguments,
            positional_arguments,
        )
        if not package:
            continue
        if call_kind not in {"advi_constructor", "svi_autoguide", "stan_model_constructor"}:
            continue
        for target in targets:
            for name in assigned_names(target):
                variables[name] = ViVariable(package=package, method=method)
    return variables


def expression_references_any_name(node: ast.AST, names: set[str]) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in names:
            return True
    return False


def collect_expression_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    aliases.update(collect_class_default_aliases(tree))
    aliases.update(collect_argparse_default_aliases(tree))
    # Class and argparse defaults were resolved with their own context; the
    # whole-tree walk below must not overwrite them.
    protected_keys = set(aliases)
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
            value = node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitOr):
            key = assignment_key(node.target)
            if key:
                text = expression_text(node.value)
                record_expression_alias(aliases, key, node.value, text, overwrite=False)
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            record_mapping_update_aliases(aliases, node.value)
            continue
        if value is None:
            continue
        text = expression_text(value)
        if not text:
            continue
        target_keys = {key for key in (assignment_key(t) for t in targets) if key}
        if target_keys and expression_references_any_name(value, target_keys):
            # Self-referential rebinding (e.g. `optimizer = getattr(x, optimizer)(...)`).
            # Substituting the earlier alias would mix values from unrelated scopes,
            # so keep the expression unresolved.
            resolved_text = text
        else:
            resolved_text = resolve_expression_alias_text(text, aliases) or text
        for target in targets:
            key = assignment_key(target)
            if key and key in protected_keys:
                continue
            if key:
                record_expression_alias(aliases, key, value, resolved_text)
    return aliases


def collect_argparse_default_aliases(tree: ast.AST) -> dict[str, str]:
    parser_defaults: dict[str, dict[str, str]] = {}
    parse_result_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                parser_name = call_name(node.func.value)
                destination = argparse_destination(node)
                default = argparse_default(node)
                if parser_name and destination and default is not None:
                    parser_defaults.setdefault(parser_name, {})[destination] = default
            elif node.func.attr == "parse_args":
                parent_name = call_name(node.func.value)
                if parent_name:
                    for candidate in assigned_call_targets(tree, node):
                        parse_result_names.add(candidate)
    if not parser_defaults or not parse_result_names:
        return {}
    merged_defaults: dict[str, str] = {}
    for defaults in parser_defaults.values():
        merged_defaults.update(defaults)
    aliases: dict[str, str] = {}
    for parse_result_name in parse_result_names:
        for destination, default in merged_defaults.items():
            aliases[f"{parse_result_name}.{destination}"] = default
    return aliases


def argparse_destination(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "dest":
            return expression_text(keyword.value).strip("'\"")
    for arg in node.args:
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            continue
        if arg.value.startswith("--"):
            return arg.value[2:].replace("-", "_")
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value.lstrip("-").replace("-", "_")
    return None


def argparse_default(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "default":
            return expression_text(keyword.value)
    return None


def assigned_call_targets(tree: ast.AST, call_node: ast.Call) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(tree):
        assigned: list[ast.expr] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            assigned = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            assigned = [node.target]
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            assigned = [node.target]
            value = node.value
        if value is call_node:
            for target in assigned:
                key = assignment_key(target)
                if key:
                    targets.append(key)
    return targets


def collect_class_default_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for class_node in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                default = class_attribute_default_text(stmt.value)
                if default:
                    aliases[f"self.{stmt.target.id}"] = default
            elif isinstance(stmt, ast.Assign):
                default = class_attribute_default_text(stmt.value)
                if not default:
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        aliases[f"self.{target.id}"] = default
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__":
                defaults = function_parameter_defaults(stmt)
                init_aliases: dict[str, str] = dict(defaults)
                for nested in ast.walk(stmt):
                    if isinstance(nested, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                        targets: list[ast.expr] = []
                        value: ast.AST | None = None
                        if isinstance(nested, ast.Assign):
                            targets = list(nested.targets)
                            value = nested.value
                        elif isinstance(nested, ast.AnnAssign):
                            targets = [nested.target]
                            value = nested.value
                        elif isinstance(nested, ast.NamedExpr):
                            targets = [nested.target]
                            value = nested.value
                        if value is None:
                            continue
                        value_text = expression_text(value)
                        resolved_value = resolve_expression_alias_text(value_text, init_aliases)
                        for target in targets:
                            key = assignment_key(target)
                            if key:
                                init_aliases[key] = resolved_value or value_text
                                if key.startswith("self."):
                                    aliases[key] = resolved_value or value_text
    return aliases


def class_attribute_default_text(value: ast.AST) -> str | None:
    if not isinstance(value, ast.Call):
        return expression_text(value)
    name = call_name(value.func)
    if name != "Field" and not name.endswith(".Field"):
        return expression_text(value)
    for keyword in value.keywords:
        if keyword.arg == "default_factory":
            if isinstance(keyword.value, ast.Lambda):
                return expression_text(keyword.value.body)
            return expression_text(keyword.value)
        if keyword.arg == "default":
            return expression_text(keyword.value)
    if value.args:
        return expression_text(value.args[0])
    return None


def collect_parent_links(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def collect_enclosing_parameter_defaults(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> dict[str, str]:
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(current)
    aliases: dict[str, str] = {}
    for function in reversed(functions):
        aliases.update(function_parameter_defaults(function))
    return aliases


def function_parameter_defaults(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    aliases: dict[str, str] = {}
    positional_args = [*function.args.posonlyargs, *function.args.args]
    defaults = [None] * (len(positional_args) - len(function.args.defaults)) + list(function.args.defaults)
    for arg, default in zip(positional_args, defaults):
        if default is not None:
            aliases[arg.arg] = expression_text(default)
    for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
        if default is not None:
            aliases[arg.arg] = expression_text(default)
    return aliases


def assignment_key(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return call_name(target)
    return None


def record_expression_alias(
    aliases: dict[str, str],
    key: str,
    value: ast.AST,
    text: str,
    *,
    overwrite: bool = True,
) -> None:
    if overwrite or key not in aliases:
        aliases[key] = text
    record_dict_entry_aliases(aliases, key, value)
    copy_mapping_aliases(aliases, key, value)
    record_self_mapping_aliases(aliases, key, value)


def record_mapping_update_aliases(aliases: dict[str, str], node: ast.Call) -> None:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "update":
        return
    target = call_name(node.func.value)
    if not target or not node.args:
        return
    record_expression_alias(aliases, target, node.args[0], expression_text(node.args[0]), overwrite=False)


def record_dict_entry_aliases(aliases: dict[str, str], base: str, value: ast.AST) -> None:
    for path, text in iter_dict_entry_texts(value):
        aliases[subscript_alias_key(base, path)] = text


def record_self_mapping_aliases(aliases: dict[str, str], key: str, value: ast.AST) -> None:
    entries = iter_dict_entry_texts(value)
    if not entries:
        return
    bases: list[str] = []
    leaf_attrs = False
    short_key = key.rsplit(".", 1)[-1]
    lower_key = short_key.lower()
    if not key.startswith("self."):
        bases.append(f"self.{key}")
    if "default" in lower_key and "kwargs" in lower_key:
        bases.extend(["self.kwargs", "self.update_kwargs", "self._update_kwargs"])
    if "default" in lower_key and "params" in lower_key:
        bases.extend(["self._default_params", "self.fitting_params"])
        leaf_attrs = True
    if key in {"self._default_params", "_default_params"} or key.endswith("._default_params"):
        bases.extend(["self._default_params", "self.fitting_params"])
        leaf_attrs = True
    for base in dict.fromkeys(bases):
        record_dict_entry_aliases(aliases, base, value)
    if leaf_attrs:
        for path, text in entries:
            if len(path) == 1 and isinstance(path[0], str) and path[0].isidentifier():
                aliases[f"self.{path[0]}"] = text


def copy_mapping_aliases(aliases: dict[str, str], target: str, value: ast.AST) -> None:
    for source in mapping_source_bases(value):
        prefix = f"{source}["
        for alias_key, alias_value in list(aliases.items()):
            if alias_key.startswith(prefix):
                aliases[f"{target}{alias_key[len(source):]}"] = alias_value


def mapping_source_bases(value: ast.AST) -> list[str]:
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, ast.Attribute):
        name = call_name(value)
        return [name] if name else []
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        return [*mapping_source_bases(value.left), *mapping_source_bases(value.right)]
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr in {"copy"}:
        name = call_name(value.func.value)
        return [name] if name else []
    if isinstance(value, ast.Dict):
        sources: list[str] = []
        for key, item_value in zip(value.keys, value.values):
            if key is None:
                sources.extend(mapping_source_bases(item_value))
        return sources
    return []


def iter_dict_entry_texts(value: ast.AST, prefix: tuple[str | int | float | bool | None, ...] = ()) -> list[tuple[tuple[str | int | float | bool | None, ...], str]]:
    entries: list[tuple[tuple[str | int | float | bool | None, ...], str]] = []
    for key, item_value in mapping_items(value):
        path = (*prefix, key)
        entries.append((path, expression_text(item_value)))
        entries.extend(iter_dict_entry_texts(item_value, path))
    return entries


def mapping_items(value: ast.AST) -> list[tuple[str | int | float | bool | None, ast.AST]]:
    if isinstance(value, ast.Dict):
        items: list[tuple[str | int | float | bool | None, ast.AST]] = []
        for key, item_value in zip(value.keys, value.values):
            literal = literal_mapping_key(key)
            if literal is not None:
                items.append((literal, item_value))
        return items
    if isinstance(value, ast.Call) and call_name(value.func).endswith("dict"):
        items = []
        for arg in value.args:
            items.extend(mapping_items(arg))
        for keyword in value.keywords:
            if keyword.arg:
                items.append((keyword.arg, keyword.value))
        return items
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        return [*mapping_items(value.left), *mapping_items(value.right)]
    return []


def literal_mapping_key(node: ast.AST | None) -> str | int | float | bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    return None


def literal_subscript_value(node: ast.Subscript) -> ast.AST | None:
    key = literal_mapping_key(node.slice)
    if key is None or not isinstance(node.value, ast.Dict):
        return None
    for item_key, item_value in zip(node.value.keys, node.value.values):
        if literal_mapping_key(item_key) == key:
            return item_value
    return None


def subscript_alias_key(base: str, path: tuple[str | int | float | bool | None, ...]) -> str:
    suffix = "".join(f"[{mapping_key_text(part)}]" for part in path)
    return f"{base}{suffix}"


def mapping_key_text(key: str | int | float | bool | None) -> str:
    return repr(key)


def pymc_advi_constructor_method(lowered: str) -> str | None:
    if lowered.endswith(".fullrankadvi") or lowered == "fullrankadvi":
        return "FullRankADVI"
    if lowered.endswith(".advi") or lowered == "advi":
        return "ADVI"
    return None


def assigned_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(assigned_names(element))
        return names
    return []


def classify_advi_call(
    node: ast.Call,
    lines: list[str],
    import_aliases: dict[str, str] | None = None,
    vi_variables: dict[str, ViVariable] | None = None,
    expression_aliases: dict[str, str] | None = None,
) -> AdvICall | None:
    import_aliases = import_aliases or {}
    vi_variables = vi_variables or {}
    expression_aliases = expression_aliases or {}
    name = call_name(node.func)
    resolved_name = resolve_call_name(node.func, import_aliases)
    lowered = resolved_name.lower()
    keyword_arguments = {kw.arg or "**": expression_text(kw.value) for kw in node.keywords}
    positional_arguments = [expression_text(arg) for arg in node.args]
    add_resolved_svi_setup_arguments(lowered, keyword_arguments, positional_arguments, expression_aliases)
    variable_call = vi_variable_fit_call(node, vi_variables)
    if not variable_call and is_pystan_vb_call(node, import_aliases):
        variable_call = ("stan", "stan_variational_method", "ADVI")
    if variable_call:
        package, call_kind, method = variable_call
        add_resolved_relevant_parameter_arguments(
            package,
            call_kind,
            lowered,
            keyword_arguments,
            positional_arguments,
            expression_aliases,
        )
        return AdvICall(
            line_number=getattr(node, "lineno", 0),
            package=package,
            call_kind=call_kind,
            function_name=name,
            method=method,
            changed_parameters=sorted(name for name in keyword_arguments if name != "**"),
            keyword_arguments=keyword_arguments,
            positional_arguments=positional_arguments,
            source_snippet=source_segment(lines, node),
        )
    method = infer_method(node, lowered, keyword_arguments, positional_arguments)
    # True when a bare name was qualified through an import (`from my_module import SVI`).
    # Attribute calls such as `model.svi(...)` are not import-qualified.
    is_import_qualified = isinstance(node.func, ast.Name) and import_aliases.get(node.func.id, node.func.id) != node.func.id
    package, call_kind = infer_package_and_kind(
        lowered, method, keyword_arguments, positional_arguments, is_import_qualified=is_import_qualified
    )
    if not package:
        return None
    add_resolved_relevant_parameter_arguments(
        package,
        call_kind,
        lowered,
        keyword_arguments,
        positional_arguments,
        expression_aliases,
    )
    changed = changed_parameter_names(package, lowered, keyword_arguments)
    return AdvICall(
        line_number=getattr(node, "lineno", 0),
        package=package,
        call_kind=call_kind,
        function_name=name,
        method=method,
        changed_parameters=changed,
        keyword_arguments=keyword_arguments,
        positional_arguments=positional_arguments,
        source_snippet=source_segment(lines, node),
    )


def is_pystan_vb_call(node: ast.Call, import_aliases: dict[str, str]) -> bool:
    """PyStan 2's ADVI is `model.vb(...)`. `vb` is too generic a name to trust on
    its own, so an untracked `.vb(...)` only counts when the file imports pystan
    (covers models loaded from a pickle or a compile-and-cache helper)."""
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "vb"
        and any(module.split(".", 1)[0] == "pystan" for module in import_aliases.values())
    )


def vi_variable_fit_call(node: ast.Call, vi_variables: dict[str, ViVariable]) -> tuple[str, str, str | None] | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    attr = node.func.attr
    if attr not in {"fit", "run", "step", "vb"}:
        return None
    if isinstance(node.func.value, ast.Name):
        variable = vi_variables.get(node.func.value.id)
        if not variable:
            return None
        if variable.package == "stan" and attr == "vb":
            return variable.package, "stan_variational_method", variable.method
        if variable.package == "pymc" and attr == "fit":
            return variable.package, "advi_fit_method", variable.method
        if variable.package == "numpyro" and attr == "run":
            return variable.package, "svi_run_method", variable.method
        if variable.package == "pyro" and attr == "step":
            return variable.package, "svi_step_method", variable.method
        if variable.package == "pyro_or_numpyro" and attr in {"run", "step"}:
            return variable.package, f"svi_{attr}_method", variable.method
        if variable.package == "stan" and attr == "variational":
            return variable.package, "stan_variational_method", variable.method
    if attr == "variational":
        return "stan", "stan_variational_method", "ADVI"
    return None


def infer_package_and_kind(
    lowered: str,
    method: str | None,
    keyword_arguments: dict[str, str],
    positional_arguments: list[str],
    *,
    is_import_qualified: bool = False,
) -> tuple[str | None, str]:
    all_args = " ".join([*keyword_arguments.values(), *positional_arguments]).lower()
    if is_pymc_fit_call(lowered) and (not method or "advi" in method):
        return "pymc", "fit"
    if lowered.endswith((".advi", ".fullrankadvi")) or lowered in {"advi", "fullrankadvi"}:
        return "pymc", "advi_constructor"
    if lowered.endswith(".svi") or lowered == "svi":
        if lowered.startswith("numpyro.") or ".numpyro." in lowered:
            return "numpyro", "svi_autoguide"
        if lowered.startswith("pyro.") or ".pyro." in lowered:
            return "pyro", "svi_autoguide"
        if "numpyro" in all_args or "autonormal" in all_args or "autodiagonalnormal" in all_args:
            return "numpyro", "svi_autoguide"
        if "pyro" in all_args or "auto" in all_args:
            return "pyro", "svi_autoguide"
        if is_import_qualified:
            # Imported from a module other than pyro/numpyro (e.g. a custom
            # `SVI` class), with no pyro/numpyro arguments: not a match.
            return None, ""
        return "pyro_or_numpyro", "svi_autoguide"
    if lowered.endswith(
        (
            ".autodiagonalnormal",
            ".automultivariatenormal",
            ".autonormal",
            ".autolowrankmultivariatenormal",
            ".autoiafnormal",
            ".autolaplaceapproximation",
            ".autoguidelist",
        )
    ):
        if "numpyro" in lowered:
            return "numpyro", "autoguide_constructor"
        if "pyro" in lowered:
            return "pyro", "autoguide_constructor"
        return "pyro_or_numpyro", "autoguide_constructor"
    if lowered.endswith(".fit_surrogate_posterior") or lowered == "fit_surrogate_posterior":
        return "tensorflow_probability", "fit_surrogate_posterior"
    if lowered.endswith((".build_factored_surrogate_posterior", ".build_affine_surrogate_posterior")):
        return "tensorflow_probability", "surrogate_posterior_constructor"
    if lowered.endswith((".stanmodel", ".cmdstanmodel")) or lowered in {"stanmodel", "cmdstanmodel"}:
        return "stan", "stan_model_constructor"
    if lowered.endswith(".variational") or lowered == "variational":
        return "stan", "stan_variational_method"
    return None, ""


def is_pymc_fit_call(lowered: str) -> bool:
    return lowered in {"pm.fit", "pymc.fit", "pymc3.fit"} or lowered.endswith(
        (".variational.fit", ".inference.fit")
    )


def infer_method(
    node: ast.Call,
    lowered: str,
    keyword_arguments: dict[str, str],
    positional_arguments: list[str],
) -> str | None:
    method = keyword_arguments.get("method")
    if method:
        return method.strip("'\"")
    if is_pymc_fit_call(lowered):
        return "ADVI(default)"
    constructor_method = pymc_advi_constructor_method(lowered)
    if constructor_method:
        return constructor_method
    if lowered.endswith(".svi") or lowered == "svi":
        return keyword_arguments.get("loss") or (positional_arguments[3] if len(positional_arguments) >= 4 else "SVI")
    if lowered.endswith(
        (
            ".autodiagonalnormal",
            ".autonormal",
            ".automultivariatenormal",
            ".autolowrankmultivariatenormal",
            ".autoiafnormal",
            ".autolaplaceapproximation",
            ".autoguidelist",
        )
    ):
        return lowered.rsplit(".", 1)[-1]
    for value in [*keyword_arguments.values(), *positional_arguments]:
        if "advi" in value.lower() or "autodiagonalnormal" in value.lower() or "autonormal" in value.lower():
            return value
    if lowered.endswith(".fit_surrogate_posterior") or lowered == "fit_surrogate_posterior":
        return "surrogate_posterior"
    if lowered.endswith((".stanmodel", ".cmdstanmodel", ".variational")) or lowered in {
        "stanmodel",
        "cmdstanmodel",
        "variational",
    }:
        return "ADVI"
    return None


def changed_parameter_names(package: str, lowered: str, keyword_arguments: dict[str, str]) -> list[str]:
    names = sorted(name for name in keyword_arguments if name != "**" and not name.startswith("__"))
    if package == "pymc" and (lowered.endswith(".fit") or lowered == "fit"):
        names = [name for name in names if name != "method"]
    return names


def add_resolved_svi_setup_arguments(
    lowered: str,
    keyword_arguments: dict[str, str],
    positional_arguments: list[str],
    expression_aliases: dict[str, str],
) -> None:
    if not (lowered.endswith(".svi") or lowered == "svi"):
        return
    optimizer = keyword_arguments.get("optim") or keyword_arguments.get("optimizer")
    if optimizer is None and len(positional_arguments) >= 3:
        optimizer = positional_arguments[2]
    loss = keyword_arguments.get("loss")
    if loss is None and len(positional_arguments) >= 4:
        loss = positional_arguments[3]
    resolved_optimizer = resolve_expression_alias_text(optimizer, expression_aliases) if optimizer else None
    resolved_loss = resolve_expression_alias_text(loss, expression_aliases) if loss else None
    if resolved_optimizer and resolved_optimizer != optimizer:
        keyword_arguments["__resolved_optimizer"] = resolved_optimizer
    if resolved_loss and resolved_loss != loss:
        keyword_arguments["__resolved_loss"] = resolved_loss
    num_particles = extract_elbo_num_particles_text(resolved_loss or loss)
    if num_particles:
        keyword_arguments["__resolved_num_particles"] = resolve_expression_alias_text(num_particles, expression_aliases) or num_particles


def extract_elbo_num_particles_text(loss_expression: str | None) -> str | None:
    if not loss_expression:
        return None
    try:
        expression = ast.parse(loss_expression, mode="eval")
    except SyntaxError:
        return None
    values: list[str] = []
    for node in ast.walk(expression):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node.func)
        if not is_standard_elbo_sample_call(name):
            continue
        value = elbo_call_num_particles(node)
        if value is not None:
            values.append(value)
    cleaned = list(dict.fromkeys(values))
    if len(cleaned) == 1:
        return cleaned[0]
    return None


def is_standard_elbo_sample_call(name: str) -> bool:
    lowered = name.lower()
    if "beta" in lowered or "tempered" in lowered:
        return False
    return lowered.endswith(
        (
            "trace_elbo",
            "tracemeanfield_elbo",
            "trace_meanfield_elbo",
            "tracegraph_elbo",
            "traceenum_elbo",
            "jittrace_elbo",
            "jittracegraph_elbo",
            "jittraceenum_elbo",
        )
    )


def elbo_call_num_particles(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "num_particles":
            return expression_text(keyword.value)
    if node.args:
        return expression_text(node.args[0])
    return "1"


def add_resolved_relevant_parameter_arguments(
    package: str,
    call_kind: str,
    lowered: str,
    keyword_arguments: dict[str, str],
    positional_arguments: list[str],
    expression_aliases: dict[str, str],
) -> None:
    del lowered
    if package == "pymc" and call_kind in {"fit", "advi_fit_method"}:
        add_resolved_keyword_argument(keyword_arguments, expression_aliases, "obj_optimizer")
        add_resolved_keyword_argument(keyword_arguments, expression_aliases, "obj_n_mc")
    elif package == "tensorflow_probability" and call_kind == "fit_surrogate_posterior":
        add_resolved_keyword_or_positional_argument(
            keyword_arguments,
            positional_arguments,
            expression_aliases,
            parameter="optimizer",
            positional_index=2,
        )
        add_resolved_keyword_or_positional_argument(
            keyword_arguments,
            positional_arguments,
            expression_aliases,
            parameter="sample_size",
            positional_index=6,
        )
    elif package == "stan" and call_kind == "stan_variational_method":
        add_resolved_keyword_argument(keyword_arguments, expression_aliases, "grad_samples")


def add_resolved_keyword_argument(
    keyword_arguments: dict[str, str],
    expression_aliases: dict[str, str],
    parameter: str,
) -> None:
    value = keyword_arguments.get(parameter)
    if value is None:
        return
    resolved = resolve_expression_alias_text(value, expression_aliases)
    if resolved and resolved != value:
        keyword_arguments[f"__resolved_{parameter}"] = resolved


def add_resolved_keyword_or_positional_argument(
    keyword_arguments: dict[str, str],
    positional_arguments: list[str],
    expression_aliases: dict[str, str],
    *,
    parameter: str,
    positional_index: int,
) -> None:
    value = keyword_arguments.get(parameter)
    if value is None and len(positional_arguments) > positional_index:
        value = positional_arguments[positional_index]
    if value is None:
        return
    resolved = resolve_expression_alias_text(value, expression_aliases)
    if resolved and resolved != value:
        keyword_arguments[f"__resolved_{parameter}"] = resolved


def resolve_expression_alias_text(text: str | None, aliases: dict[str, str], depth: int = 0) -> str | None:
    if not text or depth > 6:
        return text
    key = text.strip()
    if key not in aliases:
        return resolve_embedded_expression_aliases(text, aliases)
    resolved = aliases[key]
    if resolved == text:
        return resolved
    return resolve_expression_alias_text(resolved, aliases, depth + 1)


def resolve_embedded_expression_aliases(text: str, aliases: dict[str, str]) -> str:
    if not aliases:
        return text
    try:
        expression = ast.parse(text, mode="eval")
    except SyntaxError:
        return text
    resolver = EmbeddedAliasResolver(aliases)
    resolved = resolver.visit(expression)
    if not resolver.changed:
        return text
    ast.fix_missing_locations(resolved)
    try:
        return ast.unparse(resolved)
    except Exception:
        return text


class EmbeddedAliasResolver(ast.NodeTransformer):
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases
        self.changed = False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return self.replace_if_aliased(node, node.id)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        # Collapse `X if cond else Y` to the taken branch when `cond` resolves
        # to a known boolean (e.g. via a parameter default `closed_form_kl=True`).
        test_key = expression_text(node.test).strip()
        resolved_test = self.aliases.get(test_key)
        if resolved_test in ("True", "1"):
            self.changed = True
            return self.visit(node.body)
        if resolved_test in ("False", "0"):
            self.changed = True
            return self.visit(node.orelse)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        key = call_name(node)
        if key in self.aliases:
            return self.replace_if_aliased(node, key)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        key = expression_text(node)
        if key in self.aliases:
            return self.replace_if_aliased(node, key)
        node = self.generic_visit(node)
        literal_value = literal_subscript_value(node)
        if literal_value is not None:
            self.changed = True
            return ast.copy_location(literal_value, node)
        key = expression_text(node)
        if key in self.aliases:
            return self.replace_if_aliased(node, key)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        # Rewrite `getattr(module, 'Adam')(...)` as `module.Adam(...)` so the
        # string-based optimizer detection recognizes it.
        if (
            isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Name)
            and node.func.func.id == "getattr"
            and len(node.func.args) >= 2
            and isinstance(node.func.args[1], ast.Constant)
            and isinstance(node.func.args[1].value, str)
        ):
            attr_node = ast.Attribute(value=node.func.args[0], attr=node.func.args[1].value, ctx=ast.Load())
            node.func = ast.copy_location(attr_node, node.func)
            self.changed = True
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
            return node
        if not node.args:
            return node
        base = expression_text(node.func.value)
        lookup_key = literal_mapping_key(node.args[0])
        if lookup_key is None:
            return node
        alias_key = subscript_alias_key(base, (lookup_key,))
        if alias_key in self.aliases:
            return self.replace_if_aliased(node, alias_key)
        if len(node.args) >= 2:
            self.changed = True
            return ast.copy_location(node.args[1], node)
        return node

    def replace_if_aliased(self, node: ast.AST, key: str) -> ast.AST:
        if key not in self.aliases:
            return node
        replacement = expression_to_ast(self.aliases[key])
        if replacement is None:
            return node
        if expression_references_any_name(replacement, {key}):
            # The alias refers back to `key` itself; leave the name unresolved.
            return node
        self.changed = True
        return ast.copy_location(replacement, node)


def expression_to_ast(text: str) -> ast.AST | None:
    try:
        return ast.parse(text, mode="eval").body
    except SyntaxError:
        return None


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return ""


def resolve_call_name(node: ast.AST, import_aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return import_aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = resolve_call_name(node.value, import_aliases)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return resolve_call_name(node.func, import_aliases)
    return ""


def expression_text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def source_segment(lines: list[str], node: ast.AST) -> str:
    start = max(0, getattr(node, "lineno", 1) - 1)
    end = max(start + 1, getattr(node, "end_lineno", start + 1))
    return "\n".join(lines[start:end]).strip()


def find_textual_advi_calls(source: str) -> list[AdvICall]:
    calls = []
    for index, line in enumerate(source.splitlines(), start=1):
        lowered = line.lower()
        if any(term in lowered for term in ("advi(", "fullrankadvi(", "pm.fit", "pymc.fit", "fit_surrogate_posterior")):
            calls.append(
                AdvICall(
                    line_number=index,
                    package="unknown",
                    call_kind="textual_match",
                    function_name="unknown",
                    method=None,
                    changed_parameters=[],
                    keyword_arguments={},
                    positional_arguments=[],
                    source_snippet=line.strip(),
                )
            )
    return calls


def query_rows(conn: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(query).fetchall()]


def fetch_github_advi_report_tables(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    """Query the row tables that build_vi_github_analysis.py turns into the
    reports/*.csv outputs."""
    resolve_settings_companions(conn)
    return {
        "calls": query_rows(conn, GITHUB_ADVI_CALLS_QUERY),
        "relevant": query_rows(conn, GITHUB_ADVI_RELEVANT_PARAMETER_ANALYSIS_QUERY),
        "pymc_family": query_rows(conn, GITHUB_PYMC_ADVI_FAMILY_REPO_SUMMARY_QUERY),
        "unknown_parameters": query_rows(conn, GITHUB_ADVI_UNKNOWN_PARAMETER_REVIEW_QUERY),
    }


def fetch_duplicate_repo_cluster_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Comprehensive-only audit table: which needs_review repos are
    near-duplicates of each other, and which one is treated as canonical."""
    canonical_for = cluster_duplicate_needs_review_repos(conn)
    return [
        {"repo_full_name": repo, "canonical_repo_full_name": canonical, "is_canonical": int(repo == canonical)}
        for repo, canonical in sorted(canonical_for.items())
    ]


GITHUB_ADVI_CALLS_QUERY = """
SELECT
  c.repo_full_name,
  c.path,
  c.html_url,
  c.line_number,
  c.package,
  c.call_kind,
  c.function_name,
  c.method,
  c.changed_parameters,
  c.keyword_arguments,
  c.positional_arguments,
  COALESCE(k.usage_status, 'unclassified') AS usage_status,
  COALESCE(k.is_false_positive, 0) AS is_false_positive,
  COALESCE(k.category, 'unclassified') AS classification_category,
  k.confidence AS classification_confidence,
  k.reason AS classification_reason,
  c.source_snippet
FROM github_advi_calls c
LEFT JOIN github_advi_call_classifications k ON k.call_id = c.id
ORDER BY c.repo_full_name, c.path, c.line_number
"""


GITHUB_PYMC_ADVI_FAMILY_REPO_SUMMARY_QUERY = """
WITH pymc_fit AS (
  SELECT
    c.*,
    CASE
      WHEN lower(COALESCE(c.method, '')) LIKE '%fullrank%'
        OR lower(COALESCE(c.function_name, '')) LIKE '%fullrank%'
        THEN 'fullrank_advi'
      WHEN lower(COALESCE(c.method, '')) IN ('advi', 'meanfield', 'meanfieldadvi')
        OR lower(COALESCE(c.function_name, '')) LIKE '%advi%'
        THEN 'meanfield_advi'
      WHEN lower(COALESCE(c.method, ''))='dadvi'
        THEN 'dadvi_other'
      ELSE 'other_or_unknown'
    END AS advi_family
  FROM github_advi_calls c
  JOIN github_advi_call_classifications k ON k.call_id = c.id
  WHERE k.usage_status='likely_true_positive'
    AND c.package='pymc'
    AND c.call_kind IN ('fit', 'advi_fit_method')
),
repo_counts AS (
  SELECT
    repo_full_name,
    COUNT(*) AS total_pymc_advi_fit_call_sites,
    COUNT(DISTINCT path) AS files,
    SUM(CASE WHEN advi_family='fullrank_advi' THEN 1 ELSE 0 END) AS fullrank_call_sites,
    SUM(CASE WHEN advi_family='meanfield_advi' THEN 1 ELSE 0 END) AS meanfield_call_sites,
    SUM(CASE WHEN advi_family NOT IN ('fullrank_advi', 'meanfield_advi') THEN 1 ELSE 0 END) AS other_pymc_vi_call_sites,
    MIN(CASE WHEN advi_family='fullrank_advi' THEN html_url || '#L' || line_number END) AS example_fullrank_link,
    MIN(CASE WHEN advi_family='meanfield_advi' THEN html_url || '#L' || line_number END) AS example_meanfield_link,
    MIN(CASE WHEN advi_family NOT IN ('fullrank_advi', 'meanfield_advi') THEN html_url || '#L' || line_number END) AS example_other_link
  FROM pymc_fit
  GROUP BY repo_full_name
)
SELECT
  repo_counts.repo_full_name,
  r.html_url AS repo_url,
  r.description,
  r.stars,
  r.pushed_at,
  CASE
    WHEN fullrank_call_sites > 0 AND meanfield_call_sites > 0 THEN 'both_fullrank_and_meanfield'
    WHEN fullrank_call_sites > 0 THEN 'fullrank_only'
    WHEN meanfield_call_sites > 0 THEN 'meanfield_only'
    ELSE 'other_pymc_vi_only'
  END AS pymc_advi_family_scope,
  total_pymc_advi_fit_call_sites,
  fullrank_call_sites,
  meanfield_call_sites,
  other_pymc_vi_call_sites,
  files,
  example_fullrank_link,
  example_meanfield_link,
  example_other_link
FROM repo_counts
LEFT JOIN github_repositories r ON r.full_name = repo_counts.repo_full_name
ORDER BY
  CASE
    WHEN fullrank_call_sites > 0 AND meanfield_call_sites > 0 THEN 0
    WHEN fullrank_call_sites > 0 THEN 1
    WHEN meanfield_call_sites > 0 THEN 2
    ELSE 3
  END,
  fullrank_call_sites DESC,
  meanfield_call_sites DESC,
  repo_counts.repo_full_name
"""


GITHUB_ADVI_RELEVANT_PARAMETER_ANALYSIS_QUERY = """
WITH likely AS (
  SELECT
    c.id, c.repo_full_name, c.path, c.html_url, c.line_number, c.package, c.call_kind,
    c.function_name, c.source_snippet, sc.settings_call_id, sc.resolution_scope,
    s.path AS settings_path, s.line_number AS settings_line_number,
    COALESCE(s.keyword_arguments, '{}') AS keyword_arguments,
    COALESCE(s.positional_arguments, '[]') AS positional_arguments,
    s.method AS method,
    COALESCE(s.changed_parameters, '[]') AS changed_parameters
  FROM github_advi_calls c
  JOIN github_advi_call_classifications k ON k.call_id = c.id
  LEFT JOIN github_advi_settings_companions sc ON sc.call_id = c.id
  LEFT JOIN github_advi_calls s ON s.id = sc.settings_call_id
  WHERE k.usage_status='likely_true_positive'
),
manual_seed(repo_full_name, path, line_number, override_optimizer_expression, override_loss_expression, override_sample_or_mc_expression) AS (
  VALUES
    (
      'Abhi-Tiger/Housing_price_prediction',
      'regression-py/src/ta_lib/_vendor/taregression/BayesFramework/advi.py',
      83,
      'tf.optimizers.Adam(learning_rate=0.1)',
      NULL,
      NULL
    ),
    (
      'Manish-TA/housing-price-prediction',
      'src/ta_lib/_vendor/taregression/BayesFramework/advi.py',
      83,
      'tf.optimizers.Adam(learning_rate=0.1)',
      NULL,
      NULL
    ),
    (
      'dewansh-kumar-TA/housing-price-prediction',
      'src/ta_lib/_vendor/taregression/BayesFramework/advi.py',
      83,
      'tf.optimizers.Adam(learning_rate=0.1)',
      NULL,
      NULL
    ),
    (
      'paduruvignatejreddy-ta/Assignment5',
      'src/ta_lib/_vendor/taregression/BayesFramework/advi.py',
      83,
      'tf.optimizers.Adam(learning_rate=0.1)',
      NULL,
      NULL
    ),
    (
      'AdrianLangseth/TDT4173_Anomaly_Detection',
      'src/bnn/Net.py',
      84,
      'pyro.optim.Adam({"lr": lr})',
      NULL,
      NULL
    )
),
manual_overrides AS (
  SELECT * FROM manual_seed
  UNION ALL
  SELECT
    repo_full_name,
    path,
    line_number,
    override_optimizer_expression,
    override_loss_expression,
    override_sample_or_mc_expression
  FROM github_advi_parameter_overrides
),
target_rows AS (
  SELECT
    likely.*,
    COALESCE(manual_overrides.override_optimizer_expression, settings_overrides.override_optimizer_expression) AS override_optimizer_expression,
    COALESCE(manual_overrides.override_loss_expression, settings_overrides.override_loss_expression) AS override_loss_expression,
    COALESCE(manual_overrides.override_sample_or_mc_expression, settings_overrides.override_sample_or_mc_expression) AS override_sample_or_mc_expression,
    CASE
      WHEN settings_call_id IS NULL THEN NULL
      WHEN package='pymc' THEN 'pymc_fit'
      WHEN package='tensorflow_probability' THEN 'tfp_fit_surrogate_posterior'
      WHEN package='stan' THEN 'stan_variational'
      WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN package || '_svi_constructor'
      ELSE NULL
    END AS analysis_family
  FROM likely
  LEFT JOIN manual_overrides
    ON manual_overrides.repo_full_name=likely.repo_full_name
   AND manual_overrides.path=likely.path
   AND manual_overrides.line_number=likely.line_number
  LEFT JOIN manual_overrides AS settings_overrides
    ON settings_overrides.repo_full_name=likely.repo_full_name
   AND settings_overrides.path=likely.settings_path
   AND settings_overrides.line_number=likely.settings_line_number
),
annotated AS (
  SELECT
    *,
    CASE
      WHEN package='pymc' THEN json_array(
        CASE WHEN EXISTS (SELECT 1 FROM json_each(changed_parameters) WHERE value='obj_optimizer') THEN 'obj_optimizer' END,
        CASE WHEN EXISTS (SELECT 1 FROM json_each(changed_parameters) WHERE value='obj_n_mc') THEN 'obj_n_mc' END
      )
      WHEN package='tensorflow_probability' THEN json_array(
        CASE WHEN EXISTS (SELECT 1 FROM json_each(changed_parameters) WHERE value='optimizer') THEN 'optimizer' END,
        CASE WHEN EXISTS (SELECT 1 FROM json_each(changed_parameters) WHERE value='sample_size') THEN 'sample_size' END
      )
      WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN json_array('optimizer_setup', 'loss_setup')
      ELSE json_array()
    END AS relevant_parameters_raw
  FROM target_rows
  WHERE analysis_family IS NOT NULL
),
setup AS (
  SELECT
    *,
    COALESCE(
      override_optimizer_expression,
      json_extract(keyword_arguments, '$.__resolved_optimizer'),
      json_extract(keyword_arguments, '$.optim'),
      json_extract(keyword_arguments, '$.optimizer'),
      json_extract(positional_arguments, '$[2]'),
      ''
    ) AS resolved_optimizer_expression,
    COALESCE(
      override_loss_expression,
      json_extract(keyword_arguments, '$.__resolved_loss'),
      json_extract(keyword_arguments, '$.loss'),
      json_extract(positional_arguments, '$[3]'),
      ''
    ) AS resolved_loss_expression,
    COALESCE(
      override_optimizer_expression,
      json_extract(keyword_arguments, '$.__resolved_obj_optimizer'),
      json_extract(keyword_arguments, '$.obj_optimizer'),
      ''
    ) AS resolved_pymc_obj_optimizer_expression,
    COALESCE(
      override_sample_or_mc_expression,
      json_extract(keyword_arguments, '$.__resolved_obj_n_mc'),
      json_extract(keyword_arguments, '$.obj_n_mc'),
      ''
    ) AS resolved_pymc_obj_n_mc_expression,
    COALESCE(
      override_optimizer_expression,
      json_extract(keyword_arguments, '$.__resolved_optimizer'),
      json_extract(keyword_arguments, '$.optimizer'),
      json_extract(positional_arguments, '$[2]'),
      ''
    ) AS resolved_tfp_optimizer_expression,
    COALESCE(
      override_sample_or_mc_expression,
      json_extract(keyword_arguments, '$.__resolved_sample_size'),
      json_extract(keyword_arguments, '$.sample_size'),
      json_extract(positional_arguments, '$[6]'),
      ''
    ) AS resolved_tfp_sample_size_expression,
    COALESCE(
      override_sample_or_mc_expression,
      json_extract(keyword_arguments, '$.__resolved_num_particles'),
      ''
    ) AS resolved_svi_num_particles_expression,
    COALESCE(
      override_sample_or_mc_expression,
      json_extract(keyword_arguments, '$.__resolved_grad_samples'),
      json_extract(keyword_arguments, '$.grad_samples'),
      ''
    ) AS resolved_stan_grad_samples_expression,
    COALESCE(json_extract(keyword_arguments, '$.tol_rel_obj'), '') AS resolved_stan_tol_rel_obj_expression,
    COALESCE(json_extract(keyword_arguments, '$.adapt_engaged'), '') AS resolved_stan_adapt_engaged_expression
  FROM annotated
),
status AS (
  SELECT
    *,
    replace(replace(replace(replace(COALESCE(resolved_optimizer_expression, ''), ' ', ''), char(10), ''), char(13), ''), char(9), '') AS compact_optimizer_expression,
    replace(replace(replace(replace(COALESCE(resolved_loss_expression, ''), ' ', ''), char(10), ''), char(13), ''), char(9), '') AS compact_loss_expression,
    replace(replace(replace(replace(COALESCE(resolved_tfp_optimizer_expression, ''), ' ', ''), char(10), ''), char(13), ''), char(9), '') AS compact_tfp_optimizer_expression,
    replace(replace(replace(replace(COALESCE(resolved_svi_num_particles_expression, ''), ' ', ''), char(10), ''), char(13), ''), char(9), '') AS compact_svi_num_particles_expression,
    replace(replace(replace(replace(COALESCE(resolved_stan_grad_samples_expression, ''), ' ', ''), char(10), ''), char(13), ''), char(9), '') AS compact_stan_grad_samples_expression
  FROM setup
),
classified_setup AS (
  SELECT
    *,
    CASE
      WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN CASE
        WHEN compact_optimizer_expression='' THEN 'optimizer_variable_unresolved'
        WHEN lower(compact_optimizer_expression) LIKE '%non_adam_optimizer%'
          OR lower(compact_optimizer_expression) LIKE '%non-adam%'
          THEN 'explicit_non_adam_or_scheduler'
        WHEN lower(compact_optimizer_expression)='adam'
          OR lower(compact_optimizer_expression) GLOB 'adam(*)'
          OR lower(compact_optimizer_expression) GLOB '*.adam'
          OR lower(compact_optimizer_expression) GLOB '*.adam(*)'
          -- ClippedAdam counts as default alongside Adam: Pyro recommends
          -- "Adam or ClippedAdam by default" for SVI
          -- (https://pyro.ai/examples/svi_part_iv.html, section 2).
          OR lower(compact_optimizer_expression) LIKE '%clippedadam%'
          THEN 'explicit_adam'
        WHEN lower(compact_optimizer_expression) LIKE '%build_svi_optimizer%'
          OR lower(compact_optimizer_expression) LIKE '%generate_optimiser%'
          OR lower(compact_optimizer_expression) LIKE '%generate_optimizer%'
          OR lower(compact_optimizer_expression) LIKE '%self.optimizer%'
          OR lower(compact_optimizer_expression) LIKE "%['optimizer']%"
          OR lower(compact_optimizer_expression) LIKE '%["optimizer"]%'
          OR lower(compact_optimizer_expression) LIKE '%vi_config[%optimizer%'
          OR lower(compact_optimizer_expression) LIKE '%check_and_prepare%'
          OR lower(compact_optimizer_expression) LIKE '%[j]%'
          THEN 'optimizer_variable_unresolved'
        WHEN lower(compact_optimizer_expression) LIKE '%adamw%'
          OR lower(compact_optimizer_expression) LIKE '%adamax%'
          OR lower(compact_optimizer_expression) LIKE '%pyroadam%'
          OR lower(compact_optimizer_expression) LIKE 'adam(%'
          OR lower(compact_optimizer_expression) LIKE 'optim.adam(%'
          OR lower(compact_optimizer_expression) LIKE '%sgd%'
          OR lower(compact_optimizer_expression) LIKE '%rmsprop%'
          OR lower(compact_optimizer_expression) LIKE '%adagrad%'
          OR lower(compact_optimizer_expression) LIKE '%adadelta%'
          OR lower(compact_optimizer_expression) LIKE '%dctadam%'
          OR lower(compact_optimizer_expression) LIKE '%optax%'
          OR lower(compact_optimizer_expression) LIKE 'chain(%'
          OR lower(compact_optimizer_expression) LIKE '%chain(%'
          OR lower(compact_optimizer_expression) LIKE '%clip(%'
          OR lower(compact_optimizer_expression) LIKE '%lbfgs%'
          OR lower(compact_optimizer_expression) LIKE '%solver%'
          OR lower(compact_optimizer_expression) LIKE '%radam%'
          OR lower(compact_optimizer_expression) LIKE '%torch.optim.%'
          OR lower(compact_optimizer_expression) LIKE '%pyro.optim.%'
          OR lower(compact_optimizer_expression) LIKE '%numpyro.optim.%'
          OR lower(compact_optimizer_expression) LIKE '%multisteplr%'
          OR lower(compact_optimizer_expression) LIKE '%exponentiallr%'
          OR lower(compact_optimizer_expression) LIKE '%scheduler%'
          OR lower(compact_optimizer_expression) LIKE '%cosineannealing%'
          OR lower(compact_optimizer_expression) LIKE '%cosinescheduleadamoptimizer%'
          OR lower(compact_optimizer_expression) LIKE '%reduceonplateau%'
          OR lower(compact_optimizer_expression) LIKE '%reduce_on_plateau%'
          OR lower(compact_optimizer_expression) LIKE '%non_adam_optimizer%'
          OR lower(compact_optimizer_expression) LIKE '%non-adam%'
          THEN 'explicit_non_adam_or_scheduler'
        WHEN compact_optimizer_expression NOT GLOB '*(*'
          THEN 'optimizer_variable_unresolved'
        ELSE 'optimizer_variable_unresolved'
      END
      WHEN package='tensorflow_probability' THEN CASE
        WHEN compact_tfp_optimizer_expression='' THEN 'no_optimizer_argument'
        WHEN lower(compact_tfp_optimizer_expression) LIKE '%non_adam_optimizer%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%non-adam%'
          THEN 'explicit_non_adam_or_scheduler'
        WHEN lower(compact_tfp_optimizer_expression)='adam'
          OR lower(compact_tfp_optimizer_expression) GLOB 'adam(*)'
          OR lower(compact_tfp_optimizer_expression) GLOB '*.adam'
          OR lower(compact_tfp_optimizer_expression) GLOB '*.adam(*)'
          OR lower(compact_tfp_optimizer_expression) GLOB 'adamoptimizer(*)'
          OR lower(compact_tfp_optimizer_expression) GLOB '*.adamoptimizer(*)'
          OR lower(compact_tfp_optimizer_expression) LIKE "%optimizer='adam'%"
          OR lower(compact_tfp_optimizer_expression) LIKE '%optimizer="adam"%'
          THEN 'explicit_adam'
        WHEN lower(compact_tfp_optimizer_expression) LIKE '%self.optimizer%'
          OR lower(compact_tfp_optimizer_expression) LIKE "%['optimizer']%"
          OR lower(compact_tfp_optimizer_expression) LIKE '%["optimizer"]%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%vi_config[%optimizer%'
          THEN 'optimizer_variable_unresolved'
        WHEN lower(compact_tfp_optimizer_expression) LIKE '%sgd%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%rmsprop%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%adagrad%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%adadelta%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%lbfgs%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%solver%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%scheduler%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%reduceonplateau%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%reduce_on_plateau%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%non_adam_optimizer%'
          OR lower(compact_tfp_optimizer_expression) LIKE '%non-adam%'
          THEN 'explicit_non_adam_or_scheduler'
        WHEN compact_tfp_optimizer_expression NOT GLOB '*(*'
          THEN 'optimizer_variable_unresolved'
        ELSE 'optimizer_variable_unresolved'
      END
      ELSE ''
    END AS optimizer_relevant_status,
    CASE
      WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN CASE
        WHEN lower(compact_loss_expression) LIKE '%manual_nondefault_loss%'
          THEN 'explicit_relevant_nondefault_loss'
        WHEN compact_loss_expression IN ('''ELBO''', '"ELBO"')
          THEN 'explicit_no_arg_trace_elbo_or_meanfield'
        WHEN compact_loss_expression IN (
          'Trace_ELBO()',
          'pyro.infer.Trace_ELBO()',
          'numpyro.infer.Trace_ELBO()',
          'infer.Trace_ELBO()',
          'pyro_infer.Trace_ELBO()',
          'pr_infer.Trace_ELBO()',
          'TraceMeanField_ELBO()',
          'Trace_MeanField_ELBO()',
          'pyro.infer.TraceMeanField_ELBO()',
          'pyro.infer.Trace_MeanField_ELBO()',
          'numpyro.infer.TraceMeanField_ELBO()',
          'numpyro.infer.Trace_MeanField_ELBO()',
          'infer.TraceMeanField_ELBO()',
          'infer.Trace_MeanField_ELBO()',
          'Trace_ELBO(1)',
          'pyro.infer.Trace_ELBO(1)',
          'numpyro.infer.Trace_ELBO(1)',
          'infer.Trace_ELBO(1)',
          'pyro_infer.Trace_ELBO(1)',
          'pr_infer.Trace_ELBO(1)',
          'TraceMeanField_ELBO(1)',
          'Trace_MeanField_ELBO(1)',
          'pyro.infer.TraceMeanField_ELBO(1)',
          'pyro.infer.Trace_MeanField_ELBO(1)',
          'numpyro.infer.TraceMeanField_ELBO(1)',
          'numpyro.infer.Trace_MeanField_ELBO(1)',
          'infer.TraceMeanField_ELBO(1)',
          'infer.Trace_MeanField_ELBO(1)',
          'Trace_ELBO(num_particles=1)',
          'pyro.infer.Trace_ELBO(num_particles=1)',
          'numpyro.infer.Trace_ELBO(num_particles=1)',
          'infer.Trace_ELBO(num_particles=1)',
          'pyro_infer.Trace_ELBO(num_particles=1)',
          'pr_infer.Trace_ELBO(num_particles=1)',
          'TraceMeanField_ELBO(num_particles=1)',
          'Trace_MeanField_ELBO(num_particles=1)',
          'pyro.infer.TraceMeanField_ELBO(num_particles=1)',
          'pyro.infer.Trace_MeanField_ELBO(num_particles=1)',
          'numpyro.infer.TraceMeanField_ELBO(num_particles=1)',
          'numpyro.infer.Trace_MeanField_ELBO(num_particles=1)',
          'infer.TraceMeanField_ELBO(num_particles=1)',
          'infer.Trace_MeanField_ELBO(num_particles=1)'
        )
          THEN 'explicit_no_arg_trace_elbo_or_meanfield'
        WHEN compact_loss_expression GLOB 'Trace_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'pyro.infer.Trace_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'numpyro.infer.Trace_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'infer.Trace_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'pyro_infer.Trace_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'pr_infer.Trace_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'Trace_ELBO(*)'
          OR compact_loss_expression GLOB 'pyro.infer.Trace_ELBO(*)'
          OR compact_loss_expression GLOB 'numpyro.infer.Trace_ELBO(*)'
          OR compact_loss_expression GLOB 'infer.Trace_ELBO(*)'
          OR compact_loss_expression GLOB 'pyro_infer.Trace_ELBO(*)'
          OR compact_loss_expression GLOB 'pr_infer.Trace_ELBO(*)'
          OR compact_loss_expression GLOB 'TraceMeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'Trace_MeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'pyro.infer.TraceMeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'pyro.infer.Trace_MeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'numpyro.infer.TraceMeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'numpyro.infer.Trace_MeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'infer.TraceMeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'infer.Trace_MeanField_ELBO(num_particles=*)'
          OR compact_loss_expression GLOB 'TraceMeanField_ELBO(*)'
          OR compact_loss_expression GLOB 'Trace_MeanField_ELBO(*)'
          OR compact_loss_expression GLOB 'pyro.infer.TraceMeanField_ELBO(*)'
          OR compact_loss_expression GLOB 'pyro.infer.Trace_MeanField_ELBO(*)'
          OR compact_loss_expression GLOB 'numpyro.infer.TraceMeanField_ELBO(*)'
          OR compact_loss_expression GLOB 'numpyro.infer.Trace_MeanField_ELBO(*)'
          OR compact_loss_expression GLOB 'infer.TraceMeanField_ELBO(*)'
          OR compact_loss_expression GLOB 'infer.Trace_MeanField_ELBO(*)'
          THEN 'explicit_no_arg_trace_elbo_or_meanfield'
        WHEN compact_loss_expression='' OR compact_loss_expression NOT GLOB '*(*'
          THEN 'loss_variable_unresolved'
        ELSE 'explicit_adjusted_loss_or_other_elbo'
      END
      ELSE ''
    END AS loss_relevant_status,
    CASE
      WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN CASE
        WHEN compact_svi_num_particles_expression='' THEN CASE
          WHEN lower(compact_loss_expression) LIKE '%num_particles=%'
            AND (
              compact_loss_expression LIKE '%Trace_ELBO%'
              OR compact_loss_expression LIKE '%TraceMeanField_ELBO%'
              OR compact_loss_expression LIKE '%Trace_MeanField_ELBO%'
              OR compact_loss_expression LIKE '%TraceGraph_ELBO%'
              OR compact_loss_expression LIKE '%TraceEnum_ELBO%'
            )
            AND compact_loss_expression NOT LIKE '%BetaTraceELBO%'
            AND compact_loss_expression NOT LIKE '%TemperedTraceELBO%'
            AND compact_loss_expression NOT LIKE '%_TemperedTraceELBO%'
            THEN 'sample_or_mc_variable_unresolved'
          ELSE 'explicit_default_sample_or_mc'
        END
        WHEN lower(compact_svi_num_particles_expression) IN ('1', '1.0', 'none') THEN 'explicit_default_sample_or_mc'
        ELSE 'explicit_adjusted_sample_or_mc'
      END
      ELSE ''
    END AS sample_or_mc_relevant_status
  FROM status
)
SELECT
  repo_full_name,
  path,
  html_url,
  line_number,
  package,
  call_kind,
  analysis_family,
  function_name,
  COALESCE(method, '') AS method_or_loss,
  CASE
    WHEN package='pymc'
      THEN CASE WHEN EXISTS (
                  SELECT 1
                  FROM json_each(changed_parameters)
                  WHERE (
                      value='obj_optimizer'
                      AND lower(COALESCE(resolved_pymc_obj_optimizer_expression, '')) NOT LIKE '%adagrad_window%'
                    )
                    OR (
                      value='obj_n_mc'
                      AND lower(COALESCE(resolved_pymc_obj_n_mc_expression, '')) NOT IN ('', 'none', '1', '1.0')
                    )
                )
                THEN 'relevant_adjustment'
                ELSE 'implicit_default_for_relevant_parameters'
           END
    WHEN package='tensorflow_probability'
      THEN CASE WHEN lower(COALESCE(resolved_tfp_sample_size_expression, '')) NOT IN ('', 'none', '1', '1.0')
                  OR optimizer_relevant_status='explicit_non_adam_or_scheduler'
                THEN 'relevant_adjustment'
                WHEN optimizer_relevant_status='optimizer_variable_unresolved'
                THEN 'unknown_optimizer_setup'
                ELSE 'implicit_default_for_relevant_parameters'
           END
    WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro')
      THEN CASE
        WHEN optimizer_relevant_status='explicit_adam'
          AND loss_relevant_status!='loss_variable_unresolved'
          AND loss_relevant_status!='explicit_relevant_nondefault_loss'
          AND loss_relevant_status!='explicit_adjusted_loss_or_other_elbo'
          AND sample_or_mc_relevant_status='explicit_default_sample_or_mc'
          THEN 'implicit_default_for_relevant_parameters'
        WHEN optimizer_relevant_status='explicit_non_adam_or_scheduler'
          OR sample_or_mc_relevant_status='explicit_adjusted_sample_or_mc'
          OR loss_relevant_status='explicit_relevant_nondefault_loss'
          OR loss_relevant_status='explicit_adjusted_loss_or_other_elbo'
          OR loss_relevant_status='loss_variable_unresolved'
          OR optimizer_relevant_status='optimizer_variable_unresolved'
          THEN 'relevant_adjustment'
        ELSE 'unknown_optimizer_or_loss_setup'
      END
    -- Stan's ADVI has no optimizer choice; grad_samples (default 1, the MC
    -- draws per gradient) is the only setting scored. A value that isn't a
    -- numeric literal couldn't be traced and goes to review.
    WHEN package='stan'
      THEN CASE
        WHEN lower(compact_stan_grad_samples_expression) IN ('', '1', '1.0', 'none')
          THEN 'implicit_default_for_relevant_parameters'
        WHEN compact_stan_grad_samples_expression GLOB '[0-9]*'
          AND compact_stan_grad_samples_expression NOT GLOB '*[^0-9.]*'
          THEN 'relevant_adjustment'
        ELSE 'unknown_sample_or_mc_setup'
      END
    ELSE 'not_applicable'
  END AS relevant_parameter_mode,
  CASE
    WHEN package='pymc'
      THEN (
        SELECT json_group_array(value)
        FROM json_each(changed_parameters)
        WHERE (
            value='obj_optimizer'
            AND lower(COALESCE(resolved_pymc_obj_optimizer_expression, '')) NOT LIKE '%adagrad_window%'
          )
          OR (
            value='obj_n_mc'
            AND lower(COALESCE(resolved_pymc_obj_n_mc_expression, '')) NOT IN ('', 'none', '1', '1.0')
          )
      )
    WHEN package='tensorflow_probability'
      THEN CASE
        WHEN optimizer_relevant_status='explicit_non_adam_or_scheduler'
          AND lower(COALESCE(resolved_tfp_sample_size_expression, '')) NOT IN ('', 'none', '1', '1.0')
          THEN json_array('optimizer', 'sample_size')
        WHEN optimizer_relevant_status='explicit_non_adam_or_scheduler'
          THEN json_array('optimizer')
        WHEN lower(COALESCE(resolved_tfp_sample_size_expression, '')) NOT IN ('', 'none', '1', '1.0')
          THEN json_array('sample_size')
        ELSE json_array()
      END
    WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro')
      -- Record every adjustment present, not just the first one matched.
      -- Learning-rate changes alone don't count: only a different optimizer
      -- (anything other than Adam/ClippedAdam) is an optimizer adjustment.
      THEN (
        SELECT json_group_array(value) FROM (
          SELECT 'optimizer_choice' AS value
          WHERE optimizer_relevant_status='explicit_non_adam_or_scheduler'
          UNION ALL
          SELECT 'sample_or_mc'
          WHERE sample_or_mc_relevant_status='explicit_adjusted_sample_or_mc'
          UNION ALL
          SELECT 'loss_function'
          WHERE loss_relevant_status IN ('explicit_adjusted_loss_or_other_elbo', 'explicit_relevant_nondefault_loss')
        )
      )
    WHEN package='stan'
      THEN CASE
        WHEN compact_stan_grad_samples_expression GLOB '[0-9]*'
          AND compact_stan_grad_samples_expression NOT GLOB '*[^0-9.]*'
          AND compact_stan_grad_samples_expression NOT IN ('1', '1.0')
          THEN json_array('grad_samples')
        ELSE json_array()
      END
    ELSE json_array()
  END AS relevant_parameters,
  CASE
    WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN resolved_optimizer_expression
    WHEN package='tensorflow_probability' THEN resolved_tfp_optimizer_expression
    WHEN package='pymc' THEN resolved_pymc_obj_optimizer_expression
    ELSE ''
  END AS optimizer_expression,
  CASE
    WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN resolved_loss_expression
    ELSE ''
  END AS loss_expression,
  optimizer_relevant_status,
  loss_relevant_status,
  CASE
    WHEN package='tensorflow_probability' THEN resolved_tfp_sample_size_expression
    WHEN package='pymc' THEN resolved_pymc_obj_n_mc_expression
    WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro') THEN resolved_svi_num_particles_expression
    WHEN package='stan' THEN resolved_stan_grad_samples_expression
    ELSE ''
  END AS sample_or_mc_expression,
  resolved_stan_grad_samples_expression AS stan_grad_samples_expression,
  resolved_stan_tol_rel_obj_expression AS stan_tol_rel_obj_expression,
  resolved_stan_adapt_engaged_expression AS stan_adapt_engaged_expression,
  changed_parameters AS all_keyword_parameters,
  keyword_arguments,
  positional_arguments,
  source_snippet,
  COALESCE(settings_call_id, id) AS settings_source_call_id
FROM classified_setup
ORDER BY package, call_kind, relevant_parameter_mode DESC, repo_full_name, path, line_number
"""


GITHUB_ADVI_UNKNOWN_PARAMETER_REVIEW_QUERY = """
WITH relevant AS (
  SELECT *
  FROM (
    """ + GITHUB_ADVI_RELEVANT_PARAMETER_ANALYSIS_QUERY + """
  )
),
flagged AS (
  SELECT
    *,
    CASE
      WHEN relevant_parameter_mode LIKE 'unknown%' THEN 'classification_unknown'
      WHEN optimizer_relevant_status='optimizer_variable_unresolved' THEN 'optimizer_unresolved'
      WHEN loss_relevant_status='loss_variable_unresolved' THEN 'loss_unresolved'
      WHEN package='pymc'
        AND relevant_parameters LIKE '%obj_optimizer%'
        AND COALESCE(optimizer_expression, '') != ''
        AND lower(optimizer_expression) NOT LIKE '%adagrad_window%'
        AND lower(optimizer_expression) NOT IN ('pm.adam', 'pymc.adam')
        AND optimizer_expression NOT GLOB '*(*'
        THEN 'pymc_obj_optimizer_unresolved'
      WHEN package IN ('pymc', 'tensorflow_probability')
        AND relevant_parameters LIKE '%sample%'
        AND COALESCE(sample_or_mc_expression, '') != ''
        AND sample_or_mc_expression NOT GLOB '[0-9]*'
        AND sample_or_mc_expression NOT GLOB '*(*'
        THEN 'sample_or_mc_unresolved'
      WHEN package IN ('pyro', 'numpyro', 'pyro_or_numpyro')
        AND relevant_parameters LIKE '%sample_or_mc%'
        AND COALESCE(sample_or_mc_expression, '') != ''
        AND lower(sample_or_mc_expression) NOT IN ('1', '1.0', 'none')
        AND sample_or_mc_expression NOT GLOB '[0-9]*'
        AND sample_or_mc_expression NOT GLOB '*(*'
        THEN 'sample_or_mc_unresolved'
      ELSE ''
    END AS unresolved_reason
  FROM relevant
)
SELECT
  repo_full_name || ':' || path AS repo_identifier,
  line_number AS code_line_number,
  html_url AS github_url,
  package,
  call_kind,
  unresolved_reason,
  relevant_parameter_mode AS current_classification_mode,
  relevant_parameters AS current_relevant_parameters,
  optimizer_relevant_status,
  loss_relevant_status,
  optimizer_expression,
  loss_expression,
  sample_or_mc_expression,
  trim(replace(replace(replace(COALESCE(source_snippet, ''), char(13), ' '), char(10), ' '), '  ', ' ')) AS source_snippet
FROM flagged
WHERE unresolved_reason != ''
ORDER BY package, unresolved_reason, repo_identifier, code_line_number
"""
