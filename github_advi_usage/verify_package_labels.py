"""Check the Pyro / NumPyro label of every true-positive call site against the
imports of the file it's in.

The scanner names a call's package from the call's own (import-expanded) name,
e.g. `numpyro.infer.SVI`. When the name doesn't carry the package (`infer.SVI`
via `from pyroapi import infer`, a star import, an object passed in as a
parameter), it falls back to guessing from the arguments or labels the call
`pyro_or_numpyro`. This module re-labels each call from its file's imports:

  1. the import that provides the called name (e.g. `from numpyro.infer import SVI`,
     `from numpyro.infer import autoguide` for `autoguide.AutoNormal(...)`);
  2. otherwise, the file's imports if they mention only one of pyro / numpyro;
  3. otherwise `unknown` (imports both, or neither).

Fetches each file once from GitHub at the analyzed commit (needs GITHUB_TOKEN)
and caches it under data/package_check_files/, so reruns are offline:

    python -m github_advi_usage.verify_package_labels --db data/github-vi-software-stan.sqlite

Writes <db folder>/<db stem>/package_labels.csv (one row per call).
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import os
import re
import sqlite3
import time
from pathlib import Path

import requests

PACKAGE_RE = {"numpyro": re.compile(r"^numpyro(\.|$)"), "pyro": re.compile(r"^(pyro|pyroapi)(\.|$)")}


def module_package(module: str) -> str | None:
    for package, pattern in PACKAGE_RE.items():
        if pattern.search(module):
            return package
    return None


def import_map(tree: ast.AST) -> tuple[dict[str, str], set[str]]:
    """{bound name: package} for names bound by pyro/numpyro imports, and the
    set of packages imported anywhere in the file."""
    names: dict[str, str] = {}
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                package = module_package(alias.name)
                if package:
                    packages.add(package)
                    names[alias.asname or alias.name.split(".")[0]] = package
        elif isinstance(node, ast.ImportFrom) and node.module:
            package = module_package(node.module)
            if package:
                packages.add(package)
                for alias in node.names:
                    if alias.name != "*":
                        names[alias.asname or alias.name] = package
    return names, packages


def text_imports(source: str) -> set[str]:
    """Fallback for files that don't parse: packages named on import lines."""
    packages = set()
    for line in source.splitlines():
        match = re.match(r"\s*(?:from|import)\s+([\w.]+)", line)
        if match:
            package = module_package(match.group(1))
            if package:
                packages.add(package)
    return packages


def label(source: str, function_name: str) -> tuple[str, str]:
    """(package, how it was decided) for a call named `function_name` in `source`."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        packages = text_imports(source)
        if len(packages) == 1:
            return next(iter(packages)), "file imports (unparsed file)"
        return "unknown", "file doesn't parse; imports " + (" and ".join(sorted(packages)) or "neither")
    names, packages = import_map(tree)
    head = function_name.split(".")[0]
    if head in names:
        return names[head], f"import providing `{head}`"
    if len(packages) == 1:
        return next(iter(packages)), "only package imported in file"
    if packages:
        return "unknown", "file imports both pyro and numpyro"
    return "unknown", "file imports neither pyro nor numpyro"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/github-vi-software-stan.sqlite"))
    parser.add_argument("--cache", type=Path, default=Path("data/package_check_files"))
    args = parser.parse_args()
    out = args.db.parent / args.db.stem / "package_labels.csv"
    args.cache.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("GITHUB_TOKEN")
    env = Path(".env")
    if not token and env.exists():
        token = next((l.split("=", 1)[1].strip().strip("\"'") for l in env.read_text().splitlines()
                      if l.startswith("GITHUB_TOKEN=")), None)
    session = requests.Session()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    calls = conn.execute(
        """
        SELECT c.id, c.repo_full_name, c.path, c.html_url, c.line_number, c.package, c.function_name
        FROM github_advi_calls c JOIN github_advi_call_classifications k ON k.call_id = c.id
        WHERE k.usage_status = 'likely_true_positive' AND c.package IN ('pyro', 'numpyro', 'pyro_or_numpyro')
        ORDER BY c.repo_full_name, c.path
        """
    ).fetchall()
    files = sorted({(repo, path, re.search(r"/blob/([0-9a-f]{40})/", url).group(1)) for _id, repo, path, url, *_ in calls})
    print(f"{len(calls)} calls in {len(files)} files", flush=True)

    sources: dict[tuple[str, str], str | None] = {}
    for index, (repo, path, sha) in enumerate(files, 1):
        cached = args.cache / f"{repo.replace('/', '__')}@{sha}" / path
        if not cached.exists():
            for attempt in range(4):
                resp = session.get(f"https://api.github.com/repos/{repo}/contents/{path}", params={"ref": sha}, timeout=60)
                if resp.status_code in (403, 429) and resp.headers.get("x-ratelimit-remaining") == "0":
                    wait = int(resp.headers.get("x-ratelimit-reset", time.time() + 60)) - int(time.time()) + 5
                    print(f"rate limited; sleeping {wait}s", flush=True)
                    time.sleep(max(wait, 5))
                    continue
                break
            if resp.status_code == 200 and resp.json().get("encoding") == "base64":
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(base64.b64decode(resp.json()["content"]))
            else:
                sources[(repo, path)] = None
                print(f"[{index}/{len(files)}] {repo}/{path}: fetch failed ({resp.status_code})", flush=True)
                continue
            time.sleep(0.3)
        sources[(repo, path)] = cached.read_text(errors="replace")
        if index % 100 == 0:
            print(f"[{index}/{len(files)}] files checked", flush=True)

    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["call_id", "repo_full_name", "path", "line_number", "function_name",
                         "scanner_package", "verified_package", "how_decided"])
        for call_id, repo, path, _url, line, package, function_name in calls:
            source = sources.get((repo, path))
            verified, how = label(source, function_name) if source is not None else ("unknown", "file no longer on GitHub")
            writer.writerow([call_id, repo, path, line, function_name, package, verified, how])
    print(f"Wrote {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
