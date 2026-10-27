"""Resolve optimizer/loss/sample-count values that the static scan left
unresolved, by fetching the rest of the repository at the same commit and
following the value through callers, wrappers, and defaults."""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import settings
from .db import connect, init_db, utc_now
from .github_advi import (
    assignment_key,
    call_name,
    collect_expression_aliases,
    collect_parent_links,
    expression_text,
    function_parameter_defaults,
    record_expression_alias,
    resolve_expression_alias_text,
)


@dataclass(frozen=True)
class UnknownParameterRow:
    repo_full_name: str
    path: str
    line_number: int
    github_url: str
    package: str
    optimizer_expression: str
    loss_expression: str
    sample_or_mc_expression: str


@dataclass(frozen=True)
class CallableParam:
    name: str
    parameter: str
    index: int


@dataclass(frozen=True)
class DeepOverride:
    row: UnknownParameterRow
    optimizer_expression: str | None
    loss_expression: str | None
    sample_or_mc_expression: str | None
    reason: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-resolve GitHub ADVI unknown parameter rows.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--unknown-csv", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--max-files-per-repo", type=int, default=300)
    args = parser.parse_args()

    conn = connect(Path(args.db))
    init_db(conn)
    rows = read_unknown_rows(Path(args.unknown_csv))
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    resolver = GitHubRepoResolver(max_files_per_repo=args.max_files_per_repo)
    count = 0
    for index, row in enumerate(rows, start=args.offset + 1):
        print(f"[{index}] scanning {row.repo_full_name}:{row.path}:{row.line_number}", flush=True)
        override = resolver.resolve_row(row)
        if override:
            save_override(conn, override)
            count += 1
            print(
                f"Resolved {row.repo_full_name}:{row.path}:{row.line_number} "
                f"optimizer={override.optimizer_expression or ''} "
                f"loss={override.loss_expression or ''} "
                f"sample_or_mc={override.sample_or_mc_expression or ''}",
                flush=True,
            )
        if args.interval:
            time.sleep(args.interval)
    print(f"Saved {count} deep parameter override(s)", flush=True)


def read_unknown_rows(path: Path) -> list[UnknownParameterRow]:
    rows: list[UnknownParameterRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            repo_full_name, source_path = row["repo_identifier"].split(":", 1)
            rows.append(
                UnknownParameterRow(
                    repo_full_name=repo_full_name,
                    path=source_path,
                    line_number=int(row["code_line_number"]),
                    github_url=row["github_url"],
                    package=row["package"],
                    optimizer_expression=row.get("optimizer_expression") or "",
                    loss_expression=row.get("loss_expression") or "",
                    sample_or_mc_expression=row.get("sample_or_mc_expression") or "",
                )
            )
    return rows


def save_override(conn: sqlite3.Connection, override: DeepOverride) -> None:
    now = utc_now()
    conn.execute(
        """
        INSERT INTO github_advi_parameter_overrides(
            repo_full_name, path, line_number,
            override_optimizer_expression, override_loss_expression,
            override_sample_or_mc_expression, reason, source, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'deep_repo_caller_scan', ?, ?)
        ON CONFLICT(repo_full_name, path, line_number) DO UPDATE SET
            override_optimizer_expression=COALESCE(excluded.override_optimizer_expression, override_optimizer_expression),
            override_loss_expression=COALESCE(excluded.override_loss_expression, override_loss_expression),
            override_sample_or_mc_expression=COALESCE(excluded.override_sample_or_mc_expression, override_sample_or_mc_expression),
            reason=excluded.reason,
            source=excluded.source,
            updated_at=excluded.updated_at
        """,
        (
            override.row.repo_full_name,
            override.row.path,
            override.row.line_number,
            override.optimizer_expression,
            override.loss_expression,
            override.sample_or_mc_expression,
            override.reason,
            now,
            now,
        ),
    )
    conn.commit()


class GitHubRepoResolver:
    def __init__(self, *, max_files_per_repo: int = 300) -> None:
        self.max_files_per_repo = max_files_per_repo
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "github-advi-usage/0.1"})
        app_settings = settings()
        if app_settings.github_token:
            self.session.headers.update({"Authorization": f"Bearer {app_settings.github_token}"})
        self.cache: dict[tuple[str, str], dict[str, str]] = {}

    def resolve_row(self, row: UnknownParameterRow) -> DeepOverride | None:
        sha = github_blob_sha(row.github_url)
        if not sha:
            return None
        sources = self.repo_sources(row.repo_full_name, sha, row.path)
        target_source = sources.get(row.path)
        if not target_source:
            return None
        target = target_call_context(target_source, row.line_number)
        if not target:
            return None
        optimizer = self.resolve_expression(row, sources, target, row.optimizer_expression, kind="optimizer")
        loss = self.resolve_expression(row, sources, target, row.loss_expression, kind="loss")
        sample_or_mc = self.resolve_expression(
            row,
            sources,
            target,
            row.sample_or_mc_expression,
            kind="sample_or_mc",
        )
        if not optimizer and not loss and not sample_or_mc:
            return None
        return DeepOverride(
            row=row,
            optimizer_expression=optimizer,
            loss_expression=loss,
            sample_or_mc_expression=sample_or_mc,
            reason="repo caller/default chain",
        )

    def repo_sources(self, repo_full_name: str, sha: str, target_path: str) -> dict[str, str]:
        key = (repo_full_name, sha)
        if key in self.cache:
            return self.cache[key]
        tree_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{sha}"
        response = self.session.get(tree_url, params={"recursive": "1"}, timeout=60)
        response.raise_for_status()
        tree = response.json().get("tree", [])
        paths = [
            item["path"]
            for item in tree
            if item.get("type") == "blob"
            and item.get("path", "").endswith(".py")
            and int(item.get("size") or 0) <= 500_000
        ]
        selected = select_python_paths(paths, target_path, self.max_files_per_repo)
        sources: dict[str, str] = {}
        for path in selected:
            raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/{sha}/{path}"
            try:
                raw = self.session.get(raw_url, timeout=45)
                raw.raise_for_status()
            except requests.RequestException:
                continue
            sources[path] = raw.text
        self.cache[key] = sources
        return sources

    def resolve_expression(
        self,
        row: UnknownParameterRow,
        sources: dict[str, str],
        target: dict[str, Any],
        expression: str,
        *,
        kind: str,
    ) -> str | None:
        if not expression:
            return None
        direct = self.resolve_expression_in_target_scope(target, expression, kind=kind)
        if direct:
            return direct
        if expression_is_resolved_for_kind(expression, kind):
            return None
        candidates = initial_callable_params(target, expression)
        if not candidates:
            return None
        candidates = expand_wrapper_candidates(sources, candidates)
        resolved: list[str] = []
        for source in sources.values():
            resolved.extend(resolve_candidate_call_arguments(source, candidates, kind=kind))
        resolved.extend(resolve_repo_default_patterns(sources, expression))
        return choose_override_expression(resolved, kind=kind)

    def resolve_expression_in_target_scope(self, target: dict[str, Any], expression: str, *, kind: str) -> str | None:
        tree = target.get("tree")
        parents = target.get("parents")
        call = target.get("call")
        if not isinstance(tree, ast.AST):
            return None
        aliases = collect_expression_aliases(tree)
        if isinstance(call, ast.AST) and isinstance(parents, dict):
            aliases = enclosing_aliases(call, parents, base_aliases=aliases)
        resolved = resolve_expression_alias_text(expression, aliases)
        if resolved and resolved != expression and expression_is_valid_override(resolved, kind):
            return resolved
        return None


def select_python_paths(paths: list[str], target_path: str, max_files: int) -> list[str]:
    def score(path: str) -> tuple[int, str]:
        lower = path.lower()
        score_value = 50
        if path == target_path:
            score_value = 0
        elif path.split("/", 1)[0] == target_path.split("/", 1)[0]:
            score_value = 5
        elif any(term in lower for term in ("main", "train", "fit", "infer", "svi", "vi", "sampler", "model", "run")):
            score_value = 10
        return score_value, path

    ordered = sorted(paths, key=score)
    if target_path not in ordered:
        ordered.insert(0, target_path)
    return ordered[:max_files]


def github_blob_sha(url: str) -> str | None:
    match = re.search(r"/blob/([^/]+)/", url)
    return match.group(1) if match else None


def target_call_context(source: str, line_number: int) -> dict[str, Any] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    parents = collect_parent_links(tree)
    call_node: ast.Call | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            if start <= line_number <= end:
                call_node = node
                break
    if call_node is None:
        return None
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    classes: list[ast.ClassDef] = []
    current: ast.AST | None = call_node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(current)
        elif isinstance(current, ast.ClassDef):
            classes.append(current)
    return {
        "tree": tree,
        "parents": parents,
        "call": call_node,
        "function": functions[0] if functions else None,
        "class": classes[0] if classes else None,
    }


def initial_callable_params(target: dict[str, Any], expression: str) -> list[CallableParam]:
    function = target.get("function")
    class_node = target.get("class")
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    parameter = expression_to_parameter_name(expression)
    if not parameter:
        return []
    params = function_parameter_names(function)
    if parameter not in params:
        return []
    index = params.index(parameter)
    candidates: list[CallableParam] = []
    if function.name != "__init__":
        candidates.append(CallableParam(function.name, parameter, index))
    if function.name == "__init__" and isinstance(class_node, ast.ClassDef):
        candidates.append(CallableParam(class_node.name, parameter, index))
    return candidates


def function_parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args]]
    if names and names[0] in {"self", "cls"}:
        names = names[1:]
    names.extend(arg.arg for arg in function.args.kwonlyargs)
    return names


def expression_to_parameter_name(expression: str) -> str | None:
    expression = expression.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expression):
        return expression
    if expression.startswith("self.") and re.fullmatch(r"self\.[A-Za-z_][A-Za-z0-9_]*", expression):
        return expression.split(".", 1)[1]
    return None


def expand_wrapper_candidates(sources: dict[str, str], seeds: list[CallableParam]) -> list[CallableParam]:
    candidates = list(seeds)
    seen = {(candidate.name, candidate.parameter, candidate.index) for candidate in candidates}
    for _ in range(3):
        added = False
        for source in sources.values():
            for wrapper in discover_wrappers(source, candidates):
                key = (wrapper.name, wrapper.parameter, wrapper.index)
                if key not in seen:
                    seen.add(key)
                    candidates.append(wrapper)
                    added = True
        if not added:
            break
    return candidates


def discover_wrappers(source: str, candidates: list[CallableParam]) -> list[CallableParam]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    parents = collect_parent_links(tree)
    wrappers: list[CallableParam] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        candidate = matching_candidate(node, candidates)
        if candidate is None:
            continue
        actual = call_argument_for_parameter(node, candidate)
        parameter = expression_to_parameter_name(actual or "")
        if not parameter:
            continue
        function = enclosing_function(node, parents)
        if not function:
            continue
        params = function_parameter_names(function)
        if parameter not in params:
            continue
        wrappers.append(CallableParam(function.name, parameter, params.index(parameter)))
    return wrappers


def resolve_candidate_call_arguments(source: str, candidates: list[CallableParam], *, kind: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    parents = collect_parent_links(tree)
    global_aliases = collect_expression_aliases(tree)
    resolved: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        candidate = matching_candidate(node, candidates)
        if candidate is None:
            continue
        actual = call_argument_for_parameter(node, candidate)
        if not actual:
            continue
        aliases = enclosing_aliases(node, parents, base_aliases=global_aliases)
        resolved_text = resolve_expression_alias_text(actual, aliases)
        if resolved_text and expression_is_resolved_for_kind(resolved_text, kind):
            resolved.append(resolved_text)
    return resolved


def matching_candidate(node: ast.Call, candidates: list[CallableParam]) -> CallableParam | None:
    name = call_name(node.func)
    short = name.rsplit(".", 1)[-1]
    for candidate in candidates:
        if short == candidate.name:
            return candidate
    return None


def call_argument_for_parameter(node: ast.Call, candidate: CallableParam) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == candidate.parameter:
            return expression_text(keyword.value)
    if len(node.args) > candidate.index:
        return expression_text(node.args[candidate.index])
    return None


def enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return None


def enclosing_aliases(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    *,
    base_aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    aliases: dict[str, str] = dict(base_aliases or {})
    function = enclosing_function(node, parents)
    if function:
        aliases.update(function_parameter_defaults(function))
        for nested in ast.walk(function):
            if isinstance(nested, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = list(nested.targets) if isinstance(nested, ast.Assign) else [nested.target]
                value = nested.value
                if value is None:
                    continue
                text = expression_text(value)
                resolved = resolve_expression_alias_text(text, aliases)
                for target in targets:
                    key = assignment_key(target)
                    if key:
                        record_expression_alias(aliases, key, value, resolved or text)
    return aliases


def resolve_repo_default_patterns(sources: dict[str, str], expression: str) -> list[str]:
    parameter = expression_to_parameter_name(expression)
    if not parameter:
        return []
    resolved: list[str] = []
    pattern = re.compile(rf"\b{re.escape(parameter)}\s*=\s*([A-Za-z0-9_\.]*Adam\s*\([^#\n]*\))")
    for source in sources.values():
        for match in pattern.finditer(source):
            resolved.append(match.group(1).strip())
    return resolved


def expression_is_resolved(expression: str) -> bool:
    compact = expression.replace(" ", "")
    return "(" in compact or "Adam" in compact or "ELBO" in compact or compact in {"None", "False", "True"}


def expression_is_resolved_for_kind(expression: str, kind: str) -> bool:
    if kind == "sample_or_mc":
        return sample_expression_is_concrete(expression)
    return expression_is_resolved(expression)


def sample_expression_is_concrete(expression: str) -> bool:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    def is_allowed(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return is_allowed(node.body)
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.UAdd, ast.USub)) and is_allowed(node.operand)
        if isinstance(node, ast.BinOp):
            return isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
            ) and is_allowed(node.left) and is_allowed(node.right)
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            return name in {"int", "float", "round"} and all(is_allowed(arg) for arg in node.args)
        return False

    return is_allowed(parsed)


def choose_override_expression(expressions: list[str], *, kind: str) -> str | None:
    cleaned = [expr for expr in dict.fromkeys(expressions) if expression_is_valid_override(expr, kind)]
    if not cleaned:
        return None
    adam = [expr for expr in cleaned if re.search(r"(^|[.])Adam\s*\(", expr)]
    if adam and len(adam) == len(cleaned):
        return adam[0]
    if len(cleaned) == 1:
        return cleaned[0]
    non_adam = [expr for expr in cleaned if not re.search(r"(^|[.])Adam\s*\(", expr)]
    if len(non_adam) == 1:
        return non_adam[0]
    return None


def expression_is_valid_override(expression: str | None, kind: str) -> bool:
    if not expression:
        return False
    if kind == "sample_or_mc":
        return sample_expression_is_concrete(expression)
    if not expression_is_resolved(expression):
        return False
    if kind == "loss":
        return loss_expression_is_concrete(expression)
    compact = expression.replace(" ", "").lower()
    return any(marker in compact for marker in ("adam", "sgd", "adagrad", "adadelta", "rmsprop", "optax.", "optim."))


def loss_expression_is_concrete(expression: str) -> bool:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    return concrete_loss_call(parsed.body)


def concrete_loss_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = call_name(node.func)
    if "ELBO" not in name and "elbo" not in name:
        return False
    return all(concrete_literal_expression(arg) for arg in node.args) and all(
        concrete_literal_expression(keyword.value)
        for keyword in node.keywords
        if keyword.arg is not None
    )


def concrete_literal_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, float, bool)) or node.value is None
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)) and concrete_literal_expression(node.operand)
    if isinstance(node, ast.BinOp):
        return isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ) and concrete_literal_expression(node.left) and concrete_literal_expression(node.right)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(concrete_literal_expression(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None
            and concrete_literal_expression(key)
            and concrete_literal_expression(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.Call):
        name = call_name(node.func)
        return name in {"int", "float", "round"} and all(concrete_literal_expression(arg) for arg in node.args)
    return False


if __name__ == "__main__":
    main()
