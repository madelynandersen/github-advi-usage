"""Trace values that the scan couldn't read statically (e.g. `Adam({"lr": lr})`)
through each repository's own source, so the default analysis can classify them.

Works on repository snapshots downloaded at the analyzed commit
(data/repo_snapshots/<owner>__<repo>@<sha>/), so it needs no network access:

    python -m github_advi_usage.trace_unresolved_values \
        --db data/github-vi-software-stan.sqlite \
        --classification data/github-vi-software-stan/vignette_default_reports/strict_call_site_classification.csv

For every call involved in an unresolved call site (the settings call plus the
guide / run / constructor calls linked to it), each argument is rewritten with
its variables replaced by the values they hold, followed through:

  local      the last assignment before the call in the same function
  loop       a `for x in [a, b, ...]` loop over literal values
  parameter  a function parameter's default, plus every value callers in the repo pass
  module     a module-level constant in the same file
  import     a constant imported from another file in the repo
  self       `self.x = ...` in the same class (then traced from there)
  argparse   an argparse `add_argument` (or absl flag) default
  class      a class-level attribute, e.g. a dataclass config field
  dict       a key in a dict literal (in the same file, then the repo)
  config     a key in a YAML / JSON config file in the repo

A runtime default counts as the value used unless a caller passes another
value. When several values are possible, all are kept (as `a if __traced__
else b`); the analysis counts a setting as default if any of them is accepted.

Writes:
  <db folder>/<db stem>/traced_values.json            call id -> traced arguments
  <db folder>/<db stem>/traced_value_evidence.csv     one row per traced variable
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # optional: YAML config files
except ImportError:  # pragma: no cover
    yaml = None

MAX_ALTERNATIVES = 8
MAX_DEPTH = 8
TRACED = ast.Name(id="__traced__", ctx=ast.Load())
CONFIG_SUFFIXES = (".yaml", ".yml", ".json")
ARGPARSE_OBJECTS = {"args", "opt", "opts", "options", "flags", "FLAGS", "parsed_args", "arguments", "cli_args", "hparams", "hp"}


def last_name(node: ast.AST) -> str:
    return ast.unparse(node).rsplit(".", 1)[-1]


def literal_node(value: Any) -> ast.expr | None:
    """A Python value from a config file as an AST node (scalars only)."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return ast.Constant(value=value)
    if isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return ast.Tuple(elts=[ast.Constant(value=v) for v in value], ctx=ast.Load())
    return None


def alternatives_node(options: list[ast.expr]) -> ast.expr:
    """Several possible values -> `a if __traced__ else (b if __traced__ else c)`."""
    node = options[-1]
    for option in reversed(options[:-1]):
        node = ast.IfExp(test=TRACED, body=option, orelse=node)
    return node


def dedupe(nodes: Iterable[ast.expr]) -> list[ast.expr]:
    seen, out = set(), []
    for node in nodes:
        key = ast.unparse(node)
        if key not in seen:
            seen.add(key)
            out.append(node)
    return out[:MAX_ALTERNATIVES]


class SourceFile:
    def __init__(self, path: Path, rel: str):
        self.path, self.rel = path, rel
        try:
            self.tree = ast.parse(path.read_text(errors="replace"))
        except (SyntaxError, ValueError):
            self.tree = None
        self.parents: dict[int, ast.AST] = {}
        if self.tree is not None:
            for parent in ast.walk(self.tree):
                for child in ast.iter_child_nodes(parent):
                    self.parents[id(child)] = parent

    def enclosing(self, node: ast.AST, kinds) -> ast.AST | None:
        current = self.parents.get(id(node))
        while current is not None and not isinstance(current, kinds):
            current = self.parents.get(id(current))
        return current

    def function_at(self, line: int) -> ast.AST | None:
        best = None
        if self.tree is None:
            return None
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= (node.end_lineno or node.lineno):
                if best is None or node.lineno > best.lineno:
                    best = node
        return best


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.files: dict[str, SourceFile] = {}
        for path in root.rglob("*.py"):
            rel = str(path.relative_to(root))
            self.files[rel] = SourceFile(path, rel)
        self._index()
        self._configs: dict[str, list[tuple[ast.expr, str]]] | None = None

    def _index(self) -> None:
        self.calls_by_name: dict[str, list[tuple[ast.Call, SourceFile]]] = defaultdict(list)
        self.argparse: dict[str, list[tuple[ast.expr, SourceFile]]] = defaultdict(list)
        self.self_attrs: dict[str, list[tuple[ast.expr, SourceFile, ast.AST]]] = defaultdict(list)
        self.class_attrs: dict[str, list[tuple[ast.expr, SourceFile]]] = defaultdict(list)
        self.dict_keys: dict[str, list[tuple[ast.expr, SourceFile]]] = defaultdict(list)
        self.module_assigns: dict[str, dict[str, list[ast.Assign]]] = {}
        for rel, src in self.files.items():
            assigns: dict[str, list] = defaultdict(list)
            self.module_assigns[rel] = assigns
            if src.tree is None:
                continue
            for stmt in src.tree.body:
                for target, value in assignment_pairs(stmt):
                    if isinstance(target, ast.Name):
                        assigns[target.id].append((stmt.lineno, value))
            for node in ast.walk(src.tree):
                if isinstance(node, ast.Call):
                    name = last_name(node.func)
                    self.calls_by_name[name].append((node, src))
                    if name == "add_argument":
                        dest = argparse_dest(node)
                        if dest and "default" in {k.arg for k in node.keywords}:
                            self.argparse[dest].append((next(k.value for k in node.keywords if k.arg == "default"), src))
                    elif re.fullmatch(r"DEFINE_\w+", name) and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant):
                        self.argparse[str(node.args[0].value)].append((node.args[1], src))
                    elif name == "dict":
                        for kw in node.keywords:
                            if kw.arg:
                                self.dict_keys[kw.arg].append((kw.value, src))
                elif isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            self.dict_keys[key.value].append((value, src))
                elif isinstance(node, ast.ClassDef):
                    for stmt in node.body:
                        for target, value in assignment_pairs(stmt):
                            if isinstance(target, ast.Name) and value is not None:
                                self.class_attrs[target.id].append((value, src))
                for target, value in assignment_pairs(node) if isinstance(node, (ast.Assign, ast.AnnAssign)) else ():
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                        self.self_attrs[target.attr].append((value, src, node))

    def configs(self) -> dict[str, list[tuple[ast.expr, str]]]:
        if self._configs is None:
            self._configs = defaultdict(list)
            for path in itertools.chain.from_iterable(self.root.rglob(f"*{s}") for s in CONFIG_SUFFIXES):
                if path.stat().st_size > 2_000_000 or "node_modules" in path.parts:
                    continue
                try:
                    text = path.read_text(errors="replace")
                    data = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) if yaml else None)
                except Exception:
                    continue
                for key, value in flatten(data):
                    node = literal_node(value)
                    if node is not None:
                        self._configs[key].append((node, str(path.relative_to(self.root))))
        return self._configs


def flatten(data: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)) and not literal_node(value):
                yield from flatten(value)
            else:
                yield str(key), value
    elif isinstance(data, list):
        for item in data:
            yield from flatten(item)


def assignment_pairs(stmt: ast.AST) -> list[tuple[ast.expr, ast.expr | None]]:
    if isinstance(stmt, ast.Assign):
        return [(t, stmt.value) for t in stmt.targets]
    if isinstance(stmt, ast.AnnAssign):
        return [(stmt.target, stmt.value)]
    return []


def argparse_dest(node: ast.Call) -> str | None:
    for kw in node.keywords:
        if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    options = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    long = [o for o in options if o.startswith("--")]
    name = (long or options or [None])[0]
    return name.lstrip("-").replace("-", "_") if name else None


class Tracer:
    def __init__(self, repo: Repo):
        self.repo = repo
        self.evidence: list[tuple[str, str, str, str]] = []  # (expression, method, location, value)
        self.active: set[tuple[str, str, int]] = set()  # lookups in progress, to stop cycles

    def note(self, expr: ast.AST, method: str, src: SourceFile, line: int, values: list[ast.expr]) -> None:
        self.evidence.append((ast.unparse(expr), method, f"{src.rel}:{line}", " | ".join(ast.unparse(v) for v in values)))

    # -- core ---------------------------------------------------------------
    def resolve(self, node: ast.expr, src: SourceFile, line: int, depth: int = 0) -> list[ast.expr]:
        """Alternatives for `node` with its variables replaced. Unresolvable
        parts are left as they are (so the analysis still reports them)."""
        if depth > MAX_DEPTH or isinstance(node, (ast.Constant, ast.Lambda)):
            return [node]
        key = (src.rel, ast.unparse(node), line)
        if key in self.active:
            return [node]
        self.active.add(key)
        try:
            return self._resolve(node, src, line, depth)
        finally:
            self.active.discard(key)

    def _resolve(self, node: ast.expr, src: SourceFile, line: int, depth: int) -> list[ast.expr]:
        if isinstance(node, ast.Name):
            found = self.name(node, src, line, depth)
            return found or [node]
        if isinstance(node, ast.Attribute):
            found = self.attribute(node, src, line, depth)
            return found or [node]
        if isinstance(node, ast.Subscript):
            found = self.subscript(node, src, line, depth)
            return found or [node]
        if isinstance(node, ast.Call):
            name = last_name(node.func)
            if name in {"float", "int"} and len(node.args) == 1 and not node.keywords:
                return self.resolve(node.args[0], src, line, depth + 1)
            return self.rebuild(node, src, line, depth, args=True)
        if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.BinOp, ast.UnaryOp, ast.IfExp)):
            return self.rebuild(node, src, line, depth)
        return [node]

    def rebuild(self, node: ast.expr, src: SourceFile, line: int, depth: int, args: bool = False) -> list[ast.expr]:
        """Substitute inside a compound node, combining children's alternatives."""
        fields: list[tuple[str, int | None, ast.expr]] = []
        if isinstance(node, ast.Call):
            fields += [("args", i, a) for i, a in enumerate(node.args) if not isinstance(a, ast.Starred)]
            fields += [("keywords", i, k.value) for i, k in enumerate(node.keywords) if k.arg]
        elif isinstance(node, ast.Dict):
            fields += [("values", i, v) for i, v in enumerate(node.values)]
        elif isinstance(node, (ast.List, ast.Tuple)):
            fields += [("elts", i, e) for i, e in enumerate(node.elts)]
        elif isinstance(node, ast.BinOp):
            fields += [("left", None, node.left), ("right", None, node.right)]
        elif isinstance(node, ast.UnaryOp):
            fields += [("operand", None, node.operand)]
        elif isinstance(node, ast.IfExp):
            fields += [("body", None, node.body), ("orelse", None, node.orelse)]
        options = [self.resolve(child, src, line, depth + 1) for _f, _i, child in fields]
        out = []
        for combo in itertools.islice(itertools.product(*options), MAX_ALTERNATIVES):
            copy = ast.parse(ast.unparse(node), mode="eval").body
            for (field, index, _child), value in zip(fields, combo):
                if field == "keywords":
                    copy.keywords[index].value = value
                elif index is None:
                    setattr(copy, field, value)
                else:
                    getattr(copy, field)[index] = value
            out.append(copy)
        return dedupe(out) or [node]

    # -- names --------------------------------------------------------------
    def name(self, node: ast.Name, src: SourceFile, line: int, depth: int) -> list[ast.expr]:
        if src.tree is None:
            return []
        func = src.function_at(line)
        scope = func
        while scope is not None:
            found = self.in_function(node, scope, src, line, depth)
            if found is not None:
                return found
            scope = src.enclosing(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        return self.in_module(node.id, src, line, depth, node)

    def in_function(self, node: ast.Name, func: ast.AST, src: SourceFile, line: int, depth: int) -> list[ast.expr] | None:
        best = None
        for sub in ast.walk(func):
            if sub is not func and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if getattr(sub, "lineno", 10**9) >= line:
                continue
            for target, value in assignment_pairs(sub):
                if isinstance(target, ast.Name) and target.id == node.id and value is not None:
                    if best is None or sub.lineno > best[0]:
                        best = (sub.lineno, "local", value)
            if isinstance(sub, ast.For) and isinstance(sub.target, ast.Name) and sub.target.id == node.id:
                if isinstance(sub.iter, (ast.List, ast.Tuple)) and (best is None or sub.lineno > best[0]):
                    best = (sub.lineno, "loop", sub.iter)
        if best is not None:
            lineno, method, value = best
            if method == "loop":
                values = dedupe(v for e in value.elts for v in self.resolve(e, src, lineno, depth + 1))
            else:
                values = self.resolve(value, src, lineno, depth + 1)
            self.note(node, method, src, lineno, values)
            return values
        params = parameter_defaults(func)
        if node.id in params:
            return self.parameter(node, func, params[node.id], src, depth)
        return None

    def parameter(self, node: ast.Name, func: ast.AST, default: ast.expr | None, src: SourceFile, depth: int) -> list[ast.expr]:
        values: list[ast.expr] = []
        if default is not None and not (isinstance(default, ast.Constant) and default.value is None):
            values += self.resolve(default, src, func.lineno, depth + 1)
            self.note(node, "parameter default", src, func.lineno, values)
        index = positional_index(func, node.id)
        names = [func.name]
        if func.name == "__init__":
            cls = src.enclosing(func, ast.ClassDef)
            names = [cls.name] if cls is not None else []
        for name in names:
            for call, caller_src in self.repo.calls_by_name.get(name, []):
                passed = next((k.value for k in call.keywords if k.arg == node.id), None)
                if passed is None and index is not None and index < len(call.args) and not any(isinstance(a, ast.Starred) for a in call.args[: index + 1]):
                    passed = call.args[index]
                if passed is not None:
                    resolved = self.resolve(passed, caller_src, call.lineno, depth + 1)
                    self.note(node, "caller", caller_src, call.lineno, resolved)
                    values += resolved
        return dedupe(values)

    def in_module(self, name: str, src: SourceFile, line: int, depth: int, node: ast.expr) -> list[ast.expr]:
        if depth > MAX_DEPTH:
            return []
        assigns = self.repo.module_assigns.get(src.rel, {}).get(name, [])
        if assigns:
            before = [a for a in assigns if a[0] < line] or assigns
            lineno, value = max(before, key=lambda a: a[0])
            values = self.resolve(value, src, lineno, depth + 1)
            self.note(node, "module", src, lineno, values)
            return values
        target = self.imported_module_file(name, src)
        if target is not None:
            module_src, original = target
            values = self.in_module(original, module_src, 10**9, depth + 1, node)
            if values:
                return values
        return []

    def imported_module_file(self, name: str, src: SourceFile) -> tuple[SourceFile, str] | None:
        for stmt in ast.walk(src.tree):
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if (alias.asname or alias.name) == name:
                        module = self.find_module(stmt.module or "", stmt.level, src)
                        if module is not None:
                            return module, alias.name
        return None

    def find_module(self, dotted: str, level: int, src: SourceFile) -> SourceFile | None:
        parts = dotted.split(".") if dotted else []
        bases = [Path(src.rel).parent]
        for _ in range(max(level - 1, 0)):
            bases = [b.parent for b in bases]
        if not level:
            bases = [Path("."), Path(src.rel).parent] + [Path(p) for p in {str(Path(r).parts[0]) for r in self.repo.files if len(Path(r).parts) > 1}]
        for base in bases:
            for candidate in (base.joinpath(*parts).with_suffix(".py") if parts else None, base.joinpath(*parts, "__init__.py")):
                if candidate is not None and str(candidate) in self.repo.files:
                    return self.repo.files[str(candidate)]
        if parts:  # fall back to any file with the module's name
            matches = [f for r, f in self.repo.files.items() if r.endswith("/".join(parts) + ".py")]
            if len(matches) == 1:
                return matches[0]
        return None

    # -- attributes / subscripts --------------------------------------------
    def attribute(self, node: ast.Attribute, src: SourceFile, line: int, depth: int) -> list[ast.expr]:
        base = ast.unparse(node.value)
        attr = node.attr
        if base == "self":
            cls = src.enclosing(src.function_at(line) or node, ast.ClassDef)
            candidates = [(v, s, a) for v, s, a in self.repo.self_attrs.get(attr, [])
                          if cls is None or (s is src and s.enclosing(a, ast.ClassDef) is cls)]
            candidates = candidates or self.repo.self_attrs.get(attr, [])
            values = []
            for value, value_src, stmt in candidates:
                resolved = self.resolve(value, value_src, stmt.lineno, depth + 1)
                self.note(node, "self", value_src, stmt.lineno, resolved)
                values += resolved
            if values:
                return dedupe(values)
            return self.class_values(attr, node)
        if isinstance(node.value, ast.Name) and src.tree is not None:
            module = self.imported_module(node.value.id, src)
            if module is not None:
                return self.in_module(attr, module, 10**9, depth + 1, node)
        if isinstance(node.value, ast.Call) and last_name(node.value.func) == "parse_args" or base.split(".")[-1] in ARGPARSE_OBJECTS or base.endswith(("args", "opt", "flags", "FLAGS")):
            values = self.argparse_values(attr, node, src)
            if values:
                return values
        values = self.class_values(attr, node) or self.config_values(attr, node)
        if not values:
            values = self.argparse_values(attr, node, src)
        return values

    def imported_module(self, name: str, src: SourceFile) -> SourceFile | None:
        for stmt in ast.walk(src.tree):
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if (alias.asname or alias.name) == name:
                        return self.find_module(alias.name, 0, src)
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if (alias.asname or alias.name) == name:
                        return self.find_module(".".join(filter(None, [stmt.module, alias.name])), stmt.level, src)
        return None

    def argparse_values(self, attr: str, node: ast.expr, src: SourceFile) -> list[ast.expr]:
        entries = self.repo.argparse.get(attr, [])
        same_file = [e for e in entries if e[1] is src]
        values = []
        for value, value_src in same_file or entries:
            resolved = self.resolve(value, value_src, value.lineno, MAX_DEPTH - 2)
            self.note(node, "argparse", value_src, value.lineno, resolved)
            values += resolved
        return dedupe(values)

    def class_values(self, attr: str, node: ast.expr) -> list[ast.expr]:
        values = []
        for value, value_src in self.repo.class_attrs.get(attr, []):
            resolved = self.resolve(value, value_src, value.lineno, MAX_DEPTH - 2)
            self.note(node, "class", value_src, value.lineno, resolved)
            values += resolved
        return dedupe(values)

    def config_values(self, key: str, node: ast.expr) -> list[ast.expr]:
        values = []
        for value, path in self.repo.configs().get(key, []):
            self.evidence.append((ast.unparse(node), "config", path, ast.unparse(value)))
            values.append(value)
        return dedupe(values)

    def subscript(self, node: ast.Subscript, src: SourceFile, line: int, depth: int) -> list[ast.expr]:
        key = node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None
        if key is None:
            return []
        for base in self.resolve(node.value, src, line, depth + 1):
            if isinstance(base, ast.Dict):
                for k, v in zip(base.keys, base.values):
                    if isinstance(k, ast.Constant) and k.value == key:
                        values = self.resolve(v, src, line, depth + 1)
                        self.note(node, "dict", src, line, values)
                        return values
        same_file = [(v, s) for v, s in self.repo.dict_keys.get(key, []) if s is src]
        values = []
        for value, value_src in same_file:
            resolved = self.resolve(value, value_src, value.lineno, depth + 1)
            self.note(node, "dict", value_src, value.lineno, resolved)
            values += resolved
        if not values:
            values = self.config_values(key, node)
        if not values:
            for value, value_src in self.repo.dict_keys.get(key, []):
                resolved = self.resolve(value, value_src, value.lineno, depth + 1)
                self.note(node, "dict", value_src, value.lineno, resolved)
                values += resolved
        return dedupe(values)


def parameter_defaults(func: ast.AST) -> dict[str, ast.expr | None]:
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    args = func.args
    positional = args.posonlyargs + args.args
    defaults: dict[str, ast.expr | None] = {a.arg: None for a in positional + args.kwonlyargs}
    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        defaults[arg.arg] = default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        defaults[arg.arg] = default
    return defaults


def positional_index(func: ast.AST, name: str) -> int | None:
    names = [a.arg for a in func.args.posonlyargs + func.args.args]
    if name not in names:
        return None
    index = names.index(name)
    if names and names[0] in {"self", "cls"}:
        index -= 1
    return index if index >= 0 else None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def find_call(src: SourceFile, line: int, function_name: str) -> ast.Call | None:
    if src.tree is None:
        return None
    wanted = function_name.rsplit(".", 1)[-1]
    candidates = [n for n in ast.walk(src.tree) if isinstance(n, ast.Call) and n.lineno == line]
    for node in candidates:
        if last_name(node.func) == wanted:
            return node
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/github-vi-software-stan.sqlite"))
    parser.add_argument("--classification", type=Path, required=True,
                        help="strict_call_site_classification.csv whose unresolved call sites should be traced.")
    parser.add_argument("--snapshots", type=Path, default=Path("data/repo_snapshots"))
    args = parser.parse_args()
    out_dir = args.db.parent / args.db.stem

    csv.field_size_limit(10**9)
    unresolved = [r for r in csv.DictReader(args.classification.open()) if r["strict_classification"] == "unresolved"]
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    calls = {row["id"]: dict(row) for row in conn.execute("SELECT * FROM github_advi_calls")}
    by_location = {(c["repo_full_name"], c["path"], c["line_number"]): cid for cid, c in calls.items()}
    settings_of = dict(conn.execute("SELECT call_id, settings_call_id FROM github_advi_settings_companions"))
    members_of: dict[int, list[int]] = defaultdict(list)
    for cid, sid in settings_of.items():
        if sid is not None:
            members_of[sid].append(cid)

    targets: dict[int, str] = {}  # call id -> snapshot folder
    for row in unresolved:
        sha = re.search(r"/blob/([0-9a-f]{40})/", row["github_permalink"]).group(1)
        repo, _sha, rest = row["call_site_id"].split(":", 2)
        path, line = rest.rsplit(":", 1)
        cid = by_location.get((repo, path, int(line)))
        if cid is None:
            continue
        sid = settings_of.get(cid) or cid
        for member in {cid, sid, *members_of.get(sid, [])}:
            targets[member] = f"{repo.replace('/', '__')}@{sha}"

    by_repo: dict[str, list[int]] = defaultdict(list)
    for cid, folder in targets.items():
        by_repo[folder].append(cid)

    traced: dict[str, Any] = {}
    evidence_rows = []
    missing = 0
    for done, (folder, cids) in enumerate(sorted(by_repo.items()), 1):
        root = args.snapshots / folder
        if not (root / ".done").exists():
            missing += len(cids)
            continue
        repo = Repo(root)
        tracer = Tracer(repo)
        for cid in cids:
            call = calls[cid]
            src = repo.files.get(call["path"])
            node = find_call(src, call["line_number"], call["function_name"]) if src else None
            if node is None:
                continue
            start = len(tracer.evidence)
            positional = []
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    positional.append(ast.unparse(arg))
                    continue
                options = tracer.resolve(arg, src, node.lineno)
                positional.append(ast.unparse(alternatives_node(options)))
            keywords = {}
            for kw in node.keywords:
                if kw.arg is None:
                    keywords["**"] = ast.unparse(kw.value)
                    continue
                options = tracer.resolve(kw.value, src, node.lineno)
                keywords[kw.arg] = ast.unparse(alternatives_node(options))
            func_text = ast.unparse(node.func)
            call_text = f"{func_text}({', '.join(positional + [f'{k}={v}' for k, v in keywords.items() if k != '**'] + ([f'**{keywords[chr(42) * 2]}'] if '**' in keywords else []))})"
            traced[str(cid)] = {"positional": positional, "keywords": keywords, "call": call_text}
            for expression, method, location, value in tracer.evidence[start:]:
                evidence_rows.append({
                    "call_id": cid, "repo": call["repo_full_name"], "call_path": call["path"],
                    "call_line": call["line_number"], "expression": expression, "method": method,
                    "evidence_location": location, "traced_value": value,
                })
        if done % 50 == 0:
            print(f"traced {done}/{len(by_repo)} repos", flush=True)

    (out_dir / "traced_values.json").write_text(json.dumps(traced, indent=1))
    with (out_dir / "traced_value_evidence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence_rows[0].keys()) if evidence_rows else ["call_id"])
        writer.writeheader()
        writer.writerows(evidence_rows)
    print(f"Traced {len(traced)} calls in {len(by_repo)} repos ({missing} calls had no snapshot); "
          f"{len(evidence_rows)} evidence rows -> {out_dir}")


if __name__ == "__main__":
    main()
