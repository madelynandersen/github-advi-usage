"""Per-component default usage for NumPyro, PyMC, TFP and Stan call sites.

Each component of a call site's VI setup is scored on its own as default,
non_default, unresolved (value can't be read statically) or n/a:

  optimizer     NumPyro: Adam (numpyro.optim.Adam or a plain optax.adam) with any learning rate,
                         b1/b2/eps at their defaults, not inside an optax chain / clipping wrapper
                TFP:     Keras Adam with Keras's default learning rate (not given, or 0.001)
                PyMC:    adagrad_window with its default settings (or no optimizer given)
                Stan:    n/a (no optimizer choice)
  mc_samples    NumPyro: ELBO num_particles = 1;  TFP: sample_size = 1
                PyMC:    obj_n_mc = 1;            Stan: grad_samples = 1
  transforms    TFP:     the surrogate posterior applies no bijectors; n/a elsewhere
  iterations    PyMC: n = 10,000;  Stan: iter = 10,000;  n/a for NumPyro / TFP (any count is fine)
  convergence   PyMC: no CheckParametersConvergence callback;  Stan: tol_rel_obj = 0.01
  all           every applicable component is default

Uses the confirmed Pyro/NumPyro labels from verify_package_labels (Pyro and
unconfirmed call sites are excluded) and, for TFP transforms, source files
cached by verify_package_labels / fetched alongside it:

    python -m github_advi_usage.component_defaults --db data/github-vi-software-stan.sqlite
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from . import strict_default_analysis as sda
from .build_vi_github_analysis import call_site_id

PACKAGES = {"numpyro": "NumPyro", "tensorflow_probability": "TensorFlow Probability", "pymc": "PyMC", "stan": "Stan"}
COMPONENTS = ["optimizer", "mc_samples", "transforms", "iterations", "convergence"]
TRANSFORM_RE = re.compile(
    r"\btfb\.\w+|\bbijectors?\.\w+|TransformedDistribution|TransformedVariable|\bbijectors?\s*=|"
    r"constraining_bijectors?|default_event_space_bijector|\bSoftplus\(|\bExp\(\)|\bSigmoid\("
)


def score(check) -> str:
    v = sda.Verdict()
    check(v)
    return v.classification


KNOWN_NON_ADAM = {
    "ClippedAdam", "Adagrad", "SGD", "RMSProp", "RMSPropMomentum", "Momentum", "Minimize", "SM3", "Adadelta",
    "adabelief", "adamw", "adamax", "adagrad", "adadelta", "sgd", "rmsprop", "lamb", "lion", "lars", "yogi", "radam",
    "nadam", "noisy_sgd", "fromage", "chain", "clip", "clip_by_global_norm", "multi_transform", "inject_hyperparams",
}


def numpyro_optimizer(v: sda.Verdict, node: ast.expr | None) -> None:
    """Adam (numpyro.optim.Adam, or a plain optax.adam / adam) with any learning
    rate and default b1/b2/eps is default; a known non-Adam optimizer or any
    optax chain/clipping wrapper is non-default; an optimizer built by the
    user's own helper function (e.g. make_optimizer()) can't be read."""
    outcomes = []
    for option in sda.alternatives(node):  # several possible optimizers: default if any is
        sub = sda.Verdict()
        if isinstance(option, ast.Call) and any(w in ast.unparse(option.func) for w in ("optax_to_numpyro", "optim_to_numpyro")):
            option = option.args[0] if option.args else None
        if option is None:
            sub.unknown("optimizer not found")
        elif not isinstance(option, ast.Call):
            sub.unknown(f"optimizer can't be read: {ast.unparse(option)[:60]}")
        else:
            func = ast.unparse(option.func)
            name = sda.last_name(option.func)
            if name in {"Adam", "adam"} and "torch" not in func:
                pos, kwargs, _ = sda.call_args(option)
                for index, (key, default) in enumerate(sda.NUMPYRO_ADAM_DEFAULTS, start=1):
                    value = pos[index] if len(pos) > index else kwargs.get(key)
                    if value is not None:
                        sub.check(f"Adam {key}", lambda value=value, default=default: sda.same_number(value, default))
            elif name in KNOWN_NON_ADAM or func.startswith(("optax.", "torch.", "numpyro.optim.", "optim.")):
                sub.bad(f"optimizer {func}")
            else:
                sub.unknown(f"optimizer built by {func}()")
        outcomes.append(sub)
    kinds = {o.classification for o in outcomes}
    if "default" in kinds:
        return
    for o in outcomes:
        for reason in (o.non_default if kinds == {"non_default"} else o.unresolved):
            (v.bad if kinds == {"non_default"} else v.unknown)(reason)


def elbo_particles(v: sda.Verdict, node: ast.expr | None) -> None:
    if node is None:
        v.unknown("loss not found")
        return
    for option in sda.alternatives(node):
        if isinstance(option, ast.Call):
            pos, kwargs, _ = sda.call_args(option)
            particles = pos[0] if pos else kwargs.get("num_particles")
            if particles is None:
                return  # num_particles left at its default of 1
            try:
                if sda.same_number(particles, 1):
                    return
                v.bad(f"num_particles={ast.unparse(particles)}")
            except sda.Unresolved:
                v.unknown("num_particles can't be read")
        else:
            v.unknown("loss can't be read")
    if v.non_default and v.unresolved:
        v.non_default.clear()


def kw_or_pos(settings: dict | None, name: str, index: int | None = None):
    if settings is None:
        return None, False, True
    pos, kwargs, spread = sda.stored_args(settings)
    value = kwargs.get(name)
    if value is None and index is not None and len(pos) > index:
        value = pos[index]
    return value, spread, False


def numeric_component(value, spread: bool, missing: bool, default: float, label: str) -> str:
    if missing:
        return "unresolved"
    if value is None:
        return "unresolved" if spread else "default"
    return score(lambda v: v.check(label, lambda: sda.same_number(value, default)))


def tfp_transforms(settings: dict | None, sources: dict[tuple[str, str], str]) -> tuple[str, str]:
    """(classification, how decided) for whether the surrogate applies bijectors."""
    if settings is None:
        return "unresolved", "fit call not found"
    source = sources.get((settings["repo_full_name"], settings["path"]))
    if source is None:
        return "unresolved", "source file unavailable"
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return ("non_default" if TRANSFORM_RE.search(source) else "default"), "file-level (file doesn't parse)"
    call = next((n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.lineno == settings["line_number"]
                 and sda.last_name(n.func) == "fit_surrogate_posterior"), None)
    if call is None:
        return ("non_default" if TRANSFORM_RE.search(source) else "default"), "file-level (call not located)"
    surrogate = next((k.value for k in call.keywords if k.arg == "surrogate_posterior"), None)
    if surrogate is None and len(call.args) > 1:
        surrogate = call.args[1]
    if surrogate is None:
        return "unresolved", "surrogate_posterior not passed by name or position"
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    texts, seen, frontier = [], set(), [surrogate]
    traced_any = False
    while frontier:
        node = frontier.pop()
        for sub in ast.walk(node):
            name = sub.id if isinstance(sub, ast.Name) else sub.attr if isinstance(sub, ast.Attribute) else None
            if name is None or name in seen:
                continue
            seen.add(name)
            for stmt in ast.walk(tree):  # where this name is built
                for target, value in sda_assignments(stmt):
                    if value is not None and ast.unparse(target).split(".")[-1] == name:
                        texts.append(ast.unparse(value))
                        frontier.append(value)
                        traced_any = True
            if name in functions:  # a helper that builds the surrogate
                texts.append(ast.unparse(functions[name]))
                traced_any = True
        texts.append(ast.unparse(node))
    code = "\n".join(texts)
    if TRANSFORM_RE.search(code):
        return "non_default", "bijector in surrogate construction"
    if not traced_any and isinstance(surrogate, ast.Name):
        return ("non_default" if TRANSFORM_RE.search(source) else "default"), "file-level (surrogate built elsewhere)"
    return "default", "no bijector in surrogate construction"


def stan_interface(call: dict | None, settings: dict | None) -> str:
    """PyStan 2 runs VI with `.vb(...)`; CmdStanPy with `.variational(...)`
    (PyStan 3 has no VI). Constructor rows take their VI call's interface,
    falling back to their own class (StanModel vs CmdStanModel)."""
    for row in (settings, call):
        if row is None:
            continue
        name = row["function_name"].rsplit(".", 1)[-1]
        if name == "vb":
            return "pystan"
        if name.lower() == "variational" or name == "CmdStanModel":
            return "cmdstanpy"
        if name == "StanModel":
            return "pystan"
    return "unknown"


def sda_assignments(stmt):
    if isinstance(stmt, ast.Assign):
        return [(t, stmt.value) for t in stmt.targets]
    if isinstance(stmt, ast.AnnAssign):
        return [(stmt.target, stmt.value)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/github-vi-software-stan.sqlite"))
    parser.add_argument("--cache", type=Path, default=Path("data/package_check_files"))
    parser.add_argument("--exclude-pystan", action="store_true",
                        help="Keep only CmdStanPy Stan call sites (drop PyStan's .vb() and StanModel).")
    args = parser.parse_args()
    base = args.db.parent / args.db.stem
    out = base / "component_defaults"
    out.mkdir(parents=True, exist_ok=True)
    sda.RULES = sda.STRICT_RULES

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    relevant, calls, _groups = sda.load(conn)
    labels = sda.load_package_labels(base / "package_labels.csv")
    sources: dict[tuple[str, str], str] = {}
    for folder in args.cache.iterdir() if args.cache.exists() else []:
        if "@" not in folder.name or not folder.is_dir():
            continue
        repo = folder.name.split("@")[0].replace("__", "/", 1)
        for path in folder.rglob("*.py"):
            sources[(repo, str(path.relative_to(folder)))] = path.read_text(errors="replace")

    rows = []
    for rel in relevant:
        key = (rel["repo_full_name"], rel["path"], int(rel["line_number"]))
        package = labels.get(key, rel["package"]) if rel["package"] in {"pyro", "numpyro", "pyro_or_numpyro"} else rel["package"]
        if package not in PACKAGES:
            continue
        sid = int(rel["settings_source_call_id"]) if rel.get("settings_source_call_id") else None
        settings = calls.get(sid)
        if package == "stan" and args.exclude_pystan:
            own = next((c for c in calls.values() if c["repo_full_name"] == rel["repo_full_name"]
                        and c["path"] == rel["path"] and c["line_number"] == int(rel["line_number"])
                        and c["call_kind"] == rel["call_kind"]), None) if settings is None else None
            if stan_interface(own, settings) != "cmdstanpy":
                continue
        optimizer = sda.parse_expr(rel.get("optimizer_expression"))
        result = {c: "n/a" for c in COMPONENTS}
        how = ""
        if package == "numpyro":
            result["optimizer"] = score(lambda v: numpyro_optimizer(v, optimizer))
            result["mc_samples"] = score(lambda v: elbo_particles(v, sda.parse_expr(rel.get("loss_expression"))))
        elif package == "tensorflow_probability":
            result["optimizer"] = score(lambda v: sda.check_tfp_optimizer(v, optimizer))
            result["mc_samples"] = numeric_component(*kw_or_pos(settings, "sample_size", 6), 1, "sample_size")
            result["transforms"], how = tfp_transforms(settings, sources)
        elif package == "pymc":
            result["optimizer"] = "default" if optimizer is None else score(lambda v: sda.check_adagrad_window(v, optimizer))
            mc = sda.parse_expr(rel.get("sample_or_mc_expression")) if rel.get("sample_or_mc_expression") else None
            value, spread, missing = kw_or_pos(settings, "obj_n_mc")
            result["mc_samples"] = numeric_component(mc if mc is not None else value, spread, missing, 1, "obj_n_mc")
            result["iterations"] = numeric_component(*kw_or_pos(settings, "n", 0), 10000, "n")
            callbacks, spread, missing = kw_or_pos(settings, "callbacks")
            result["convergence"] = ("unresolved" if missing or (callbacks is None and spread) else
                                     "non_default" if callbacks is not None and "Convergence" in ast.unparse(callbacks) else "default")
        elif package == "stan":
            result["mc_samples"] = numeric_component(*kw_or_pos(settings, "grad_samples"), 1, "grad_samples")
            result["iterations"] = numeric_component(*kw_or_pos(settings, "iter"), 10000, "iter")
            result["convergence"] = numeric_component(*kw_or_pos(settings, "tol_rel_obj"), 0.01, "tol_rel_obj")
        applicable = [result[c] for c in COMPONENTS if result[c] != "n/a"]
        result["all"] = ("non_default" if "non_default" in applicable else
                         "unresolved" if "unresolved" in applicable else "default")
        rows.append({
            "call_site_id": call_site_id(rel["repo_full_name"], rel["html_url"], rel["path"], rel["line_number"]),
            "repo_full_name": rel["repo_full_name"], "package": PACKAGES[package], "call_kind": rel["call_kind"],
            "github_permalink": rel["html_url"], **result, "transforms_how_decided": how,
            "optimizer_expression": rel.get("optimizer_expression") or "", "loss_expression": rel.get("loss_expression") or "",
        })

    with (out / "component_call_sites.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for unit in ("call_site", "repository"):
        for package in [*PACKAGES.values(), "All packages"]:
            subset = [r for r in rows if package == "All packages" or r["package"] == package]
            for component in [*COMPONENTS, "all"]:
                if unit == "call_site":
                    labels_ = [r[component] for r in subset]
                else:
                    by_repo = defaultdict(list)
                    for r in subset:
                        by_repo[r["repo_full_name"]].append(r[component])
                    labels_ = [("non_default" if "non_default" in v else "unresolved" if "unresolved" in v else
                                "default" if "default" in v else "n/a") for v in by_repo.values()]
                counts = Counter(labels_)
                d, n, u = counts["default"], counts["non_default"], counts["unresolved"]
                if d + n + u == 0:
                    continue
                summary.append({"unit": unit, "package": package, "component": component, "default": d,
                                "non_default": n, "unresolved": u,
                                "default_pct_of_resolved": f"{100 * d / (d + n):.1f}" if d + n else ""})
    with (out / "component_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(",".join(str(row[k]) for k in row))
    print(f"Wrote {out}")
    print("TFP transforms decided by:", Counter(r["transforms_how_decided"] for r in rows if r["package"] == "TensorFlow Probability"))


if __name__ == "__main__":
    main()
