"""What the non-default settings are, for each cell of the component table.

For every call site that component_defaults.py scored non-default on a
component, records the value it used (e.g. `num_particles=10`, Adam with
`learning_rate=0.1`), then writes:

  component_defaults/nondefault_breakdown.csv
  component_defaults/figures/nondefault_breakdown.{png,svg,pdf}   grid: package x component

    python -m github_advi_usage.plot_nondefault_breakdown --db data/github-vi-software-stan.sqlite
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import component_defaults as cd  # noqa: E402
from . import strict_default_analysis as sda  # noqa: E402
from .build_vi_github_analysis import call_site_id  # noqa: E402
from .plot_component_defaults import (  # noqa: E402
    GRID, NON_DEFAULT_COLOR, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
)

PACKAGES = ["NumPyro", "TensorFlow Probability", "PyMC", "Stan"]
COMPONENTS = [
    ("optimizer", "Optimizer"),
    ("mc_samples", "MC samples"),
    ("transforms", "Transformations"),
    ("iterations", "Total/Max iterations"),
    ("convergence", "Convergence setting"),
]
TOP = 6


def short(node) -> str:
    return re.sub(r"\s+", " ", ast.unparse(node))[:48] if node is not None else "?"


def num(node) -> str:
    try:
        return f"{sda.number(node):g}"
    except Exception:
        return short(node)


def numpyro_optimizer(node) -> str | None:
    for o in sda.alternatives(node):
        if isinstance(o, ast.Call):
            func = ast.unparse(o.func)
            if "to_numpyro" in func and o.args and isinstance(o.args[0], ast.Call):
                o = o.args[0]
                func = ast.unparse(o.func)
            name = sda.last_name(o.func)
            if name in ("Adam", "adam") and "torch" not in func:
                pos, kw, _ = sda.call_args(o)
                changed = [f"{k}={num(v)}" for i, (k, _d) in enumerate(sda.NUMPYRO_ADAM_DEFAULTS, 1)
                           for v in [pos[i] if len(pos) > i else kw.get(k)] if v is not None]
                return "Adam, changed " + ", ".join(changed)
            if "chain" in func:
                return "optax chain (e.g. clipping + Adam)"
            if name == "ClippedAdam":
                return "ClippedAdam"
            return ("optax." if "optax" in func or name.islower() else "") + name
    return None


def tfp_optimizer(node) -> str | None:
    for o in sda.alternatives(node):
        if isinstance(o, ast.Call) and sda.last_name(o.func) in ("Adam", "AdamOptimizer"):
            pos, kw, _ = sda.call_args(o)
            lr = pos[0] if pos else kw.get("learning_rate")
            if lr is not None and sda.is_schedule(lr):
                return "Adam, learning-rate schedule"
            if lr is not None:
                return f"Adam, learning_rate={num(lr)}"
            return "Adam, changed " + ", ".join(f"{k}={num(v)}" for k, v in kw.items())
    return None  # e.g. an optimizer built from a config dict: not readable


def pymc_optimizer(node) -> str | None:
    if isinstance(node, ast.Call):
        name = sda.last_name(node.func)
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if name == "adagrad_window":
            return "adagrad_window, changed " + ", ".join(f"{k}={num(v)}" for k, v in kw.items())
        return f"pm.{name}"
    return None


def transform_kind(src: str) -> str:
    signals = {m.group(0).split("(")[0].strip() for m in cd.TRANSFORM_RE.finditer(src)}
    if "TransformedVariable" in signals:
        return "TransformedVariable"
    if any("bijector" in s for s in signals):
        return "bijector(s) on the surrogate"
    if signals:
        return "explicit tfb bijectors"
    return "?"


def collect(db: Path, cache: Path) -> list[dict]:
    sda.RULES = sda.STRICT_RULES
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    relevant, calls, _ = sda.load(conn)
    base = db.parent / db.stem
    scored = {r["call_site_id"]: r for r in csv.DictReader((base / "component_defaults" / "component_call_sites.csv").open())}
    sources: dict[tuple[str, str], Path] = {}
    for folder in cache.iterdir() if cache.exists() else []:
        if "@" in folder.name and folder.is_dir():
            repo = folder.name.split("@")[0].replace("__", "/", 1)
            for path in folder.rglob("*.py"):
                sources[(repo, str(path.relative_to(folder)))] = path
    out = []
    for rel in relevant:
        row = scored.get(call_site_id(rel["repo_full_name"], rel["html_url"], rel["path"], rel["line_number"]))
        if row is None:
            continue
        package = row["package"]
        sid = int(rel["settings_source_call_id"]) if rel.get("settings_source_call_id") else None
        settings = calls.get(sid)
        for component, _label in COMPONENTS:
            if row[component] != "non_default":
                continue
            value = None
            if component == "optimizer":
                node = sda.parse_expr(row["optimizer_expression"])
                value = {"NumPyro": numpyro_optimizer, "TensorFlow Probability": tfp_optimizer,
                         "PyMC": pymc_optimizer}[package](node)
            elif component == "mc_samples":
                if package == "NumPyro":
                    for o in sda.alternatives(sda.parse_expr(row["loss_expression"])):
                        if isinstance(o, ast.Call):
                            pos, kw, _ = sda.call_args(o)
                            value = f"num_particles={num(pos[0] if pos else kw.get('num_particles'))}"
                else:
                    name = {"TensorFlow Probability": "sample_size", "PyMC": "obj_n_mc", "Stan": "grad_samples"}[package]
                    v, _, _ = cd.kw_or_pos(settings, name, 6 if package == "TensorFlow Probability" else None)
                    if package == "PyMC" and rel.get("sample_or_mc_expression"):
                        v = sda.parse_expr(rel["sample_or_mc_expression"])
                    value = f"{name}={num(v)}"
            elif component == "iterations":
                name = "n" if package == "PyMC" else "iter"
                v, _, _ = cd.kw_or_pos(settings, name, 0 if package == "PyMC" else None)
                try:
                    value = f"{name}={int(sda.number(v)):,}"
                except Exception:
                    value = f"{name}={num(v)}"
            elif component == "convergence":
                if package == "PyMC":
                    v, _, _ = cd.kw_or_pos(settings, "callbacks")
                    match = re.search(r"CheckParametersConvergence\(([^)]*)\)", ast.unparse(v))
                    args = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
                    # every PyMC convergence change is this callback; the label keeps only its settings
                    value = args if args and len(args) <= 34 else ("custom settings" if args else "default settings")
                else:
                    v, _, _ = cd.kw_or_pos(settings, "tol_rel_obj")
                    value = f"tol_rel_obj={num(v)}"
            elif component == "transforms" and settings is not None:
                path = sources.get((settings["repo_full_name"], settings["path"]))
                value = transform_kind(path.read_text(errors="replace")) if path else "?"
            if value:
                out.append({"package": package, "component": component, "value": value,
                            "repo_full_name": row["repo_full_name"], "call_site_id": row["call_site_id"]})
    return out


def plot(records: list[dict], out: Path) -> None:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    repos: dict[tuple[str, str], dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in records:
        key = (r["package"], r["component"])
        counts[key][r["value"]] += 1
        repos[key][r["value"]].add(r["repo_full_name"])

    fig, axes = plt.subplots(len(PACKAGES), len(COMPONENTS), figsize=(22, 15.5), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.12, "left": 0.085, "right": 0.99,
                                          "top": 0.885, "bottom": 0.02})
    fig.text(0.012, 0.965, "What the non-default settings are", fontsize=20, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(0.012, 0.938, "For each package and component: the values used by call sites that changed the default. "
             "Bar length = call sites; label = call sites · repositories. Six most common values shown, the rest pooled "
             "as \"other\".", fontsize=11.5, color=TEXT_SECONDARY)
    for j, (_component, label) in enumerate(COMPONENTS):
        axes[0, j].annotate(label, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 34), textcoords="offset points",
                            fontsize=13.5, fontweight="bold", color=TEXT_PRIMARY, ha="left", va="bottom")
    for i, package in enumerate(PACKAGES):
        axes[i, 0].annotate(package.replace("TensorFlow Probability", "TensorFlow\nProbability"), xy=(0, 0.5),
                            xycoords="axes fraction", xytext=(-14, 0), textcoords="offset points", fontsize=13.5,
                            fontweight="bold", color=TEXT_PRIMARY, ha="right", va="center")
        for j, (component, _label) in enumerate(COMPONENTS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            c = counts.get((package, component))
            if not c:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", fontsize=16, color=TEXT_SECONDARY)
                continue
            total = sum(c.values())
            total_repos = len(set().union(*repos[(package, component)].values()))
            ax.set_title(f"{total:,} call sites · {total_repos} repos changed it", loc="left", fontsize=10,
                         color=TEXT_SECONDARY, pad=6)
            items = c.most_common(TOP)
            rest = total - sum(n for _v, n in items)
            if rest:
                others = len(c) - TOP
                items.append((f"other ({others} value{'s' if others != 1 else ''})", rest))
            peak = max(n for _v, n in items)
            for y, (value, n) in enumerate(items):
                is_other = value.startswith("other (")
                ax.barh(y, n / peak, height=0.34, color="#d9d7d2" if is_other else NON_DEFAULT_COLOR, linewidth=0)
                r = "" if is_other else f" · {len(repos[(package, component)][value])} repos"
                ax.text(0, y - 0.24, value, fontsize=10, color=TEXT_PRIMARY, va="bottom", ha="left")
                ax.text(1.0, y - 0.24, f"{n:,}{r}", fontsize=9.5, color=TEXT_SECONDARY, va="bottom", ha="right")
            ax.set_ylim(TOP + 0.6, -0.75)
            ax.set_xlim(0, 1.0)
            ax.axvline(0, color=GRID, linewidth=0.8)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"nondefault_breakdown.{ext}", dpi=400, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/github-vi-software-stan.sqlite"))
    parser.add_argument("--cache", type=Path, default=Path("data/package_check_files"))
    args = parser.parse_args()
    base = args.db.parent / args.db.stem / "component_defaults"
    records = collect(args.db, args.cache)
    with (base / "nondefault_breakdown.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot(records, figures)
    print(f"Wrote {base / 'nondefault_breakdown.csv'} and {figures / 'nondefault_breakdown.png'}")


if __name__ == "__main__":
    main()
