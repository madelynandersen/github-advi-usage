"""TFP optimizer choices: how many repositories use Adam, how many of those
use step size 0.1, and what the non-default optimizer changes are.

Reads component_defaults/component_call_sites.csv (from component_defaults.py):

    python -m github_advi_usage.plot_tfp_adam --db data/github-vi-software-stan.sqlite
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import strict_default_analysis as sda  # noqa: E402
from .plot_component_defaults import (  # noqa: E402
    DEFAULT_COLOR, NON_DEFAULT_COLOR, NON_DEFAULT_HATCH_COLOR, TEXT_PRIMARY, TEXT_SECONDARY,
)

SURFACE = "white"
REST_COLOR = "#d9d7d2"
OTHER_LR_COLOR = "#F6B98A"  # lighter step of the non-default orange
BOTH_COLOR = "#F6B98A"
OTHER_CHANGE_COLOR = "#BBBBBB"

ONLY_01 = "Step size → 0.1 only"
ONLY_OTHER = "Step size → another value only"
BOTH = "0.1 in some fits, another value in others"
OTHER = "Schedule or other Adam settings"


def optimizer_info(expression: str):
    """(is_adam, step size: float | None | 'schedule' | 'unreadable', other settings changed)
    or None when the optimizer can't be read."""
    node = sda.parse_expr(expression)
    if node is None:
        return None
    if isinstance(node, (ast.Name, ast.Attribute)):
        return (True, None, False) if sda.last_name(node) in ("Adam", "AdamOptimizer") else None
    if not isinstance(node, ast.Call):
        return None
    name = sda.last_name(node.func)
    if name == "dict":
        # Config dict, e.g. dict(optimizer='adam', optimizer_kwargs=dict(learning_rate=0.005), ...)
        _, config, _ = sda.call_args(node)
        choice, inner = config.get("optimizer"), config.get("optimizer_kwargs")
        if not (isinstance(choice, ast.Constant) and isinstance(choice.value, str)):
            return None
        if choice.value.lower() != "adam":
            return False, None, True
        if inner is None:
            return True, None, False
        if not (isinstance(inner, ast.Call) and sda.last_name(inner.func) == "dict"):
            return None
        node = inner
    elif name not in ("Adam", "AdamOptimizer") or "optax" in ast.unparse(node.func):
        return False, None, True
    pos, kwargs, _ = sda.call_args(node)
    lr = pos[0] if pos else kwargs.get("learning_rate")
    if lr is None:
        step = None
    elif sda.is_schedule(lr):
        step = "schedule"
    else:
        try:
            step = sda.number(lr)
        except sda.Unresolved:
            step = "unreadable"
    other = any(k not in ("learning_rate", "name", "jit_compile") for k in kwargs)
    return True, step, other


def change_kind(info) -> str | None:
    is_adam, step, other = info
    if not is_adam or other or step == "schedule":
        return OTHER
    if step == "unreadable":
        return None
    if isinstance(step, float) and math.isclose(step, 0.1):
        return ONLY_01
    return ONLY_OTHER


def compute(rows: list[dict]) -> dict:
    by_repo = defaultdict(list)
    for row in rows:
        by_repo[row["repo_full_name"]].append((row, optimizer_info(row["optimizer_expression"])))
    readable_repos = {r for r, calls in by_repo.items() if any(i is not None for _, i in calls)}
    adam_repos = {r for r, calls in by_repo.items() if any(i and i[0] for _, i in calls)}
    is_01 = lambda i: i and i[0] and isinstance(i[1], float) and math.isclose(i[1], 0.1)  # noqa: E731
    repos_01 = {r for r in adam_repos if any(is_01(i) for _, i in by_repo[r])}
    calls_readable = [i for calls in by_repo.values() for _, i in calls if i is not None]
    calls_adam = [i for i in calls_readable if i[0]]

    change_calls = defaultdict(int)
    change_repos: dict[str, int] = defaultdict(int)
    for repo, calls in by_repo.items():
        kinds = set()
        for row, info in calls:
            if row["optimizer"] == "non_default" and info is not None:
                kind = change_kind(info)
                if kind:
                    change_calls[kind] += 1
                    kinds.add(kind)
        if not kinds:
            continue
        if kinds == {ONLY_01}:
            change_repos[ONLY_01] += 1
        elif kinds == {ONLY_OTHER}:
            change_repos[ONLY_OTHER] += 1
        elif kinds == {ONLY_01, ONLY_OTHER}:
            change_repos[BOTH] += 1
        else:
            change_repos[OTHER] += 1
    return {
        "repos_total": len(by_repo), "repos_readable": len(readable_repos), "repos_adam": len(adam_repos),
        "repos_01": len(repos_01), "calls_readable": len(calls_readable), "calls_adam": len(calls_adam),
        "calls_01": sum(1 for i in calls_adam if is_01(i)),
        "change_repos": dict(change_repos), "change_calls": dict(change_calls),
    }


def segment(ax, y, left, width, color, label, hatch=None, dark_text=False):
    ax.barh(y, width, left=left, height=0.5, color=color, hatch=hatch, linewidth=0,
            edgecolor=NON_DEFAULT_HATCH_COLOR if hatch else SURFACE)
    ax.barh(y, width, left=left, height=0.5, fill=False, edgecolor=SURFACE, linewidth=2)
    if label and width >= 0.06:
        ax.text(left + width / 2, y, label, ha="center", va="center", fontsize=16, fontweight="bold",
                color=TEXT_PRIMARY if dark_text else "white")


def plot(stats: dict, out: Path, name: str = "tfp_adam_step_size") -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.04)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(3.35, -0.75)
    ax.axis("off")
    fig.text(0.03, 0.94, "TensorFlow Probability: every repository uses Adam, and most change only its step size",
             fontsize=18, fontweight="bold", color=TEXT_PRIMARY)

    rows = [
        ("Use Adam (any step size)", f"of all {stats['repos_readable']} TFP repositories",
         stats["repos_adam"], stats["repos_readable"]),
        ("Use step size 0.1", f"of the {stats['repos_adam']} Adam repositories",
         stats["repos_01"], stats["repos_adam"]),
    ]
    for y, (title, subtitle, part, whole) in enumerate(rows):
        share = part / whole
        ax.text(0, y - 0.42, title, fontsize=16.5, fontweight="bold", color=TEXT_PRIMARY, va="bottom")
        ax.text(0.405, y - 0.42, subtitle, fontsize=15.5, color=TEXT_SECONDARY, va="bottom")
        segment(ax, y, 0, share, DEFAULT_COLOR if y == 0 else NON_DEFAULT_COLOR,
                f"{share:.0%}  ({part} repos)", hatch=None if y == 0 else "///")
        if share < 1:
            segment(ax, y, share, 1 - share, REST_COLOR, "")

    repos = stats["change_repos"]
    total = sum(repos.values())
    y = 2
    ax.text(0, y - 0.42, "What the non-default optimizers change", fontsize=16.5, fontweight="bold",
            color=TEXT_PRIMARY, va="bottom")
    ax.text(0.405, y - 0.42, f"of the {total} repositories with a non-default TFP optimizer", fontsize=15.5,
            color=TEXT_SECONDARY, va="bottom")
    left = 0.0
    legend = []
    for kind, color, hatch, dark in ((ONLY_01, NON_DEFAULT_COLOR, "///", False),
                                     (ONLY_OTHER, OTHER_LR_COLOR, None, True),
                                     (BOTH, BOTH_COLOR, "...", True),
                                     (OTHER, OTHER_CHANGE_COLOR, None, True)):
        n = repos.get(kind, 0)
        if not n:
            continue
        share = n / total
        segment(ax, y, left, share, color, f"{share:.0%}", hatch=hatch, dark_text=dark)
        legend.append((kind, n, color, hatch))
        left += share
    # Two rows, filled column by column so the legend reads in bar order.
    for k, (kind, n, color, hatch) in enumerate(legend):
        lx = 0.47 * (k // 2)
        ly = y + 0.5 + 0.3 * (k % 2)
        ax.add_patch(plt.Rectangle((lx, ly - 0.09), 0.02, 0.18, facecolor=color, hatch=hatch,
                                   edgecolor=NON_DEFAULT_HATCH_COLOR if hatch == "///" else TEXT_SECONDARY,
                                   linewidth=0 if hatch != "..." else 0.4))
        ax.text(lx + 0.03, ly, f"{kind}: {n} repo{'s' if n != 1 else ''}", fontsize=18, color=TEXT_PRIMARY, va="center")

    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=400, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/github-vi-software-stan.sqlite"))
    parser.add_argument("--drop-unreadable", action="store_true",
                        help="drop call sites with any unreadable component (writes *_readable_only)")
    args = parser.parse_args()
    base = args.db.parent / args.db.stem / "component_defaults"
    csv.field_size_limit(10**9)
    rows = [r for r in csv.DictReader((base / "component_call_sites.csv").open())
            if r["package"] == "TensorFlow Probability"]
    if args.drop_unreadable:
        rows = [r for r in rows if "unresolved" not in
                (r[c] for c in ("optimizer", "mc_samples", "transforms", "iterations", "convergence"))]
    name = "tfp_adam_step_size" + ("_readable_only" if args.drop_unreadable else "")
    stats = compute(rows)
    print(stats)
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot(stats, out, name)
    print(f"Wrote {out / (name + '.png')}")


if __name__ == "__main__":
    main()
