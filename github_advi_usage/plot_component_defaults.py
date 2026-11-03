"""Shareable figures for the per-component default analysis.

Reads component_defaults/component_summary.csv (from component_defaults.py) and
writes, for call sites and for repositories:

  component_defaults_table_<unit>.{png,svg,pdf}   the summary as a table figure
  component_defaults_plot_<unit>.{png,svg,pdf}    100% stacked bars per package

    python -m github_advi_usage.plot_component_defaults --db data/github-vi-software-stan.sqlite
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
RULE = "#c9c7c1"
# Same colorblind-safe pair as the main analysis figures (Paul Tol), with a
# hatch on the non-default segment so the two stay distinct in grayscale.
DEFAULT_COLOR = "#004488"
NON_DEFAULT_COLOR = "#EE7733"
NON_DEFAULT_HATCH_COLOR = "#B5541A"

PACKAGES = ["NumPyro", "TensorFlow Probability", "PyMC", "Stan"]
COMPONENTS = [
    ("optimizer", "Optimizer"),
    ("mc_samples", "MC samples"),
    ("transforms", "No transformations"),
    ("iterations", "Total/Max iterations"),
    ("convergence", "Convergence setting"),
    ("all", "All components"),
]
UNIT_LABEL = {"call_site": "call sites", "repository": "repositories"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "hatch.linewidth": 0.8,
    "svg.fonttype": "none",
})


def load(path: Path) -> dict[tuple[str, str, str], dict[str, int]]:
    rows = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows[(row["unit"], row["package"], row["component"])] = {
                key: int(row[key]) for key in ("default", "non_default", "unresolved")
            }
    return rows


def load_readable_only(path: Path) -> dict[tuple[str, str, str], dict[str, int]]:
    """Summary counts from component_call_sites.csv after dropping every call
    site with any unreadable component (same aggregation as component_defaults.py)."""
    csv.field_size_limit(10**9)
    with path.open() as handle:
        sites = [r for r in csv.DictReader(handle)
                 if not any(r[c] == "unresolved" for c, _label in COMPONENTS if c != "all")]
    rows = {}
    for unit in UNIT_LABEL:
        for package in [*PACKAGES, "All packages"]:
            subset = [r for r in sites if package == "All packages" or r["package"] == package]
            for component, _label in COMPONENTS:
                if unit == "call_site":
                    labels = [r[component] for r in subset]
                else:
                    by_repo = defaultdict(list)
                    for r in subset:
                        by_repo[r["repo_full_name"]].append(r[component])
                    labels = ["non_default" if "non_default" in v else "default" if "default" in v else "n/a"
                              for v in by_repo.values()]
                counts = Counter(labels)
                if counts["default"] + counts["non_default"]:
                    rows[(unit, package, component)] = {"default": counts["default"],
                                                        "non_default": counts["non_default"], "unresolved": 0}
    return rows


def pct(d: int, n: int) -> str:
    return f"{100 * d / (d + n):.0f}%" if d + n else "–"


def totals(rows, unit: str, package: str) -> int:
    counts = rows[(unit, package, "all")]
    return sum(counts.values())


def plot(rows, unit: str, out: Path, suffix: str = "") -> None:
    fig = plt.figure(figsize=(10.5, 10.0), facecolor=SURFACE)
    panels = [(p, [(c, label) for c, label in COMPONENTS if (unit, p, c) in rows]) for p in PACKAGES]
    heights = [len(c) for _p, c in panels]
    grid = fig.add_gridspec(len(panels), 1, height_ratios=heights, hspace=0.75,
                            left=0.2, right=0.9, top=0.855, bottom=0.05)
    fig.text(0.02, 0.965, "How often is each part of the VI setup left at its default?",
             fontsize=15, fontweight="bold", color=TEXT_PRIMARY, ha="left")
    fig.text(0.02, 0.937, f"Share of GitHub {UNIT_LABEL[unit]} by package. Right-hand number: % default.", fontsize=10, color=TEXT_SECONDARY, ha="left")
    fig.legend(
        handles=[Patch(facecolor=DEFAULT_COLOR, label="Default"),
                 Patch(facecolor=NON_DEFAULT_COLOR, edgecolor=NON_DEFAULT_HATCH_COLOR, hatch="///", linewidth=0,
                       label="Non-default")],
        loc="upper left", bbox_to_anchor=(0.195, 0.925), ncol=3, frameon=False, fontsize=10,
        labelcolor=TEXT_PRIMARY, handlelength=1.4, columnspacing=1.6,
    )
    for index, (package, components) in enumerate(panels):
        ax = fig.add_subplot(grid[index])
        ax.set_facecolor(SURFACE)
        for y, (component, label) in enumerate(components):
            counts = rows[(unit, package, component)]
            d, n = counts["default"], counts["non_default"]
            total = d + n
            left = 0.0
            emphasis = component == "all"
            for value, color, hatch in ((d, DEFAULT_COLOR, None), (n, NON_DEFAULT_COLOR, "///")):
                share = value / total if total else 0
                if share:
                    ax.barh(y, share, left=left, height=0.62, color=color, hatch=hatch,
                            edgecolor=NON_DEFAULT_HATCH_COLOR if hatch else SURFACE, linewidth=0)
                    ax.barh(y, share, left=left, height=0.62, fill=False, edgecolor=SURFACE, linewidth=1.5)
                left += share
            ax.text(1.02, y, pct(d, n), va="center", ha="left", fontsize=11 if emphasis else 10,
                    fontweight="bold" if emphasis else "normal", color=TEXT_PRIMARY, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels([label for _c, label in components], fontsize=10, color=TEXT_PRIMARY)
        for tick, (component, _label) in zip(ax.get_yticklabels(), components):
            if component == "all":
                tick.set_fontweight("bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"] if index == len(panels) - 1 else [],
                           fontsize=9, color=TEXT_SECONDARY)
        ax.tick_params(axis="both", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.annotate(package, xy=(0.02, 1), xycoords=("figure fraction", "axes fraction"), xytext=(0, 22), textcoords="offset points",
                    fontsize=12, fontweight="bold", color=TEXT_PRIMARY, va="bottom", ha="left")
        ax.annotate(f"{totals(rows, unit, package):,} {UNIT_LABEL[unit]} in total", xy=(0.02, 1), xycoords=("figure fraction", "axes fraction"),
                    xytext=(0, 8), textcoords="offset points", fontsize=9, color=TEXT_SECONDARY,
                    va="bottom", ha="left")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"component_defaults_plot_{unit}{suffix}.{ext}", dpi=400, facecolor=SURFACE)
    plt.close(fig)


def table(rows, unit: str, out: Path, suffix: str = "") -> None:
    fig = plt.figure(figsize=(13.5, 4.8), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0.08, 0.98)
    ax.text(0.025, 0.93, f"Default usage by component: GitHub {UNIT_LABEL[unit]}", fontsize=15,
            fontweight="bold", color=TEXT_PRIMARY, ha="left", va="center")
    ax.text(0.025, 0.865, f"Each cell: % default, then default / non-default {UNIT_LABEL[unit]}.   — = not applicable.",
            fontsize=10, color=TEXT_SECONDARY, ha="left", va="center")
    left, right, first_col = 0.025, 0.975, 0.2
    col_w = (right - first_col) / len(COMPONENTS)
    y_header, row_h = 0.77, 0.12
    table_rows = [*PACKAGES, "All packages"]
    y_bottom = y_header - 0.05 - row_h * len(table_rows)
    ax.add_patch(plt.Rectangle((right - col_w, y_bottom), col_w, y_header + 0.06 - y_bottom,
                               facecolor="#eef2f7", edgecolor="none", zorder=0))
    ax.text(left + 0.005, y_header, "Package", fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY, va="center")
    for i, (_component, label) in enumerate(COMPONENTS):
        ax.text(first_col + col_w * (i + 0.5), y_header, label, fontsize=10.5, fontweight="bold",
                color=TEXT_PRIMARY, ha="center", va="center")
    ax.plot([left, right], [y_header - 0.05] * 2, color=TEXT_PRIMARY, linewidth=1)
    for r, package in enumerate(table_rows):
        y = y_header - 0.05 - row_h * (r + 0.5)
        if package == "All packages":
            ax.plot([left, right], [y + row_h / 2] * 2, color=TEXT_PRIMARY, linewidth=1)
        ax.text(left + 0.005, y + 0.022, package, fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY, va="center")
        ax.text(left + 0.005, y - 0.03, f"{totals(rows, unit, package):,} {UNIT_LABEL[unit]} in total", fontsize=8.5,
                color=TEXT_SECONDARY, va="center")
        for i, (component, _label) in enumerate(COMPONENTS):
            cx = first_col + col_w * (i + 0.5)
            counts = rows.get((unit, package, component))
            if counts is None:
                ax.text(cx, y, "—", fontsize=11, color=TEXT_SECONDARY, ha="center", va="center")
                continue
            d, n = counts["default"], counts["non_default"]
            ax.text(cx, y + 0.022, pct(d, n), fontsize=13, fontweight="bold", color=TEXT_PRIMARY,
                    ha="center", va="center")
            ax.text(cx, y - 0.03, f"{d:,} / {n:,}", fontsize=8.5, color=TEXT_SECONDARY,
                    ha="center", va="center")
        ax.plot([left, right], [y - row_h / 2] * 2, color=TEXT_PRIMARY if package == "All packages" else RULE,
                linewidth=1 if package == "All packages" else 0.6)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"component_defaults_table_{unit}{suffix}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=Path("data/github-vi-software-stan.sqlite"))
    parser.add_argument("--drop-unreadable", action="store_true",
                        help="drop call sites with any unreadable component (writes *_readable_only)")
    args = parser.parse_args()
    base = args.db.parent / args.db.stem / "component_defaults"
    if args.drop_unreadable:
        rows = load_readable_only(base / "component_call_sites.csv")
    else:
        rows = load(base / "component_summary.csv")
    suffix = "_readable_only" if args.drop_unreadable else ""
    out = base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for unit in ("call_site", "repository"):
        plot(rows, unit, out, suffix)
        table(rows, unit, out, suffix)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
