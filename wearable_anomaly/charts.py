"""Charts for the README, generated from reports/benchmark.json.

    python -m wearable_anomaly.charts

Writes reports/pr_auc_by_detector.png and reports/recall_by_fault_type.png.
Needs matplotlib (requirements-dev.txt); nothing at runtime imports this module.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from wearable_anomaly import config, faults

# Reference palette (light mode): surface, text inks, one series hue, a sequential blue ramp.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)


def pr_auc_chart(results: dict, path) -> None:
    items = sorted(results.items(), key=lambda kv: kv[1]["detection"]["pr_auc"])
    names = [n for n, _ in items]
    values = [r["detection"]["pr_auc"] for _, r in items]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.barh(names, values, height=0.5, color=SERIES)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.18)
    for y, v in enumerate(values):
        ax.text(v + max(values) * 0.015, y, f"{v:.3f}", va="center", color=TEXT, fontsize=10)
    ax.set_title(
        "PR-AUC by detector (higher is better)", loc="left", color=TEXT, fontsize=13, pad=12
    )
    ax.set_xlabel(
        "Average precision over all readings, unseen test subjects with injected faults",
        color=TEXT_2,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def recall_heatmap(results: dict, path) -> None:
    names = sorted(results, key=lambda n: results[n]["detection"]["pr_auc"], reverse=True)
    kinds = list(faults.FAULT_TYPES)
    grid = [
        [results[n]["detection"]["per_fault_type"][k]["event_recall"] for k in kinds] for n in names
    ]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(kinds)), kinds)
    # Recall alone rewards flagging everything, so show how much each detector flags.
    labels = [
        f"{n}\nflags {results[n]['detection']['flagged_pct']:.1f}% of readings" for n in names
    ]
    ax.set_yticks(range(len(names)), labels, fontsize=9)
    ax.tick_params(colors=TEXT_2, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 2px surface gaps between cells
    ax.set_xticks([x - 0.5 for x in range(1, len(kinds))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(names))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                f"{v:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if v >= 0.55 else TEXT,
            )
    ax.set_title(
        "Share of injected faults caught, by fault type",
        loc="left",
        color=TEXT,
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    report = json.loads((config.REPORTS_DIR / "benchmark.json").read_text())
    results = report["detectors"]
    pr_auc_chart(results, config.REPORTS_DIR / "pr_auc_by_detector.png")
    recall_heatmap(results, config.REPORTS_DIR / "recall_by_fault_type.png")
    print("Wrote reports/pr_auc_by_detector.png and reports/recall_by_fault_type.png")


if __name__ == "__main__":
    main()
