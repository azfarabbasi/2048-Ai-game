"""Draw the report charts from the experiment CSV files.

Run after experiments.py (needs matplotlib):

    python charts.py

Every chart is computed from per_game.csv, so nothing is typed in by hand.
"""

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402


BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TILE_RAMP = ["#dbe9f9", "#b5d2f3", "#86b5eb", "#5797e0", "#2a78d6", "#1d5ca8", "#123f75", "#0a2547"]

plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlecolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in ("score", "highest_tile", "moves", "reached_2048"):
            row[key] = int(row[key])
        for key in ("mean_decision_ms", "mean_nodes_expanded"):
            row[key] = float(row[key])
    return rows


def group_by_label(rows):
    """Return {agent label: [rows]} keeping the order in which labels appear."""
    groups = {}
    for row in rows:
        groups.setdefault(row["agent_label"], []).append(row)
    return groups


def short(label):
    return label.replace("Expectimax depth ", "Expectimax d")


def save(fig, folder, name):
    path = folder / name
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def bar_chart(groups, value, title, xlabel, folder, name, fmt):
    labels = list(groups)
    values = [value(groups[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.2))
    bars = ax.barh([short(l) for l in labels], values, color=BLUE, height=0.6)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + fmt(v),
                va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, max(values) * 1.15 or 1)
    save(fig, folder, name)


def tile_distribution(groups, folder):
    tiles = sorted({row["highest_tile"] for rows in groups.values() for row in rows})
    tiles = tiles[-len(TILE_RAMP):]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(groups) + 1.6))
    labels = [short(l) for l in groups]
    # Spread the tiles evenly over the light-to-dark ramp (bigger tile = darker).
    step = (len(TILE_RAMP) - 1) / max(len(tiles) - 1, 1)
    colours = [TILE_RAMP[round(i * step)] for i in range(len(tiles))]
    left = [0.0] * len(groups)
    for index, tile in enumerate(tiles):
        shares = []
        for rows in groups.values():
            counts = Counter(min(max(r["highest_tile"], tiles[0]), tiles[-1]) for r in rows)
            shares.append(100 * counts[tile] / len(rows))
        ax.barh(labels, shares, left=left, color=colours[index],
                edgecolor="white", linewidth=1.5, height=0.6, label=str(tile))
        left = [a + b for a, b in zip(left, shares)]
    ax.invert_yaxis()
    ax.grid(visible=False)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of games")
    ax.set_title("Highest tile reached (share of games)", loc="left")
    ax.legend(title="Highest tile", ncol=len(tiles), loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)
    save(fig, folder, "highest_tile_distribution.png")


def score_boxplot(groups, folder):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([[r["score"] for r in rows] for rows in groups.values()],
               tick_labels=[short(l) for l in groups], patch_artist=True,
               boxprops={"facecolor": "#b5d2f3", "edgecolor": BLUE},
               medianprops={"color": "#123f75", "linewidth": 2},
               whiskerprops={"color": BLUE}, capprops={"color": BLUE},
               flierprops={"markeredgecolor": BLUE, "markersize": 4})
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Final score")
    ax.set_title("Score distribution per agent", loc="left")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, folder, "score_boxplot.png")


def depth_cost(groups, folder):
    """Two separate panels (never two y-axes on one chart): time and nodes."""
    points = sorted(
        (int(rows[0]["depth"]), rows) for label, rows in groups.items()
        if rows[0]["agent"] == "expectimax" and not rows[0]["ablated_feature"]
    )
    if not points:
        return
    depths = [d for d, _ in points]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ylabel, title in (
        (left, "mean_decision_ms", "ms per move (log scale)", "Decision time vs depth"),
        (right, "mean_nodes_expanded", "nodes per move (log scale)", "Nodes expanded vs depth"),
    ):
        values = [statistics.fmean(r[key] for r in rows) for _, rows in points]
        ax.plot(depths, values, color=BLUE, linewidth=2, marker="o", markersize=8)
        for d, v in zip(depths, values):
            ax.annotate(f"{v:,.1f}" if v < 100 else f"{v:,.0f}", (d, v),
                        textcoords="offset points", xytext=(0, 10), ha="center",
                        color=TEXT, fontsize=9)
        ax.set_yscale("log")
        ax.set_xticks(depths)
        ax.set_xlabel("Expectimax depth")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    save(fig, folder, "depth_cost.png")


def ablation(groups, folder):
    pairs = []
    for label, rows in groups.items():
        if rows[0]["agent"] == "expectimax" and not rows[0]["ablated_feature"]:
            off = [l for l, r in groups.items()
                   if r[0]["ablated_feature"] and r[0]["depth"] == rows[0]["depth"]]
            if off:
                pairs.append((rows[0]["depth"], rows, groups[off[0]], groups[off[0]][0]["ablated_feature"]))
    if not pairs:
        return
    feature = pairs[0][3]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    xs = range(len(pairs))
    full = [statistics.fmean(r["score"] for r in p[1]) for p in pairs]
    without = [statistics.fmean(r["score"] for r in p[2]) for p in pairs]
    ax.bar([x - width / 2 - 0.01 for x in xs], full, width, color=BLUE, label="Full heuristic")
    ax.bar([x + width / 2 + 0.01 for x in xs], without, width, color=ORANGE, label=f"Without {feature}")
    for x, a, b in zip(xs, full, without):
        ax.text(x - width / 2, a, f"{a:,.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
        ax.text(x + width / 2, b, f"{b:,.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(list(xs), [f"Depth {p[0]}" for p in pairs])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Average score")
    ax.set_title(f"Ablation: Expectimax with and without {feature}", loc="left")
    ax.set_ylim(0, max(full + without) * 1.12)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, folder, "ablation.png")


def main():
    parser = argparse.ArgumentParser(description="Draw charts from experiment results.")
    parser.add_argument("--input", type=Path, default=Path("results/final"))
    parser.add_argument("--output", type=Path, default=Path("results/charts"))
    args = parser.parse_args()

    rows = load_rows(args.input / "per_game.csv")
    groups = group_by_label(rows)
    main_groups = {k: v for k, v in groups.items() if not v[0]["ablated_feature"]}
    args.output.mkdir(parents=True, exist_ok=True)

    bar_chart(main_groups, lambda rs: statistics.fmean(r["score"] for r in rs),
              "Average final score", "Average score", args.output, "average_score.png",
              lambda v: f"{v:,.0f}")
    bar_chart(main_groups, lambda rs: 100 * statistics.fmean(r["reached_2048"] for r in rs),
              "Games that reached the 2048 tile", "% of games", args.output, "win_rate.png",
              lambda v: f"{v:.0f}%")
    tile_distribution(main_groups, args.output)
    score_boxplot(main_groups, args.output)
    depth_cost(groups, args.output)
    ablation(groups, args.output)


if __name__ == "__main__":
    main()
