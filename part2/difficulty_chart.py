"""MRR by query difficulty (Easy / Medium / Hard) for every model - Figure 3 of the paper.

Part 1 feedback: "For the analysis for TF-IDF and BM25 per categories you could
have picked a metric like MRR and provided an informative bar chart."

Reads data/part2_eval/per_query_<level>.csv (made by part2.evaluate_part2) and the
difficulty labels in data/Study.json. No GPU needed.

Run from the repo root:
    python -m part2.difficulty_chart
Writes data/part2_eval/:
    mrr_by_difficulty_<level>.png   grouped bar chart
    mrr_by_difficulty_<level>.csv   the numbers behind it (+ query counts)
"""
import argparse
import csv
import json
from pathlib import Path

from part2.data import DATA_DIR

EVAL_DIR = DATA_DIR / "part2_eval"
LEVELS = ["Easy", "Medium", "Hard"]
ORDER = ["bm25", "densePretrained", "dense", "sparse"]          # chart order, when present
NAMES = {"bm25": "BM25 (Part 1)", "densePretrained": "Dense, pretrained",
         "dense": "Dense, fine-tuned", "sparse": "Sparse (SPLADE)"}
COLORS = {"bm25": "#888780", "densePretrained": "#AFA9EC", "dense": "#534AB7", "sparse": "#1D9E75"}


def mrr_by_difficulty(per_query_csv, labels, metric="mrr"):
    """{model: {difficulty: mean metric}} and {difficulty: number of queries}."""
    rows = list(csv.DictReader(open(per_query_csv, encoding="utf-8")))
    models = [c[:-len(f" {metric}")] for c in rows[0] if c.endswith(f" {metric}")]
    sums = {m: {d: 0.0 for d in LEVELS} for m in models}
    counts = {d: 0 for d in LEVELS}
    for r in rows:
        d = labels.get(r["query_id"])
        if d not in counts:
            continue
        counts[d] += 1
        for m in models:
            sums[m][d] += float(r[f"{m} {metric}"])
    means = {m: {d: (sums[m][d] / counts[d] if counts[d] else 0.0) for d in LEVELS} for m in models}
    return means, counts


def plot(means, counts, level, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in ORDER if m in means] + [m for m in means if m not in ORDER]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for j, m in enumerate(models):
        xs = [i + (j - (len(models) - 1) / 2) * width for i in range(len(LEVELS))]
        bars = ax.bar(xs, [means[m][d] for d in LEVELS], width, label=NAMES.get(m, m),
                      color=COLORS.get(m, None))
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(range(len(LEVELS)), [f"{d}\n({counts[d]} queries)" for d in LEVELS])
    ax.set_ylabel("MRR (Relevant + Partially Relevant)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{level.capitalize()} level, Study queries", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+", default=["paragraph", "paper"])
    args = p.parse_args()
    labels = {str(q["query_id"]): q["label"] for q in json.load(open(DATA_DIR / "Study.json"))}
    for level in args.levels:
        means, counts = mrr_by_difficulty(EVAL_DIR / f"per_query_{level}.csv", labels)
        with open(EVAL_DIR / f"mrr_by_difficulty_{level}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model"] + [f"{d} (n={counts[d]})" for d in LEVELS])
            for m, by in means.items():
                w.writerow([m] + [round(by[d], 3) for d in LEVELS])
        plot(means, counts, level, EVAL_DIR / f"mrr_by_difficulty_{level}.png")
        print(f"[{level}] " + " | ".join(
            f"{m}: " + ", ".join(f"{d[0]} {by[d]:.3f}" for d in LEVELS) for m, by in means.items()))
    print(f"Saved to {EVAL_DIR}/")


if __name__ == "__main__":
    main()
