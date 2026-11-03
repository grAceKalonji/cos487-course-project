"""Score Part 2 runs on the Study queries and test significance vs each other and Part 1 BM25.

Spec: "Using the Study.json file, provide the following metrics: nDCG, nDCG@5, MRR,
Precision@5, Precision@10. For MRR and Precision@5, you should provide two variants,
one considering both Relevant and Partially Relevant as relevant and one just
considering the Relevant (with score 2) as relevant. Apply statistical significance
testing between your Part 2 models, as well as between your Part 2 models and your
best-performing model from Part 1."

Run from the repo root (no GPU needed, takes seconds):
    python -m part2.evaluate_part2
It picks up every  *_study.tsv  in data/part2_runs/ plus Part 1 BM25 from
data/part1_runs/, for both the paper and paragraph level.

Writes data/part2_eval/:
    results_<level>.md     table you can paste into notes (scores + significance)
    results_<level>.json   every number, incl. p-values and win/tie/loss
    per_query_<level>.csv  per-query nDCG@5 and MRR for each model (for analysis)
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

from part2.data import DATA_DIR, load_qrels

RUN_DIRS = [DATA_DIR / "part2_runs", DATA_DIR / "part1_runs"]
OUT_DIR = DATA_DIR / "part2_eval"
METRICS_ANY = ["ndcg", "ndcg@5", "mrr", "precision@5", "precision@10"]   # grade 1 or 2 = relevant
METRICS_FULL = ["mrr", "precision@5"]                                   # only grade 2 = relevant


def model_name(path, level):
    """'marcelsSearchParty_paragraph_densePretrained_study.tsv' -> 'densePretrained'."""
    parts = Path(path).stem.split("_")
    return "_".join(parts[parts.index(level) + 1:-1])


def read_run(path):
    """TREC run file (tabs or spaces) -> {qid: {doc_id: score}}."""
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                run.setdefault(p[0], {})[p[2]] = float(p[4])
    return run


def find_runs(level, run_dirs=RUN_DIRS):
    """{model name: run} for every <team>_<level>_<model>_study.tsv we can find."""
    runs = {}
    for d in run_dirs:
        for path in sorted(Path(d).glob(f"*_{level}_*_study.tsv")):
            runs[model_name(path, level)] = read_run(path)
    return runs


def full_only(qrels):
    """Keep only grade-2 (Relevant) judgments, for the 'full' variants."""
    out = {q: {d: g for d, g in docs.items() if g >= 2} for q, docs in qrels.items()}
    return {q: docs for q, docs in out.items() if docs}


def evaluate_level(level, runs, qrels, stat_test="student", max_p=0.05):
    """Scores + pairwise significance for every model at one level."""
    from ranx import Qrels, Run, compare

    names = list(runs)
    variants = {"any": (Qrels.from_dict(qrels), METRICS_ANY),
                "full": (Qrels.from_dict(full_only(qrels)), METRICS_FULL)}
    result = {"models": names, "scores": {n: {} for n in names}, "pairs": {}}
    for variant, (q, metrics) in variants.items():
        ranx_runs = []
        for n in names:
            kept = {qid: docs for qid, docs in runs[n].items() if qid in q.qrels}
            ranx_runs.append(Run.from_dict(kept, name=n))
        rep = compare(q, ranx_runs, metrics=metrics, stat_test=stat_test, max_p=max_p).to_dict()
        for n in names:
            for m in metrics:
                result["scores"][n][f"{m} ({variant})"] = rep[n]["scores"][m]
        for a, b in combinations(names, 2):
            for m in metrics:
                p = rep[a]["comparisons"][b][m]
                wtl = rep[a]["win_tie_loss"][b][m]
                sa, sb = rep[a]["scores"][m], rep[b]["scores"][m]
                result["pairs"].setdefault(f"{a} vs {b}", {})[f"{m} ({variant})"] = {
                    "p_value": p, "significant": p < max_p,
                    "better": (a if sa > sb else b) if p < max_p else "none",
                    "win_tie_loss": f'{wtl["W"]}/{wtl["T"]}/{wtl["L"]}'}
    return result


def per_query(runs, qrels, metrics=("ndcg@5", "mrr")):
    """{model: {qid: {metric: value}}} so we can find wins, losses and failures later."""
    from ranx import Qrels, Run, evaluate
    q = Qrels.from_dict(qrels)
    out = {}
    for n, run in runs.items():
        r = Run.from_dict({qid: d for qid, d in run.items() if qid in qrels}, name=n)
        evaluate(q, r, list(metrics), make_comparable=True)
        out[n] = {qid: {m: r.scores[m][qid] for m in metrics} for qid in r.scores[metrics[0]]}
    return out


def markdown_table(level, result):
    cols = [f"{m} (any)" for m in METRICS_ANY] + [f"{m} (full)" for m in METRICS_FULL]
    lines = [f"### {level.capitalize()} level (Study, 250 queries)", "",
             "| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for n in result["models"]:
        lines.append(f"| {n} | " + " | ".join(f'{result["scores"][n][c]:.3f}' for c in cols) + " |")
    lines += ["", "Significance (paired t-test, p < 0.05). Win/tie/loss = queries where the first model was better / equal / worse.", "",
              "| Pair | Metric | p-value | Better | W/T/L |", "|---|---|---|---|---|"]
    for pair, ms in result["pairs"].items():
        for m, v in ms.items():
            lines.append(f'| {pair} | {m} | {v["p_value"]:.4f} | {v["better"]} | {v["win_tie_loss"]} |')
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", nargs="+", default=["paper", "paragraph"])
    p.add_argument("--stat_test", default="student", choices=["student", "fisher"])
    p.add_argument("--max_p", type=float, default=0.05)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        runs = find_runs(level)
        if len(runs) < 2:
            print(f"[{level}] need at least 2 study runs, found {list(runs)}; skipping")
            continue
        qrels = load_qrels("Study", level)
        result = evaluate_level(level, runs, qrels, args.stat_test, args.max_p)
        (OUT_DIR / f"results_{level}.json").write_text(json.dumps(result, indent=2))
        table = markdown_table(level, result)
        (OUT_DIR / f"results_{level}.md").write_text(table)

        pq = per_query(runs, qrels)
        with open(OUT_DIR / f"per_query_{level}.csv", "w", newline="") as f:
            w = csv.writer(f)
            names = list(pq)
            w.writerow(["query_id"] + [f"{n} {m}" for n in names for m in ("ndcg@5", "mrr")])
            for qid in sorted(qrels, key=int):
                w.writerow([qid] + [round(pq[n].get(qid, {}).get(m, 0.0), 4) for n in names for m in ("ndcg@5", "mrr")])
        print(table)
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
