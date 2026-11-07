"""Manually grade the best model's top paragraph for every Test query (terminal tool).

Spec: "For your best-performing model on the Study queries, evaluate the top
retrieved paragraph for each query in the Test query set. Create a standard QREL
file by manually assigning relevance scores using this 3-point scale:
  2 (Fully Relevant): The paragraph fully addresses the query.
  1 (Partially Relevant): The paragraph provides useful context but does not fully address the query.
  0 (Non-Relevant): The paragraph provides no useful information.
You should provide a file named teamName_qrel_part2_test.tsv in standard TREC
format with 4 columns: qid 0 paraid relevance"

Our best Study model is SPLADE (sparse), so by default we grade its run.

Run from the repo root (on your Mac, no GPU needed):
    python -m part2.annotate                 # start or continue grading
    python -m part2.annotate --summary       # counts of 2 / 1 / 0 so far

Keys while grading:  2 / 1 / 0 = grade,  b = back one,  s = skip for now,  q = save and quit
Every grade is saved immediately, so you can stop and continue any time.
Words shared by the query and the paragraph are highlighted to speed up reading.
"""
import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path

from part2.data import DATA_DIR, JSON_DIR, clean_text, load_queries

TEAM = "marcelsSearchParty"
RUN_DIR = DATA_DIR / "part2_runs"
OUT_DIR = DATA_DIR / "part2_annotation"
QREL_PATH = OUT_DIR / f"{TEAM}_qrel_part2_test.tsv"
SCALE = ("2 = Fully relevant: fully addresses the query\n"
         "1 = Partially relevant: useful context, but does not fully address it\n"
         "0 = Non-relevant: no useful information")
STOP = set("a an the of in on for to and or with by is are was were be how what why which when "
           "does do did can could this that these those from as at into its it their used use using "
           "between within about than such over under based".split())
BOLD, RESET = "\033[1;33m", "\033[0m"


def top_paragraphs(run_path):
    """{qid: rank-1 paragraph id} from a TREC run file."""
    top = {}
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and p[3] == "1":
                top[p[0]] = p[2]
    return top


def paragraph_info(para_id, json_dir=JSON_DIR):
    """(paper title, paragraph text) for one paragraph id like '2409.10230_100'."""
    paper_id = para_id.rsplit("_", 1)[0]
    with open(Path(json_dir) / f"{paper_id}.json", encoding="utf-8") as f:
        paper = json.load(f)
    return clean_text(paper["title"]), clean_text(paper["paragraphs"].get(para_id, ""))


def load_grades(path=QREL_PATH):
    """{qid: (paragraph id, grade)} already saved."""
    grades = {}
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) == 4:
                grades[p[0]] = (p[2], int(p[3]))
    return grades


def save_grades(grades, path=QREL_PATH):
    """Write the qrel file: qid <TAB> 0 <TAB> paraid <TAB> relevance, sorted by qid."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(grades, key=int):
            para_id, grade = grades[qid]
            f.write(f"{qid}\t0\t{para_id}\t{grade}\n")


def highlight(text, query, color=True):
    """Make words the paragraph shares with the query stand out."""
    words = {w for w in re.findall(r"[a-z0-9\-]+", query.lower()) if w not in STOP and len(w) > 2}
    if not color or not words:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")\b",
                         re.IGNORECASE)
    return pattern.sub(lambda m: f"{BOLD}{m.group(0)}{RESET}", text)


def summary(grades, total):
    counts = {g: sum(1 for _, x in grades.values() if x == g) for g in (2, 1, 0)}
    done = len(grades)
    line = f"Graded {done}/{total} | 2: {counts[2]}  1: {counts[1]}  0: {counts[0]}"
    if done:
        line += (f" | Precision@1 (any) = {(counts[2] + counts[1]) / done:.3f}"
                 f", Precision@1 (full) = {counts[2] / done:.3f}")
    return line


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=str(RUN_DIR / f"{TEAM}_paragraph_sparse_test.tsv"),
                   help="run file of the best Study model (default: SPLADE)")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--no_color", action="store_true")
    args = p.parse_args()

    queries = load_queries("test")
    labels = {str(q["query_id"]): q.get("label", "") for q in json.load(open(DATA_DIR / "Test.json"))}
    top = top_paragraphs(args.run)
    grades = load_grades()
    if args.summary:
        print(summary(grades, len(queries)))
        return

    todo = [(qid, text) for qid, text in queries if qid not in grades]
    width = min(100, shutil.get_terminal_size((100, 20)).columns)
    history, i = [], 0
    while i < len(todo):
        qid, query = todo[i]
        para_id = top.get(qid)
        if para_id is None:
            print(f"Query {qid}: no rank-1 paragraph in {args.run}; skipping.")
            i += 1
            continue
        title, text = paragraph_info(para_id)
        print("\n" + "=" * width)
        print(summary(grades, len(queries)))
        print(f"Query {qid} [{labels.get(qid, '')}]:")
        print(textwrap.fill(query, width))
        print(f"\nTop paragraph {para_id}  |  Paper: {title}")
        print(highlight(textwrap.fill(text, width), query, not args.no_color))
        print("\n" + SCALE)
        key = input("Grade [2/1/0], b = back, s = skip, q = quit: ").strip().lower()
        if key in ("2", "1", "0"):
            grades[qid] = (para_id, int(key))
            save_grades(grades)
            history.append(i)
            i += 1
        elif key == "b" and history:
            i = history.pop()
            grades.pop(todo[i][0], None)
            save_grades(grades)
        elif key == "s":
            i += 1
        elif key == "q":
            break
    print("\n" + summary(grades, len(queries)))
    print(f"Saved to {QREL_PATH}")


if __name__ == "__main__":
    main()
