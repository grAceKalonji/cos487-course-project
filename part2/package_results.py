"""Build and check the "Retrieval Results" folder for submission.

Spec: "A folder named Retrieval Results containing all result files named exactly
as specified. Evaluation scripts are automated; incorrect filenames or formats
will result in a zero score for evaluation."

Copies the 8 required run files (+ the manual qrel, if it exists) into
submission/Retrieval Results/ and checks every file:
  - exact file name
  - 6 tab-separated columns: qid Q0 docid rank score run_id
  - every query of the set present (250 Study / 250 Test), at most 100 results each,
    ranks 1..n in order, no duplicate documents per query
Only the required files are copied (no densePretrained runs).

Run from the repo root:
    python -m part2.package_results
"""
import json
import shutil
from pathlib import Path

from part2.data import DATA_DIR

TEAM = "marcelsSearchParty"
RUN_DIR = DATA_DIR / "part2_runs"
QREL = DATA_DIR / "part2_annotation" / f"{TEAM}_qrel_part2_test.tsv"
OUT = Path("submission") / "Retrieval Results"
REQUIRED = [f"{TEAM}_{level}_{model}_{split}.tsv"
            for split in ("study", "test") for level in ("paper", "paragraph") for model in ("dense", "sparse")]


def check_run(path, expected_qids, top_k=100):
    """Return a list of problems (empty = file is fine)."""
    problems, seen = [], {}
    for n, line in enumerate(open(path, encoding="utf-8"), start=1):
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 6:
            problems.append(f"line {n}: {len(cols)} tab-separated columns, expected 6")
            break
        qid, q0, doc, rank = cols[0], cols[1], cols[2], cols[3]
        if q0 != "Q0":
            problems.append(f"line {n}: second column is {q0!r}, expected 'Q0'")
            break
        docs = seen.setdefault(qid, [])
        if int(rank) != len(docs) + 1:
            problems.append(f"query {qid}: rank {rank} out of order")
            break
        if doc in docs:
            problems.append(f"query {qid}: {doc} appears twice")
            break
        docs.append(doc)
    missing = set(expected_qids) - set(seen)
    extra = set(seen) - set(expected_qids)
    if missing:
        problems.append(f"{len(missing)} queries missing, e.g. {sorted(missing, key=int)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unexpected query ids, e.g. {sorted(extra, key=int)[:3]}")
    over = [q for q, d in seen.items() if len(d) > top_k]
    short = [q for q, d in seen.items() if len(d) < top_k]
    if over:
        problems.append(f"{len(over)} queries have more than {top_k} results")
    if short:
        problems.append(f"note: {len(short)} queries have fewer than {top_k} results")
    return problems


def check_qrel(path, expected_qids):
    problems, seen = [], set()
    for n, line in enumerate(open(path, encoding="utf-8"), start=1):
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 4 or cols[1] != "0" or cols[3] not in ("0", "1", "2"):
            problems.append(f"line {n}: expected 'qid 0 paraid 0|1|2', got {line.strip()!r}")
            break
        seen.add(cols[0])
    if set(expected_qids) - seen:
        problems.append(f"{len(set(expected_qids) - seen)} Test queries not graded yet")
    return problems


def main():
    qids = {s: [str(q["query_id"]) for q in json.load(open(DATA_DIR / f"{s.capitalize()}.json"))]
            for s in ("study", "test")}
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in REQUIRED:
        src = RUN_DIR / name
        if not src.exists():
            print(f"MISSING  {name}")
            ok = False
            continue
        problems = check_run(src, qids["study" if name.endswith("_study.tsv") else "test"])
        hard = [p for p in problems if not p.startswith("note:")]
        ok &= not hard
        shutil.copy2(src, OUT / name)
        print(f"{'OK     ' if not hard else 'PROBLEM'}  {name}" + "".join(f"\n         - {p}" for p in problems))
    if QREL.exists():
        problems = check_qrel(QREL, qids["test"])
        shutil.copy2(QREL, OUT / QREL.name)
        print(f"{'OK     ' if not problems else 'CHECK  '}  {QREL.name}" + "".join(f"\n         - {p}" for p in problems))
    else:
        print(f"MISSING  {QREL.name} (run python -m part2.annotate)")
        ok = False
    print(f"\n{'All files ready' if ok else 'Fix the items above before submitting'}: {OUT}/")


if __name__ == "__main__":
    main()
