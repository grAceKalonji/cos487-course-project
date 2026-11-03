"""Part 2 data loading: everything the dense and sparse models read and write.

Run every Part 2 script from the repo root, e.g.  python -m part2.dense_pretrained
so "data/" below points at the repo's data folder.

Transformer models bring their own tokenizer, so we do NOT reuse Part 1's
cleaned files in data/processed/ (those had stopwords removed for BM25).
Here we only do light cleaning and keep real sentences.
"""
import json
import random
import re
from pathlib import Path

DATA_DIR = Path("data")
JSON_DIR = DATA_DIR / "JSON Files"
QREL_DIR = DATA_DIR / "qrels"

CITATION = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")   # [1], [1, 2], [3-5]
URL = re.compile(r"https?://\S+")
SPACES = re.compile(r"\s+")


def clean_text(text):
    """Light cleaning for transformers: drop citation numbers and URLs, tidy spaces.

    We keep stopwords, capital letters and punctuation because the model was
    trained on normal sentences and uses them to understand meaning.
    """
    text = CITATION.sub(" ", text or "")
    text = URL.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def iter_papers(json_dir=JSON_DIR):
    """Yield one paper dict at a time (keeps memory low with 34,308 files)."""
    for path in sorted(Path(json_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            yield json.load(f)


def load_paragraphs(json_dir=JSON_DIR, limit=None):
    """Return (paragraph_ids, texts) for every non-empty paragraph.

    limit = stop after this many paragraphs (handy for quick tests).
    """
    ids, texts = [], []
    for paper in iter_papers(json_dir):
        for para_id, text in paper["paragraphs"].items():
            text = clean_text(text)
            if text:
                ids.append(para_id)
                texts.append(text)
                if limit and len(ids) >= limit:
                    return ids, texts
    return ids, texts


def load_paper_texts(json_dir=JSON_DIR):
    """Return {paper_id: "title. abstract"} (only for the optional title+abstract model)."""
    return {p["paper_id"]: clean_text(f'{p["title"]}. {p["abstract"]}') for p in iter_papers(json_dir)}


def load_queries(name, data_dir=DATA_DIR):
    """Return [(query_id, text)] from data/Train.json, Study.json or Test.json."""
    path = Path(data_dir) / f"{name.capitalize()}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - add it to data/ first")
    with open(path, encoding="utf-8") as f:
        return [(str(q["query_id"]), clean_text(q["query"])) for q in json.load(f)]


def load_qrels(split, level, qrel_dir=QREL_DIR):
    """Return {query_id: {doc_id: grade}} for grades > 0 (1 = partial, 2 = relevant).

    split = "Train" or "Study", level = "paper" or "paragraph".
    """
    qrels = {}
    with open(Path(qrel_dir) / f"{split.capitalize()}_{level}_qrel.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


def train_val_split(query_ids, val_ratio=0.1, seed=42):
    """Shuffle Train query ids with a fixed seed and hold out val_ratio for validation."""
    ids = sorted(query_ids, key=int)
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return sorted(ids[n_val:], key=int), sorted(ids[:n_val], key=int)


def write_run(run, path, run_name, top_k=100):
    """Write {query_id: [(doc_id, score), ...]} as a 6-column, TAB-separated TREC file.

    The spec asks for .tsv files: qid  Q0  docid  rank  score  run_id
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for qid in sorted(run, key=int):
            ranked = sorted(run[qid], key=lambda x: -x[1])[:top_k]
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                out.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{run_name}\n")
