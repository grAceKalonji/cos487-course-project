"""Dense retrieval: rank paragraphs and papers with a sentence-embedding model.

The model turns every paragraph and every query into an embedding (a list of
768 numbers that captures meaning). Vectors are normalized to length 1, so a
dot product between a query and a paragraph IS their cosine similarity.

Works with any Hugging Face model that sentence-transformers can load:
    --model malteos/scincl            (pretrained, science papers; default)
    --model part2/models/dense_ft     (our fine-tuned copy, later commit)

Run from the repo root (on the GPU server for the full corpus):
    python -m part2.dense_retrieve --split study
    python -m part2.dense_retrieve --split study --limit 50000      # quick test

Writes (data/part2_runs/):
    marcelsSearchParty_paragraph_dense_<split>.tsv
    marcelsSearchParty_paper_dense_<split>.tsv
The paragraph embeddings are cached in data/part2_cache/ so the second split
(and re-runs) skip the slow encoding step.
"""
import argparse
import time
from pathlib import Path

import torch

from part2.data import DATA_DIR, load_paragraphs, load_queries, write_run

TEAM = "marcelsSearchParty"
RUN_DIR = DATA_DIR / "part2_runs"
CACHE_DIR = DATA_DIR / "part2_cache"


def paper_of(para_id):
    """'2408.00001_7' -> '2408.00001'"""
    return para_id.rsplit("_", 1)[0]


def encode(model, texts, batch_size):
    """Embed texts, normalized to length 1, stored as float16 on the CPU.

    float16 halves memory: 2.28M paragraphs x 768 numbers is ~3.5 GB instead of ~7 GB.
    """
    emb = model.encode(texts, batch_size=batch_size, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=True)
    return emb.to(torch.float16).cpu()


def search(query_emb, doc_emb, k, device="cpu", chunk=200_000):
    """Top-k documents per query by dot product (= cosine for normalized vectors).

    The corpus is scored in chunks so it never has to fit on the GPU at once.
    After each chunk we keep only the best k so far (a running top-k).
    Returns (scores, indexes), both shaped [num_queries, k].
    """
    q = query_emb.to(device=device, dtype=torch.float32)
    best_s = best_i = None
    for start in range(0, doc_emb.shape[0], chunk):
        d = doc_emb[start:start + chunk].to(device=device, dtype=torch.float32)
        s = q @ d.T
        s, i = torch.topk(s, k=min(k, s.shape[1]), dim=1)
        i = i + start
        if best_s is not None:
            s, i = torch.cat([best_s, s], dim=1), torch.cat([best_i, i], dim=1)
            s, pick = torch.topk(s, k=min(k, s.shape[1]), dim=1)
            i = torch.gather(i, 1, pick)
        best_s, best_i = s, i
    return best_s.cpu(), best_i.cpu()


def paragraph_run(query_ids, scores, idx, para_ids, k=100):
    """{qid: [(paragraph_id, score), ...]} keeping the top k."""
    return {qid: [(para_ids[j], float(s)) for s, j in zip(scores[n, :k].tolist(), idx[n, :k].tolist())]
            for n, qid in enumerate(query_ids)}


def paper_run(query_ids, scores, idx, para_ids, k=100):
    """Paper score = score of its best paragraph ("find the needle, credit the haystack").

    Uses every retrieved paragraph (we retrieve 1,000 per query) so there are
    enough distinct papers to fill the top k.
    """
    run = {}
    for n, qid in enumerate(query_ids):
        best = {}
        for s, j in zip(scores[n].tolist(), idx[n].tolist()):
            pid = paper_of(para_ids[j])
            if pid not in best:          # results arrive best-first, so the first hit is the max
                best[pid] = float(s)
        run[qid] = sorted(best.items(), key=lambda x: -x[1])[:k]
    return run


def load_or_encode_corpus(model, model_name, batch_size, limit):
    """Paragraph embeddings, from cache if this model + limit was encoded before."""
    tag = model_name.strip("/").replace("/", "__") + (f"_limit{limit}" if limit else "")
    path = CACHE_DIR / f"{tag}.pt"
    if path.exists():
        print(f"Loading cached embeddings: {path}")
        saved = torch.load(path)
        return saved["ids"], saved["emb"]
    ids, texts = load_paragraphs(limit=limit)
    print(f"Encoding {len(ids):,} paragraphs with {model_name} ...")
    t = time.time()
    emb = encode(model, texts, batch_size)
    print(f"Encoded in {time.time() - t:.0f}s")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "emb": emb}, path)
    return ids, emb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="malteos/scincl")
    p.add_argument("--split", default="study", choices=["study", "test"])
    p.add_argument("--run_tag", default="dense", help="dense, or MODEL1/MODEL2 for optional runs")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--para_pool", type=int, default=1000, help="paragraphs retrieved before ranking papers")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="only the first N paragraphs (testing)")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer   # heavy import, only when running

    if torch.cuda.is_available():
        device = "cuda"                      # Nvidia GPU (school server)
    elif torch.backends.mps.is_available():
        device = "mps"                       # Apple Silicon Mac
    else:
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cpu" and not args.limit:
        raise SystemExit("No GPU found. The full corpus would take days on the CPU. "
                         "Fix torch/CUDA first (see part2/requirements.txt) or use --limit to test.")
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length

    para_ids, para_emb = load_or_encode_corpus(model, args.model, args.batch_size, args.limit or None)
    queries = load_queries(args.split)
    qids = [q for q, _ in queries]
    q_emb = encode(model, [t for _, t in queries], args.batch_size)

    scores, idx = search(q_emb, para_emb, k=args.para_pool, device=device)
    for level, run in (("paragraph", paragraph_run(qids, scores, idx, para_ids, args.top_k)),
                       ("paper", paper_run(qids, scores, idx, para_ids, args.top_k))):
        name = f"{TEAM}_{level}_{args.run_tag}_{args.split}"
        write_run(run, RUN_DIR / f"{name}.tsv", name, args.top_k)
        print(f"Wrote {RUN_DIR / name}.tsv")


if __name__ == "__main__":
    main()
