"""Sparse neural retrieval with SPLADE: rank paragraphs and papers with learned word weights.

SPLADE reads a paragraph with a BERT model and outputs one weight for every word
in BERT's vocabulary (30,522 words). Almost all weights are 0. The non-zero ones
are the words the model thinks the text is about - including related words that
never appear in it ("expansion"). Score = sum over shared words of
query weight x paragraph weight (a dot product), just like BM25 adds up per-word
scores. So it is "BM25 with learned weights".

Run from the repo root (GPU server, inside tmux):
    python -m part2.sparse_retrieve --split study
    python -m part2.sparse_retrieve --split test
    python -m part2.sparse_retrieve --split study --limit 50000         # quick test
    python -m part2.sparse_retrieve --show "How does RAG reduce computation?"   # see the learned words

Writes (data/part2_runs/):
    marcelsSearchParty_paragraph_sparse_<split>.tsv
    marcelsSearchParty_paper_sparse_<split>.tsv
Paragraph vectors are cached in data/part2_cache/ (<model>.npz + ids) so the
second split skips the slow encoding step.
"""
import argparse
import json
import time

import numpy as np
import torch

from part2.data import load_paragraphs, load_queries, write_run
from part2.dense_retrieve import CACHE_DIR, RUN_DIR, TEAM, paper_run, paragraph_run

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


def to_csr(sparse_tensor):
    """torch sparse (rows x 30,522) -> scipy CSR matrix (only non-zero weights stored)."""
    from scipy.sparse import csr_matrix
    t = sparse_tensor.coalesce().cpu()
    rows, cols = t.indices().numpy()
    return csr_matrix((t.values().float().numpy(), (rows, cols)), shape=tuple(t.shape), dtype=np.float32)


def encode_corpus(model, texts, batch_size, max_active_dims, chunk=50_000):
    """Encode paragraphs chunk by chunk into one big CSR matrix (paragraphs x vocabulary)."""
    from scipy.sparse import vstack
    parts = []
    for start in range(0, len(texts), chunk):
        emb = model.encode_document(texts[start:start + chunk], batch_size=batch_size,
                                    convert_to_sparse_tensor=True, max_active_dims=max_active_dims,
                                    show_progress_bar=True)
        parts.append(to_csr(emb))
        print(f"  encoded {min(start + chunk, len(texts)):,} / {len(texts):,}")
    return vstack(parts).tocsr()


def search(query_csr, doc_csr, k, chunk=16):
    """Top-k paragraphs per query by dot product over the shared (non-zero) words.

    Returns torch tensors (scores, indexes) shaped [num_queries, k], the same shape
    dense_retrieve.search returns, so paragraph_run / paper_run work unchanged.
    """
    doc_t = doc_csr.T.tocsr()                       # vocabulary x paragraphs
    all_s, all_i = [], []
    for start in range(0, query_csr.shape[0], chunk):
        scores = (query_csr[start:start + chunk] @ doc_t).toarray()     # [chunk, num_paragraphs]
        kk = min(k, scores.shape[1])
        idx = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        all_i.append(np.take_along_axis(idx, order, axis=1))
        all_s.append(np.take_along_axis(part, order, axis=1))
    return torch.from_numpy(np.vstack(all_s)), torch.from_numpy(np.vstack(all_i))


def load_or_encode_corpus(model, model_name, batch_size, max_active_dims, limit):
    """Paragraph vectors, from cache if this model + settings + limit was encoded before."""
    from scipy.sparse import load_npz, save_npz
    tag = model_name.strip("/").replace("/", "__") + f"_top{max_active_dims}" + (f"_limit{limit}" if limit else "")
    mat_path, ids_path = CACHE_DIR / f"{tag}.npz", CACHE_DIR / f"{tag}_ids.json"
    if mat_path.exists() and ids_path.exists():
        print(f"Loading cached sparse vectors: {mat_path}")
        return json.loads(ids_path.read_text()), load_npz(mat_path).tocsr()
    ids, texts = load_paragraphs(limit=limit)
    print(f"Encoding {len(ids):,} paragraphs with {model_name} ...")
    t = time.time()
    mat = encode_corpus(model, texts, batch_size, max_active_dims)
    print(f"Encoded in {time.time() - t:.0f}s | avg non-zero words per paragraph: {mat.nnz / mat.shape[0]:.1f}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(mat_path, mat)
    ids_path.write_text(json.dumps(ids))
    return ids, mat


def show_terms(model, text, top_k=25):
    """Print the words SPLADE gives the highest weight to (handy for the paper/slides)."""
    emb = model.encode_query([text], convert_to_sparse_tensor=True)
    print(f"\nQuery: {text}")
    for word, weight in model.decode(emb[0], top_k=top_k):
        in_text = "" if word.lower() in text.lower() else "   <- added (not in the query)"
        print(f"  {word:20s} {weight:.2f}{in_text}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--split", default="study", choices=["study", "test"])
    p.add_argument("--run_tag", default="sparse")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--para_pool", type=int, default=3000,
                   help="paragraphs retrieved before ranking papers (enough to reach 100 distinct papers)")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--max_active_dims", type=int, default=256,
                   help="keep at most this many non-zero words per paragraph (limits memory)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="only the first N paragraphs (testing)")
    p.add_argument("--show", default="", help="print the learned word weights for this text and exit")
    args = p.parse_args()

    from sentence_transformers import SparseEncoder

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    model = SparseEncoder(args.model, device=device)
    model.max_seq_length = args.max_seq_length

    if args.show:
        show_terms(model, args.show)
        return
    if device == "cpu" and not args.limit:
        raise SystemExit("No GPU found. The full corpus would take days on the CPU. "
                         "Fix torch/CUDA first (see part2/requirements.txt) or use --limit to test.")

    para_ids, para_mat = load_or_encode_corpus(model, args.model, args.batch_size,
                                               args.max_active_dims, args.limit or None)
    queries = load_queries(args.split)
    qids = [q for q, _ in queries]
    q_mat = to_csr(model.encode_query([t for _, t in queries], batch_size=args.batch_size,
                                      convert_to_sparse_tensor=True))

    scores, idx = search(q_mat, para_mat, k=args.para_pool)
    for level, run in (("paragraph", paragraph_run(qids, scores, idx, para_ids, args.top_k)),
                       ("paper", paper_run(qids, scores, idx, para_ids, args.top_k))):
        name = f"{TEAM}_{level}_{args.run_tag}_{args.split}"
        write_run(run, RUN_DIR / f"{name}.tsv", name, args.top_k)
        print(f"Wrote {RUN_DIR / name}.tsv")


if __name__ == "__main__":
    main()
