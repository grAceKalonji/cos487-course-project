"""Fine-tune the dense model on the Train queries (never Study or Test).

Spec: "Models must be fine-tuned only on the training queries. The study set and
test set must not be used for any training purposes. In your report, include a
loss diagram per epoch/step for both the training and validation sets."

What happens:
  1. Train.json (2,000 queries) is split 90/10 into train / validation queries.
  2. Each (query, relevant paragraph) pair from Train_paragraph_qrel.tsv is one
     training example.
  3. Loss = MultipleNegativesRankingLoss: in every batch, each query has to pick
     its own paragraph out of all the paragraphs in the batch. The other
     paragraphs act as free "wrong answers", so bigger batches = harder practice.
  4. Training loss is logged every few steps, validation loss + validation
     nDCG@10 at the end of every epoch. The best epoch (lowest validation loss)
     is kept.
  5. Saves the model, a loss chart (PNG) and the raw numbers (CSV).

Run from the repo root on the GPU server (inside tmux):
    python -m part2.dense_finetune
    python -m part2.dense_finetune --max_train_queries 50 --epochs 1   # quick test

Then search with the fine-tuned copy:
    python -m part2.dense_retrieve --model data/part2_models/dense_ft --split study
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from part2.data import DATA_DIR, clean_text, load_qrels, load_queries, train_val_split

MODEL_DIR = DATA_DIR / "part2_models"
LAST_TRAIN_QUERY_ID = 2000          # Train ids are 1-2000; Study starts at 2001


def paper_of(para_id):
    return para_id.rsplit("_", 1)[0]


def load_paragraph_texts(para_ids, extra_papers=0, seed=42, json_dir=None):
    """Return {paragraph_id: text} for the given ids.

    Only opens the papers we need (fast). extra_papers = also read this many
    random other papers; all their paragraphs become distractors for validation.
    """
    json_dir = Path(json_dir or DATA_DIR / "JSON Files")
    want = set(para_ids)
    needed = {paper_of(p) for p in want}
    others = sorted(p.stem for p in json_dir.glob("*.json") if p.stem not in needed)
    random.Random(seed).shuffle(others)
    distractors = set(others[:extra_papers])

    texts = {}
    for pid in sorted(needed | distractors):
        path = json_dir / f"{pid}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            paper = json.load(f)
        for para_id, text in paper["paragraphs"].items():
            text = clean_text(text)
            if text and (para_id in want or pid in distractors):
                texts[para_id] = text
    return texts


def build_pairs(query_ids, queries, qrels, para_texts):
    """(query text, relevant paragraph text) pairs - one training example each."""
    pairs = []
    for qid in query_ids:
        for para_id in sorted(qrels.get(qid, {})):
            if para_id in para_texts:
                pairs.append((queries[qid], para_texts[para_id]))
    return pairs


def check_train_only(query_ids):
    """Safety check: refuse to train if any Study/Test query id sneaks in."""
    bad = [q for q in query_ids if int(q) > LAST_TRAIN_QUERY_ID]
    if bad:
        raise ValueError(f"Non-Train query ids in training data: {bad[:5]}")


def loss_rows(log_history):
    """Turn the trainer's log into rows: step, epoch, train_loss, val_loss, val_ndcg@10."""
    rows = []
    for entry in log_history:
        if "loss" in entry or "eval_loss" in entry:
            ndcg = next((v for k, v in entry.items() if k.endswith("ndcg@10")), "")
            rows.append({"step": entry.get("step"), "epoch": round(entry.get("epoch", 0), 3),
                         "train_loss": entry.get("loss", ""), "val_loss": entry.get("eval_loss", ""),
                         "val_ndcg@10": ndcg})
    return rows


def save_loss_chart(rows, out_dir):
    """Write loss_history.csv and loss_chart.png (train loss per step, val loss per epoch)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    with open(out_dir / "loss_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "val_loss", "val_ndcg@10"])
        w.writeheader()
        w.writerows(rows)

    tr = [(r["step"], r["train_loss"]) for r in rows if r["train_loss"] != ""]
    va = [(r["step"], r["val_loss"]) for r in rows if r["val_loss"] != ""]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if tr:
        ax.plot(*zip(*tr), label="Training loss", color="#378ADD")
    if va:
        ax.plot(*zip(*va), "o-", label="Validation loss", color="#D85A30")
    ax.set_xlabel("Training step")
    ax.set_ylabel("MultipleNegativesRankingLoss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_chart.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="malteos/scincl")
    p.add_argument("--out", default=str(MODEL_DIR / "dense_ft"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32, help="bigger = more in-batch wrong answers")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_distractor_papers", type=int, default=300,
                   help="random papers added to the validation corpus so nDCG@10 isn't trivially easy")
    p.add_argument("--max_train_queries", type=int, default=0, help="only use N train queries (testing)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- data: Train only ----
    queries = dict(load_queries("train"))
    qrels = load_qrels("Train", "paragraph")
    train_ids, val_ids = train_val_split([q for q in queries if q in qrels], args.val_ratio, args.seed)
    if args.max_train_queries:
        train_ids = train_ids[:args.max_train_queries]
        val_ids = val_ids[:max(5, args.max_train_queries // 9)]
    check_train_only(train_ids + val_ids)

    needed = {d for q in train_ids + val_ids for d in qrels[q]}
    para_texts = load_paragraph_texts(needed, extra_papers=args.val_distractor_papers, seed=args.seed)
    train_pairs = build_pairs(train_ids, queries, qrels, para_texts)
    val_pairs = build_pairs(val_ids, queries, qrels, para_texts)
    print(f"Train queries {len(train_ids)} -> {len(train_pairs)} pairs | "
          f"Validation queries {len(val_ids)} -> {len(val_pairs)} pairs")

    train_ds = Dataset.from_dict({"anchor": [a for a, _ in train_pairs], "positive": [b for _, b in train_pairs]})
    val_ds = Dataset.from_dict({"anchor": [a for a, _ in val_pairs], "positive": [b for _, b in val_pairs]})

    # validation search: val queries vs their relevant paragraphs + distractor paragraphs
    val_corpus = {d: t for d, t in para_texts.items()
                  if d not in {x for q in train_ids for x in qrels[q]}}
    evaluator = InformationRetrievalEvaluator(
        queries={q: queries[q] for q in val_ids},
        corpus=val_corpus,
        relevant_docs={q: set(qrels[q]) & set(val_corpus) for q in val_ids},
        ndcg_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1, 10], precision_recall_at_k=[10],
        map_at_k=[100], name="val", show_progress_bar=False,
    )
    print(f"Validation corpus: {len(val_corpus):,} paragraphs")

    # ---- model + loss ----
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length
    loss = losses.MultipleNegativesRankingLoss(model)

    print("Validation BEFORE fine-tuning:", evaluator(model))

    targs = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out) / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=(device == "cuda"),
        batch_sampler=BatchSamplers.NO_DUPLICATES,   # same paragraph twice in a batch = fake "wrong answer"
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(1, len(train_pairs) // args.batch_size // 20),   # ~20 points per epoch
        seed=args.seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=train_ds,
                                         eval_dataset=val_ds, loss=loss, evaluator=evaluator)
    trainer.train()

    print("Validation AFTER fine-tuning (best epoch):", evaluator(model))
    out = Path(args.out)
    model.save(str(out))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)   # ~1.3 GB of optimizer state we no longer need
    save_loss_chart(loss_rows(trainer.state.log_history), out)
    print(f"Saved model, loss_chart.png and loss_history.csv to {out}")


if __name__ == "__main__":
    main()
