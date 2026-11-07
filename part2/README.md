# Part 2: Neural Retrieval (marcelsSearchParty)

One **dense** model (SciNCL, fine-tuned on the Train queries) and one **sparse neural** model (SPLADE), compared with Part 1's BM25 on the Study queries at paper and paragraph level. Every script runs from the repository root with `python -m part2.<script>` and takes its settings as command-line options with defaults (`--help` lists them).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
# USM CS GPU servers (NVIDIA driver = CUDA 12.6): install this torch build FIRST
python -m pip install "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r part2/requirements.txt
python -m unittest discover -s part2/tests -t .        # 28 tests
```

On the CS servers, keep Hugging Face downloads on netstore:
`export HF_HOME=/mnt/netstore1_home/<user>/hf_cache` and `export HF_HUB_CACHE=$HF_HOME/hub`.

Data (not in git) goes in `data/`: `JSON Files/`, `Train.json`, `Study.json`, `Test.json`, `qrels/`, and Part 1's BM25 Study runs in `part1_runs/`.

## Pipeline

| Step | Command | Output | GPU time (ours) |
|---|---|---|---|
| 1. Dense, pretrained (baseline) | `python -m part2.dense_retrieve --split study --run_tag densePretrained` (then `--split test`) | `data/part2_runs/*_densePretrained_*.tsv` | about 3.5 h to embed 2.28M paragraphs, cached after |
| 2. Fine-tune dense on Train | `python -m part2.dense_finetune` | `data/part2_models/dense_ft/` + `loss_chart.png`, `loss_history.csv` | about 20 min |
| 3. Dense, fine-tuned | `python -m part2.dense_retrieve --model data/part2_models/dense_ft --split study` (then `--split test`) | `*_dense_*.tsv` | about 3.5 h, cached after |
| 4. Sparse (SPLADE) | `python -m part2.sparse_retrieve --split study` (then `--split test`) | `*_sparse_*.tsv` | several hours, cached after |
| 5. Evaluate on Study | `python -m part2.evaluate_part2` | `data/part2_eval/results_*.md/.json`, `per_query_*.csv` | seconds, CPU |
| 6. MRR by difficulty | `python -m part2.difficulty_chart` | `data/part2_eval/mrr_by_difficulty_*.png/.csv` | seconds, CPU |
| 7. Manual Test grades | `python -m part2.annotate` | `data/part2_annotation/marcelsSearchParty_qrel_part2_test.tsv` | by hand |
| 8. Build submission folder | `python -m part2.package_results` | `submission/Retrieval Results/` (checked) | seconds |

Quick tests on a slice of the corpus: add `--limit 50000` to steps 1, 3 and 4, or `--max_train_queries 50 --epochs 1` to step 2.

## Files

| File | What it does |
|---|---|
| `data.py` | Loads paragraphs (light cleaning: citation markers and URLs removed, stopwords kept), queries and qrels; 90/10 Train split; writes tab-separated TREC runs |
| `dense_retrieve.py` | Embeds paragraphs and queries, cosine top-k in chunks, paragraph run + paper run (paper score = best paragraph) |
| `dense_finetune.py` | MultipleNegativesRankingLoss on (Train query, relevant paragraph) pairs; validation loss and nDCG@10 per epoch; keeps the best epoch; refuses non-Train query ids |
| `sparse_retrieve.py` | SPLADE vectors stored as a sparse matrix, dot-product top-k; `--show "text"` prints the learned word weights |
| `evaluate_part2.py` | nDCG, nDCG@5, MRR, P@5, P@10 (any) + MRR, P@5 (full); paired t-test between every pair of models incl. Part 1 BM25 |
| `difficulty_chart.py` | MRR per Easy / Medium / Hard per model |
| `annotate.py` | Terminal tool to grade the best model's rank-1 Test paragraph (2 / 1 / 0); saves after every grade |
| `package_results.py` | Copies the 8 required runs + qrel into `submission/Retrieval Results/` and checks names and format |

## Key settings and why

- **Models:** `malteos/scincl` (trained on scientific papers via citation links) and `naver/splade-cocondenser-ensembledistil`.
- **512 tokens max:** paragraphs are 77 words at the median and about 10% are over 191 words.
- **Top 3,000 paragraphs before paper ranking:** enough distinct papers to fill the top 100 for every query.
- **Fine-tuning:** 1,800 / 200 Train queries (seed 42), 3 epochs, batch 32, learning rate 2e-5, no duplicate paragraphs in a batch, best epoch by validation loss. Study and Test are never used for training.
- **SPLADE:** at most 256 non-zero words per paragraph, to fit the full corpus in memory.
