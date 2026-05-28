# MRL for Dense Text Retrieval: A Cross-Domain Comparison

We study whether Matryoshka
Representation Learning (Kusupati et al., NeurIPS 2022) can produce
compression-aware dense retrieval embeddings through lightweight fine-tuning
of a compact base encoder, rather than from-scratch pretraining. We fine-tune
`bge-base-en-v1.5` twice from identical initialization, data, and
hyperparameters, changing only the loss (`MultipleNegativesRankingLoss` vs.
that same loss wrapped in `MatryoshkaLoss`). The resulting models are
compared against naive truncation, post-hoc PCA, and the much larger
`nomic-embed-text-v1.5` across five BEIR datasets and six embedding
dimensions from 768 down to 32.

## Findings

1. **Standard dense encoders are not naturally prefix-composable.** Under
   naive truncation, the MNRL fine-tune drops from 0.4455 nDCG@10 at d=768
   to 0.1495 at d=32, a 66% loss. Without explicit MRL supervision the
   encoder has no incentive to concentrate retrieval-relevant information
   into the early coordinates.
2. **MRL beats post-hoc PCA at every evaluated dim**, with the largest
   gains at low dimensions: +0.0399 nDCG@10 at d=64 and +0.0301 at d=32.
   At moderate compression (d ≥ 128), PCA on a standard fine-tune is a
   surprisingly strong baseline, retaining ~89% of full-dim nDCG@10 at
   d=128. The MRL training signal becomes practically necessary only at
   d ≤ 64.
3. **Multi-scale MRL training also improves full-dim performance.** At
   d=768 the MRL encoder reaches 0.4612 nDCG@10 versus 0.4455 for the
   plain MNRL fine-tune (+0.016), consistent with a positive regularization
   effect from supervising six nested prefixes simultaneously.
4. **Compression behavior is heavily domain-dependent.** At d=32 the MRL
   fine-tune retains 66% of its full-dim nDCG@10 on ArguAna and 63% on
   SciFact, but only 28% on FiQA (financial) and 44% on TREC-COVID
   (biomedical). Practitioners cannot assume uniform behavior across
   domains.
5. **A small MRL fine-tune is competitive with a larger pretrained MRL
   model on resilient domains.** At d=32, the MRL fine-tune of bge-base
   (110M params) outperforms `nomic-embed-text-v1.5` on SciFact (0.4077
   vs. 0.3577), NFCorpus (0.1888 vs. 0.1671), and ArguAna (0.2581 vs.
   0.1963). Nomic still wins on FiQA and TREC-COVID, where its larger
   pretraining footprint matters.

## Headline numbers

Average nDCG@10 across the five BEIR datasets (SciFact, NFCorpus, ArguAna,
FiQA, TREC-COVID). The best score at each dimension is in **bold**.

| Dim | MNRL + truncate | MNRL + PCA | MRL + truncate (ours) | Nomic Embed v1.5 |
|---|---|---|---|---|
| 768 | 0.4455 | 0.4455 | 0.4612 | **0.5214** |
| 512 | 0.4227 | 0.4415 | 0.4463 | **0.5160** |
| 256 | 0.4069 | 0.4271 | 0.4315 | **0.5000** |
| 128 | 0.3395 | 0.3874 | 0.3960 | **0.4684** |
|  64 | 0.2511 | 0.3069 | 0.3467 | **0.4025** |
|  32 | 0.1495 | 0.2121 | 0.2422 | **0.2657** |

Direct MRL-versus-PCA comparison, same five datasets:

| Dim | MNRL + PCA | MRL + truncate | Δ (MRL − PCA) |
|---|---|---|---|
| 768 | 0.4455 | 0.4612 | +0.0156 |
| 512 | 0.4415 | 0.4463 | +0.0049 |
| 256 | 0.4271 | 0.4315 | +0.0043 |
| 128 | 0.3874 | 0.3960 | +0.0086 |
|  64 | 0.3069 | 0.3467 | **+0.0399** |
|  32 | 0.2121 | 0.2422 | **+0.0301** |

The full per-dataset breakdown lives in `outputs/results/eval_results.csv`
(120 rows). Generated tables and figures live in `outputs/tables/` and
`outputs/figures/`.

## Methodology

We fine-tune `bge-base-en-v1.5` under two conditions (MNRL vs.
MRL-wrapped MNRL) from identical initialization, data, and
hyperparameters, then evaluate both against PCA-reduced and nomic-embed
baselines across six embedding dimensions (768→32) on five BEIR domains.

The two fine-tune runs differ only in the loss. **Run A** uses
`MultipleNegativesRankingLoss` directly on the full 768-dim embedding.
**Run B** wraps that same loss in `MatryoshkaLoss` with nested dims
`[768, 512, 256, 128, 64]`, training the encoder to keep each prefix
useful for retrieval on its own. Both runs use 200K MS MARCO triplets,
batch size 32, learning rate 2e-5, 1 epoch, seed 42, and the same data
shuffle. The PCA baseline is fit on 50K unique positive passages encoded
by Run A, giving a fair non-MRL compression alternative trained from the
same data distribution. Evaluation extends one step below the trained
schedule to d=32 as a stress test of graceful degradation; nothing is
trained at d=32, so any survivor there is exhibiting structural
robustness rather than training-set memorization.

See the paper for the formal MRL objective, the full notation, and the
discussion of why prefix supervision differs from post-hoc projection.

## System requirements

- Python 3.10+
- CUDA GPU with at least 16 GB VRAM (tested on RTX 4070 Ti Super)
- ~5 GB free disk for models, datasets, and intermediate caches
- ~80 minutes wall-clock for the full pipeline

## Setup

```powershell
# 1. Create and activate a virtual env (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download all models and datasets (~1.9 GB, ~15-25 min)
python 00_download_models_datasets.py
```

The download script writes everything to `./models/` and `./datasets/`
next to the script. Nothing lands in the user's HuggingFace cache or any
hidden directory.

## Reproducing the results

### One-shot

```powershell
.\run_all.ps1
```

Runs phases 1 through 6 sequentially, halting on the first failure.
Total wall-clock ~80 minutes on the hardware listed above.

### Phase by phase

| Step | Script | Output | Time |
|---|---|---|---|
| 1 | `01_train_baseline.py` | `checkpoints/bge-runA/` (Run A, MNRL fine-tune) | ~18 min |
| 2 | `02_train_mrl.py` | `checkpoints/bge-runB/` (Run B, MRL fine-tune) | ~18 min |
| 3 | `03_fit_pca.py` | `outputs/pca_models/pca_*.joblib` (5 files) | ~1 min |
| 4 | `04_evaluate.py` | `outputs/results/eval_results.csv` (120 rows) | ~35 min |
| 5 | `05_make_tables.py` | `outputs/tables/*.tex` (3 booktabs tables) | ~5 sec |
| 6 | `06_make_plots.py` | `outputs/figures/*.pdf` (3 PDFs) | ~10 sec |

Phase 4 is the longest because it encodes five BEIR corpora with three
different encoders (Run A, Run B, Nomic). Results checkpoint after each
dataset, so a crash mid-run still leaves all completed datasets in the
CSV. The embedding cache also survives crashes; re-running picks up
where it stopped.

## Project structure

```
mrl-dense-retrieval/
├── configs/                         Hyperparameters, no code-level values
│   ├── base.yaml                    Shared defaults; everything inherits this
│   ├── train_baseline.yaml          Run A overrides (loss=mnrl)
│   ├── train_mrl.yaml               Run B overrides (loss=mrl)
│   └── eval.yaml                    Eval matrix: variants × dims × datasets
├── lib/                             Importable library code
│   ├── beir_loader.py               Load BEIR corpus/queries/qrels from disk
│   ├── encoder.py                   SentenceTransformer wrapper with prefix handling
│   ├── transforms.py                MRL truncation + PCA projection
│   ├── metrics.py                   nDCG@k, Recall@k, MAP@k
│   ├── training.py                  Phase 1 + 2 shared training pipeline
│   ├── pca_fitting.py               Phase 3 helpers
│   ├── evaluation.py                Phase 4 cache + transform-builder
│   ├── analysis.py                  Phase 5 + 6 table + plot writers
│   ├── config.py                    YAML deep-merge loader
│   └── logging_utils.py             Console + file logging
├── 00_download_models_datasets.py   Phase 0: setup
├── 01_train_baseline.py             Phase 1: Run A (MNRL only)
├── 02_train_mrl.py                  Phase 2: Run B (MRL-wrapped)
├── 03_fit_pca.py                    Phase 3: fit PCA on Run A
├── 04_evaluate.py                   Phase 4: 120-cell eval matrix
├── 05_make_tables.py                Phase 5: LaTeX tables
├── 06_make_plots.py                 Phase 6: PDF figures
├── Appendix_Plot.ipynb              Independent notebooks for adding table and plot
├── run_all.ps1                      Sequential runner
├── requirements.txt
└── .gitignore
```

After running, the following directories are also populated locally
(all gitignored):

```
models/                  3 base models (~1.4 GB)
datasets/                BEIR + MS MARCO triplets (~470 MB)
checkpoints/             bge-runA, bge-runB
outputs/embeddings_cache/  full-dim encoder outputs, per (encoder, dataset)
outputs/pca_models/      fitted PCA projection matrices
outputs/logs/            per-phase logs
```

## Configuration

Every hyperparameter lives in `configs/`. Phase scripts load `base.yaml`
first, then layer a phase-specific override file on top. Changing a
learning rate or batch size never requires touching Python code.

For example, to try a different MRL dimension schedule, edit
`configs/base.yaml`:

```yaml
mrl:
  dims: [768, 384, 96, 24]
  weights: [1, 1, 1, 1]
```

Then re-run Phase 2 (Run B fine-tune) and Phase 4 (eval). Phases 5 and 6
regenerate from the new CSV automatically.

## Limitations

- **Single base model.** Run A and Run B both start from
  `bge-base-en-v1.5`. We did not test whether the findings hold for
  other encoder backbones.
- **MS MARCO triplets only.** Training data is 200K passages sampled
  from the `triplet` subset of
  `sentence-transformers/msmarco-msmarco-MiniLM-L6-v3`. This is
  in-domain web QA, so absolute numbers on out-of-domain BEIR tasks
  (NFCorpus, FiQA, TREC-COVID) are below state-of-the-art. The Run A
  vs. Run B comparison is internally consistent.
- **Five BEIR tasks.** SciFact, NFCorpus, ArguAna, FiQA-2018, and
  TREC-COVID were selected for diversity across short, medical,
  argumentative, financial, and scientific retrieval. Adding more tasks
  would strengthen the cross-domain finding but did not fit the
  project timeline.
- **Single epoch fine-tuning.** Both runs use 1 epoch at lr 2e-5.
  Multi-epoch runs may shift absolute numbers but should not change the
  Run A vs. Run B comparison.
- **Nomic mode for truncation.** The Nomic variant applies the
  published `layer_norm → truncate → L2-normalize` recipe; the BGE
  variants use plain `truncate → L2-normalize`. This matches each
  model's published guidance and is the fair per-model comparison.
- **MRL trained to d=64; d=32 is extrapolation.** Run B and Nomic are
  both trained with nested dims down to 64. Numbers at d=32 are
  reported as a stress test of graceful degradation off the trained
  schedule, not as a directly supervised operating point.

## Authors

- Julian Alex Joshua (2206082606)
- Lucinda Laurent (2206024745)
- Nasywa Kamila Az Zahra (2206083060)
- Fikri Risyad Indratno (2206031170)

## License

MIT.

## References

- Kusupati et al. [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). NeurIPS 2022.
- Xiao et al. [C-Pack: Packed Resources For General Chinese Embeddings](https://arxiv.org/abs/2309.07597). SIGIR 2024. (Official BGE framework)
- Nussbaum et al. [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613). 2024.
- Thakur et al. [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663). NeurIPS 2021 Datasets & Benchmarks.