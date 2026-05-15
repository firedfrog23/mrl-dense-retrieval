# MRL for Dense Text Retrieval: A Cross-Domain Comparison

## Methodology

We fine-tune `bge-base-en-v1.5` under two conditions (MNRL vs. MRL-wrapped MNRL) from identical initialization, data, and hyperparameters, then evaluate both against PCA-reduced and nomic-embed baselines across six embedding dimensions (768→32) on five BEIR domains.

## References

- Kusupati et al. *Matryoshka Representation Learning.* NeurIPS 2022.
- Zhang et al. *BGE: BAAI General Embedding.* 2023.
- Nussbaum et al. *Nomic Embed: Training a Reproducible Long Context Text Embedder.* 2024.
- Thakur et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS 2021 Datasets & Benchmarks.
