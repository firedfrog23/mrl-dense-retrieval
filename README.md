# MRL for Dense Text Retrieval: A Cross-Domain Comparison

## Methodology

We fine-tune `bge-base-en-v1.5` under two conditions (MNRL vs. MRL-wrapped MNRL) from identical initialization, data, and hyperparameters, then evaluate both against PCA-reduced and nomic-embed baselines across six embedding dimensions (768→32) on five BEIR domains.

## References

- Kusupati et al. [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). NeurIPS 2022.
- Xiao et al. [C-Pack: Packed Resources For General Chinese Embeddings](https://arxiv.org/abs/2309.07597). SIGIR 2024. (Official BGE framework)
- Nussbaum et al. [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613). 2024.
- Thakur et al. [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663). NeurIPS 2021 Datasets & Benchmarks.
