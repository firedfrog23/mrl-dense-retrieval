"""Load BEIR datasets from local parquet+tsv files.

Expects the layout produced by `download_all.py`:

    datasets/beir/{name}/corpus/corpus-00000-of-00001.parquet
    datasets/beir/{name}/queries/queries-00000-of-00001.parquet
    datasets/beir/{name}-qrels/{split}.tsv
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BeirDataset:
    """A loaded BEIR dataset, ready for retrieval evaluation."""

    name: str
    corpus: pd.DataFrame   # columns: _id, title (optional), text
    queries: pd.DataFrame  # columns: _id, text
    qrels: pd.DataFrame    # columns: query_id, corpus_id, score

    @property
    def corpus_text(self) -> list[str]:
        """Concatenated title + text per document. Falls back to text only."""
        if "title" in self.corpus.columns:
            title = self.corpus["title"].fillna("")
            text = self.corpus["text"].fillna("")
            return (title + " " + text).str.strip().tolist()
        return self.corpus["text"].fillna("").tolist()

    @property
    def query_text(self) -> list[str]:
        return self.queries["text"].fillna("").tolist()

    @property
    def corpus_ids(self) -> list[str]:
        return self.corpus["_id"].tolist()

    @property
    def query_ids(self) -> list[str]:
        return self.queries["_id"].tolist()

    def __repr__(self) -> str:
        return (
            f"BeirDataset(name={self.name!r}, "
            f"corpus={len(self.corpus)}, "
            f"queries={len(self.queries)}, "
            f"qrels={len(self.qrels)})"
        )


def load_beir(
    name: str,
    datasets_dir: str | Path,
    qrels_split: str = "test",
) -> BeirDataset:
    """Load a BEIR dataset from local disk.

    Filters queries down to those that have judgments in the requested split.
    Ids are normalized to strings to avoid int/str join mismatches.

    Parameters
    ----------
    name : str
        Folder name under datasets/beir/. e.g. "nfcorpus", "scifact".
    datasets_dir : str | Path
        Project-relative or absolute path to the datasets folder.
    qrels_split : str
        Which split file to read from the -qrels repo. Defaults to "test".
    """
    base = Path(datasets_dir) / "beir"

    corpus = pd.read_parquet(base / name / "corpus" / "corpus-00000-of-00001.parquet")
    queries = pd.read_parquet(base / name / "queries" / "queries-00000-of-00001.parquet")
    qrels = pd.read_csv(
        base / f"{name}-qrels" / f"{qrels_split}.tsv",
        sep="\t", header=0,
        names=["query_id", "corpus_id", "score"],
        dtype={"query_id": str, "corpus_id": str, "score": int},
    )

    corpus["_id"] = corpus["_id"].astype(str)
    queries["_id"] = queries["_id"].astype(str)

    qids_with_judgments = set(qrels["query_id"])
    queries = queries[queries["_id"].isin(qids_with_judgments)].reset_index(drop=True)

    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)
