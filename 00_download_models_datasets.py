"""
Download all models and datasets for the MRL retrieval study.
Everything lands in ./models/ and ./datasets/ next to this script.
"""
import os
import shutil
from pathlib import Path

# Pin every cache to a local visible folder BEFORE importing HF libs
PROJECT_ROOT = Path(__file__).parent.resolve()
LOCAL_HF_CACHE = PROJECT_ROOT / "_hf_cache_tmp"
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["HF_DATASETS_CACHE"] = str(LOCAL_HF_CACHE / "datasets")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download
from datasets import load_dataset

MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

# Skip bloat formats. Keep only safetensors weights and the configs PyTorch needs.
MODEL_IGNORE = [
    "*.onnx", "*.onnx_data", "onnx/*", "openvino/*",
    "*.h5", "tf_model.h5",
    "flax_model.msgpack", "rust_model.ot",
    "pytorch_model.bin",  # we keep safetensors instead
    "*.gguf", "ggml-*",
]

MODELS = [
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
]

# BEIR eval data. Each task lives in two repos: corpus+queries, and qrels.
BEIR_TASKS = ["scifact", "nfcorpus", "arguana", "fiqa", "trec-covid"]


def folder_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def download_models():
    for repo_id in MODELS:
        local = MODELS_DIR / repo_id.split("/")[-1]
        print(f"\n[model] {repo_id} -> {local}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local),
            ignore_patterns=MODEL_IGNORE,
        )
        print(f"  size: {folder_size_mb(local):.1f} MB")


def download_msmarco_triplets():
    """503K-row triplet subset, then save a 200K-row parquet for training."""
    print("\n[data] sentence-transformers/msmarco-msmarco-MiniLM-L6-v3 (triplet, 503K rows)")
    ds = load_dataset(
        "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
        "triplet",
        split="train",
    )
    print(f"  loaded {len(ds):,} rows. Subsampling to 200K and saving parquet.")
    out = DATASETS_DIR / "msmarco_triplets_200k.parquet"
    ds.shuffle(seed=42).select(range(200_000)).to_parquet(str(out))
    print(f"  saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")


def download_beir():
    for task in BEIR_TASKS:
        for suffix in ["", "-qrels"]:
            repo_id = f"BeIR/{task}{suffix}"
            local = DATASETS_DIR / "beir" / f"{task}{suffix}"
            print(f"\n[data] {repo_id} -> {local}")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local),
            )
            print(f"  size: {folder_size_mb(local):.1f} MB")


def cleanup_temp_cache():
    if LOCAL_HF_CACHE.exists():
        print(f"\n[cleanup] removing temp HF cache: {LOCAL_HF_CACHE}")
        shutil.rmtree(LOCAL_HF_CACHE, ignore_errors=True)


def report_totals():
    models_mb = folder_size_mb(MODELS_DIR)
    data_mb = folder_size_mb(DATASETS_DIR)
    print("\n=== totals ===")
    print(f"  ./models   {models_mb:>9.1f} MB  ({models_mb/1024:.2f} GB)")
    print(f"  ./datasets {data_mb:>9.1f} MB  ({data_mb/1024:.2f} GB)")
    print(f"  combined   {models_mb + data_mb:>9.1f} MB  ({(models_mb + data_mb)/1024:.2f} GB)")


if __name__ == "__main__":
    download_models()
    download_msmarco_triplets()
    download_beir()
    cleanup_temp_cache()
    report_totals()
    print("\nDone. Models in ./models, datasets in ./datasets.")
