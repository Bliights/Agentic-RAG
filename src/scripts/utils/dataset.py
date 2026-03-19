import os
from pathlib import Path

from datasets import Dataset, load_dataset


def save_image(sample: dict, image_dir: Path) -> dict:
    os.makedirs(image_dir, exist_ok=True)

    path = image_dir / f"{sample['corpus_id']}.png"

    if not os.path.exists(path):
        sample["image"].save(path)
    sample["image_path"] = str(path)
    return sample


def load_vidore_dataset(image_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    dataset_name = "vidore/vidore_v3_pharmaceuticals"
    corpus = load_dataset(dataset_name, "corpus", split="test")
    queries = load_dataset(dataset_name, "queries", split="test")
    qrels = load_dataset(dataset_name, "qrels", split="test")

    corpus = corpus.map(
        lambda sample: save_image(sample, image_dir),
        num_proc=4,
        desc="Saving images",
    )
    return corpus, queries, qrels
