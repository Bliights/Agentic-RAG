import os
from pathlib import Path

from datasets import Dataset, load_dataset


def save_image(sample: dict, image_dir: Path) -> dict:
    """
    Save the image from a dataset sample to disk and add its file path

    Parameters
    ----------
    sample : dict
        Dataset sample
    image_dir : Path
        Directory where images will be stored

    Returns
    -------
    dict
        Updated sample with an added image_path field
    """
    os.makedirs(image_dir, exist_ok=True)

    path = image_dir / f"{sample['corpus_id']}.png"

    if not os.path.exists(path):
        sample["image"].save(path)
    sample["image_path"] = str(path)
    return sample


def load_vidore_dataset(image_dir: Path) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load the ViDoRe dataset and persist corpus images to disk

    Parameters
    ----------
    image_dir : Path
        Directory where corpus images will be stored

    Returns
    -------
    tuple[Dataset, Dataset, Dataset]
        corpus dataset, queries dataset, qrels dataset (relevance annotations)
    """
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


def dataset_join(
    corpus: Dataset,
    queries: Dataset,
    qrels: Dataset,
) -> tuple[dict, dict, dict]:
    """
    Convert dataset splits into dictionary-based structures for fast access

    Parameters
    ----------
    corpus : Dataset
        Dataset containing corpus items
    queries : Dataset
        Dataset containing query items
    qrels : Dataset
        Dataset containing relevance annotations (query-document pairs)

    Returns
    -------
    tuple[dict, dict, dict]
        A tuple containing:
        - corpus_dict: mapping from corpus_id to corpus item
        - query_dict: mapping from query_id to query item
        - labels: mapping from query_id to dictionaries of corpus_id to relevance score
    """
    corpus_dict = {item["corpus_id"]: item for item in corpus}
    query_dict = {item["query_id"]: item for item in queries}

    labels = {}
    for item in qrels:
        qid = item["query_id"]
        doc_id = item["corpus_id"]
        relevance = item["score"]

        if qid not in labels:
            labels[qid] = {}

        labels[qid][doc_id] = relevance

    return corpus_dict, query_dict, labels
