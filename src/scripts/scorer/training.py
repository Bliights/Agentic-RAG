import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from agentic_rag.retriever.core.types import TextualResult, VisualResult
from agentic_rag.retriever.textual.retriever import TextualRetriever
from agentic_rag.retriever.visual.retriever import VisualRetriever
from agentic_rag.scorer.dataset import ScorerDataset
from agentic_rag.scorer.model import ScorerModel
from agentic_rag.vectordb.handler import QdrantHandler
from scripts.retriever.textual.config import DATABASE_NAME as TEXTUAL_DATABASE_NAME
from scripts.retriever.textual.config import EMBEDDER_MODEL_NAME as TEXTUAL_EMBEDDER
from scripts.retriever.visual.config import DATABASE_NAME as VISUAL_DATABASE_NAME
from scripts.retriever.visual.config import EMBEDDER_MODEL_NAME as VISUAL_EMBEDDER
from scripts.scorer.config import (
    BATCH_SIZE,
    DATA_FOLDER,
    DB_HOST,
    DB_PORT,
    EPOCHS,
    FEATURE_CACHE_PATH,
    HIDDEN_DIM,
    METRICS_NB_DOC,
    MODEL_PATH,
    NB_DOCS_RETRIEVAL,
    TEST_RATIO,
    TOTAL_MAX_DOC_RETRIEVAL,
    TRANSFORMER_NAME,
    VAL_RATIO,
)
from scripts.utils.cache import CacheManager
from scripts.utils.dataset import dataset_join, load_vidore_dataset
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def train_validation_test_split(
    query_dict: dict,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split query identifiers into training, validation, and test subsets

    Parameters
    ----------
    query_dict : dict
        Mapping from query IDs to query metadata
    val_ratio : float
        Proportion of the non-test data assigned to the validation split
    test_ratio : float
        Proportion of the full dataset assigned to the test split

    Returns
    -------
    tuple[list[int], list[int], list[int]]
        Training, validation, and test IDS
    """
    query_ids = list(query_dict.keys())

    total = len(query_ids)
    test_size = int(total * test_ratio)
    val_size = int((total - test_size) * val_ratio)

    train_end = total - test_size - val_size
    val_end = total - test_size

    train_ids = query_ids[:train_end]
    val_ids = query_ids[train_end:val_end]
    test_ids = query_ids[val_end:]

    return train_ids, val_ids, test_ids


def build_doc_maps(textual_res: TextualResult, visual_res: VisualResult) -> tuple[dict, dict, list]:
    """
    Aggregate retrieval results into corpus-level score maps

    Parameters
    ----------
    textual_res : TextualResult
        Textual retrieval results for a query
    visual_res : VisualResult
        Visual retrieval results for a query

    Returns
    -------
    tuple[dict, dict, list]
        A tuple containing:
        - a mapping from corpus ID to textual score
        - a mapping from corpus ID to visual score
        - the sorted list of all corpus IDs appearing in either map
    """
    textual_map = {}
    for r in textual_res:
        if r.corpus_id not in textual_map:
            textual_map[r.corpus_id] = r.score
        else:
            textual_map[r.corpus_id] = max(textual_map[r.corpus_id], r.score)

    visual_map = {}
    for r in visual_res:
        if r.corpus_id not in visual_map:
            visual_map[r.corpus_id] = r.score
        else:
            visual_map[r.corpus_id] = max(visual_map[r.corpus_id], r.score)

    doc_ids = sorted(set(textual_map.keys()) | set(visual_map.keys()))
    return textual_map, visual_map, doc_ids


def build_score_vectors(
    doc_ids: list[str],
    textual_map: dict,
    visual_map: dict,
    labels: dict,
    query_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build aligned score and label tensors for a single query

    Parameters
    ----------
    doc_ids : list[str]
        Ordered list of corpus IDS
    textual_map : dict
        Mapping from corpus IDS to textual retrieval score
    visual_map : dict
        Mapping from corpus IDS to visual retrieval score
    labels : dict
        Mapping from query IDs, corpus IDS to relevance labels
    query_id : int
        Identifier of the query

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tensors containing textual scores, visual scores, and relevance labels
    """
    textual_scores, visual_scores, q_labels = [], [], []

    for doc_id in doc_ids:
        textual_scores.append(textual_map.get(doc_id, 0.0))
        visual_scores.append(visual_map.get(doc_id, 0.0))
        q_labels.append(labels.get(query_id, {}).get(doc_id, 0.0))

    return (
        torch.tensor(textual_scores, dtype=torch.float32),
        torch.tensor(visual_scores, dtype=torch.float32),
        torch.tensor(q_labels, dtype=torch.float32),
    )


def normalize_scores(
    textual_scores: torch.Tensor,
    visual_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize textual and visual score vectors independently

    Parameters
    ----------
    textual_scores : torch.Tensor
        Tensor of textual retrieval scores
    visual_scores : torch.Tensor
        Tensor of visual retrieval scores

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Normalized textual and visual score tensors
    """
    if len(textual_scores) > 1:
        textual_scores = (textual_scores - textual_scores.mean()) / (textual_scores.std() + 1e-6)
        visual_scores = (visual_scores - visual_scores.mean()) / (visual_scores.std() + 1e-6)

    return textual_scores, visual_scores


def pad_or_truncate(
    textual_scores: torch.Tensor,
    visual_scores: torch.Tensor,
    q_labels: torch.Tensor,
    max_docs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad or truncate score and label tensors to a fixed length

    Parameters
    ----------
    textual_scores : torch.Tensor
        Tensor of textual retrieval scores
    visual_scores : torch.Tensor
        Tensor of visual retrieval scores
    q_labels : torch.Tensor
        Tensor of relevance labels
    max_docs : int
        Target number of items per query

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Textual scores, visual scores, and labels adjusted to max_docs
    """
    pad_len = max_docs - len(textual_scores)

    if pad_len > 0:
        textual_scores = torch.cat([textual_scores, torch.zeros(pad_len)])
        visual_scores = torch.cat([visual_scores, torch.zeros(pad_len)])
        q_labels = torch.cat([q_labels, torch.zeros(pad_len)])
    else:
        textual_scores = textual_scores[:max_docs]
        visual_scores = visual_scores[:max_docs]
        q_labels = q_labels[:max_docs]

    return textual_scores, visual_scores, q_labels


def process_single_query(
    query_id: int,
    query_dict: dict,
    labels: dict,
    textual_retriever: TextualRetriever,
    visual_retriever: VisualRetriever,
    k: int,
    max_docs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract normalized retrieval features and labels for a single query

    Parameters
    ----------
    query_id : int
        Query to process
    query_dict : dict
        Mapping from query IDs to query metadata
    labels : dict
        Mapping from query IDs, corpus id to relevance labels
    textual_retriever : TextualRetriever
        Textual retriever
    visual_retriever : VisualRetriever
        Visual retriever
    k : int
        Number of candidates retrieved by each retriever
    max_docs : int
        Maximum number of documents retained

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Textual scores, visual scores, and labels for the query
    """
    query = query_dict[query_id]["query"]

    textual_res = textual_retriever.search(query, k=k)
    visual_res = visual_retriever.search(query, k=k)

    textual_map, visual_map, doc_ids = build_doc_maps(textual_res, visual_res)

    textual_scores, visual_scores, q_labels = build_score_vectors(
        doc_ids,
        textual_map,
        visual_map,
        labels,
        query_id,
    )

    textual_scores, visual_scores = normalize_scores(textual_scores, visual_scores)

    textual_scores, visual_scores, q_labels = pad_or_truncate(
        textual_scores,
        visual_scores,
        q_labels,
        max_docs,
    )

    return textual_scores, visual_scores, q_labels


def build_retrieval_features(
    query_dict: dict,
    labels: dict,
    textual_retriever: TextualRetriever,
    visual_retriever: VisualRetriever,
    k: int = 20,
    max_docs: int = 40,
) -> tuple[dict, dict, dict]:
    """
    Build retrieval feature dictionaries for all queries

    Parameters
    ----------
    query_dict : dict
        Mapping from query IDs to query metadata
    labels : dict
        Mapping from query IDs, corpus id to relevance labels
    textual_retriever : TextualRetriever
        Textual retriever
    visual_retriever : VisualRetriever
        Visual retriever
    k : int
        Number of candidates retrieved by each retriever
    max_docs : int
        Maximum number of documents retained

    Returns
    -------
    tuple[dict, dict, dict]
        Dictionaries mapping query IDs to textual score tensors, visual score
        tensors, and relevance label tensors
    """
    docs_textual_scores, docs_visual_scores, docs_labels_dict = {}, {}, {}

    for query_id in tqdm(
        query_dict.keys(),
        desc="Building retrieval_feature",
        colour="cyan",
    ):
        textual_scores, visual_scores, q_labels = process_single_query(
            query_id,
            query_dict,
            labels,
            textual_retriever,
            visual_retriever,
            k,
            max_docs,
        )

        docs_textual_scores[query_id] = textual_scores
        docs_visual_scores[query_id] = visual_scores
        docs_labels_dict[query_id] = q_labels

    return docs_textual_scores, docs_visual_scores, docs_labels_dict


def create_dataloader(
    query_dict: dict,
    textual_scores: dict,
    visual_scores: dict,
    labels: dict,
    query_ids: list[int],
    shuffle: bool,
) -> DataLoader:
    """
    Create a DataLoader for scorer training or evaluation

    Parameters
    ----------
    query_dict : dict
        Mapping from query IDs to query metadata
    textual_scores : dict
        Mapping from query IDs to textual score
    visual_scores : dict
        apping from query IDs to visual score
    labels : dict
        Mapping from query IDs to relevance label
    query_ids : list[int]
        Query IDs to include in the dataset
    shuffle : bool
        Whether to shuffle dataset items in the DataLoader

    Returns
    -------
    DataLoader
        DataLoader wrapping the scorer dataset
    """
    dataset = ScorerDataset(query_dict, textual_scores, visual_scores, labels, query_ids)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def main() -> None:
    """
    Train and evaluate the scorer model used for hybrid score fusion
    """
    logger.info("Starting Scorer model training...")
    try:
        corpus, queries, qrels = load_vidore_dataset(DATA_FOLDER)
        _, query_dict, labels = dataset_join(corpus, queries, qrels)
        logger.info("Dataset successfully loaded")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info("Building retrieval features...")
    if not FEATURE_CACHE_PATH.exists():
        logger.info("Features not found — generating them...")
        db_handler = QdrantHandler(DB_HOST, DB_PORT)
        textual_retriever = TextualRetriever(TEXTUAL_DATABASE_NAME, db_handler, TEXTUAL_EMBEDDER)
        visual_retriever = VisualRetriever(VISUAL_DATABASE_NAME, db_handler, VISUAL_EMBEDDER)

        textual_scores, visual_scores, labels_vectors = build_retrieval_features(
            query_dict,
            labels,
            textual_retriever,
            visual_retriever,
            NB_DOCS_RETRIEVAL,
            TOTAL_MAX_DOC_RETRIEVAL,
        )
        CacheManager.save(
            (textual_scores, visual_scores, labels_vectors),
            FEATURE_CACHE_PATH,
        )
    else:
        logger.info("Features found, loading them !")
        textual_scores, visual_scores, labels_vectors = CacheManager.load(
            FEATURE_CACHE_PATH,
        )
    logger.info("Feature extraction completed")

    train_ids, val_ids, test_ids = train_validation_test_split(
        query_dict,
        VAL_RATIO,
        TEST_RATIO,
    )
    logger.info(
        f"Dataset sizes — Train: {len(train_ids)} | "
        f"Validation: {len(val_ids)} | "
        f"Test: {len(test_ids)}",
    )

    train_loader = create_dataloader(
        query_dict,
        textual_scores,
        visual_scores,
        labels_vectors,
        train_ids,
        shuffle=True,
    )

    val_loader = create_dataloader(
        query_dict,
        textual_scores,
        visual_scores,
        labels_vectors,
        val_ids,
        shuffle=False,
    )

    test_loader = create_dataloader(
        query_dict,
        textual_scores,
        visual_scores,
        labels_vectors,
        test_ids,
        shuffle=False,
    )

    model = ScorerModel(TRANSFORMER_NAME, HIDDEN_DIM, METRICS_NB_DOC)

    logger.info("Starting training...")
    model.fit(train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS)

    logger.info("Evaluating on test set...")
    test_loss, test_metrics = model._evaluate(test_loader, margin=0.1, verbose=True)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(
        f"Test Metrics — nDCG@{METRICS_NB_DOC}: {test_metrics[f'ndcg@{METRICS_NB_DOC}']:.4f} | "
        f"Recall@{METRICS_NB_DOC}: {test_metrics['recall@{METRICS_NB_DOC}']:.4f}",
    )
    logger.info(f"Saving trained model to: {MODEL_PATH}")
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
