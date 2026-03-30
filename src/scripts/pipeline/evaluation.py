import logging

import numpy as np
from tqdm import tqdm

from agentic_rag.pipeline.pipeline import HybridRAGPipeline
from agentic_rag.utils.metrics import ndcg_at_k, recall_at_k
from scripts.pipeline.config import (
    DATA_FOLDER,
    DB_HOST,
    DB_PORT,
    LLM_MODEL_NAME,
    METRICS_NB_DOC,
    MODE,
    NB_RETRIEVE,
    SCORER_PATH,
)
from scripts.retriever.textual.config import DATABASE_NAME as TEXTUAL_DATABASE_NAME
from scripts.retriever.textual.config import EMBEDDER_MODEL_NAME as TEXTUAL_EMBEDDER
from scripts.retriever.visual.config import DATABASE_NAME as VISUAL_DATABASE_NAME
from scripts.retriever.visual.config import EMBEDDER_MODEL_NAME as VISUAL_EMBEDDER
from scripts.utils.dataset import dataset_join, load_vidore_dataset
from scripts.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def get_scores(
    pipeline: HybridRAGPipeline,
    query: str,
    labels_of_query: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute predicted and ground-truth scores for a single query

    Parameters
    ----------
    pipeline : HybridRAGPipeline
        Pipeline used to retrieve documents
    query : str
        Input query
    labels_of_query : dict
        Mapping from document to labels

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The predicted retrieval scores and the corresponding ground-truth relevance scores
    """
    docs_retrieved = pipeline.retrieve(query, NB_RETRIEVE, mode=MODE)
    pred_map = {}
    for doc in docs_retrieved:
        pred_map[doc.corpus_id] = max(doc.score, pred_map.get(doc.corpus_id, 0.0))

    all_doc_ids = set(labels_of_query.keys()) | set(pred_map.keys())

    pred_scores, true_scores = [], []
    for doc_id in all_doc_ids:
        pred_scores.append(pred_map.get(doc_id, 0.0))
        true_scores.append(labels_of_query.get(doc_id, 0.0))
    return np.array(pred_scores, dtype=float), np.array(true_scores, dtype=float)


def evaluate_rag(
    pipeline: HybridRAGPipeline,
    query_dict: dict,
    labels: dict,
    k: int,
) -> tuple[float, float]:
    """
    Evaluate the hybrid RAG pipeline over a set of queries

    Parameters
    ----------
    pipeline : HybridRAGPipeline
        Pipeline to evaluate
    query_dict : dict
        Mapping from query IDs to query metadata
    labels : dict
        Mapping from query IDs to relevance labels
    k : int
        Cutoff rank used to compute evaluation metrics

    Returns
    -------
    tuple[float, float]
        Mean nDCG@k and mean Recall@k over all evaluated queries
    """
    ndcg_scores = []
    recall_scores = []

    with tqdm(
        query_dict.keys(),
        desc="Evaluating RAG",
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} {postfix}",
        colour="green",
    ) as eval_bar:
        for query_id in eval_bar:
            pred_scores, true_scores = get_scores(
                pipeline,
                query_dict[query_id]["query"],
                labels.get(query_id, {}),
            )

            ndcg_score = ndcg_at_k(pred_scores, true_scores, k)
            recall_score = recall_at_k(pred_scores, true_scores, k)
            ndcg_scores.append(ndcg_score)
            recall_scores.append(recall_score)

            eval_bar.set_postfix_str(
                f"ndcg@{k}={ndcg_score:.3f}  recall@{k}={recall_score:.3f}",
            )

    return float(np.mean(ndcg_scores)), float(np.mean(recall_scores))


def main() -> None:
    """
    Run the evaluation pipeline for the hybrid RAG system. It loads the evaluation dataset, initializes the hybrid
    retrieval pipeline, computes retrieval metrics, and logs the final nDCG and Recall scores
    """
    logger.info("Starting Hybrid RAG evaluation...")
    try:
        corpus, queries, qrels = load_vidore_dataset(DATA_FOLDER)
        _, query_dict, labels = dataset_join(corpus, queries, qrels)
        logger.info("Dataset successfully loaded")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info("Loading pipeline...")
    pipeline = HybridRAGPipeline(
        DB_HOST,
        DB_PORT,
        TEXTUAL_DATABASE_NAME,
        VISUAL_DATABASE_NAME,
        TEXTUAL_EMBEDDER,
        VISUAL_EMBEDDER,
        scorer_model_path=SCORER_PATH,
        llm_model_name=LLM_MODEL_NAME,
    )
    logger.info(f"Starting evaluation for mode : {MODE.value}")
    ndcg_score, recall_score = evaluate_rag(pipeline, query_dict, labels, METRICS_NB_DOC)
    logger.info(f"nDCG@{METRICS_NB_DOC}: {ndcg_score:.4f}")
    logger.info(f"Recall@{METRICS_NB_DOC}: {recall_score:.4f}")


if __name__ == "__main__":
    main()
