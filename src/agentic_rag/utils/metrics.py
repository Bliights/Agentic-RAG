import numpy as np


def dcg(scores: np.ndarray) -> float:
    """
    Compute the DCG for a list of relevance scores

    Parameters
    ----------
    scores : np.ndarray
        Relevance scores ordered by predicted ranking

    Returns
    -------
    float
        Discounted cumulative gain value
    """
    return sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(scores))


def ndcg_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int = 5) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k

    Parameters
    ----------
    pred_scores : np.ndarray
        Predicted scores used to rank items
    true_labels : np.ndarray
        Ground-truth relevance labels
    k : int, optional
        Cutoff rank for evaluation

    Returns
    -------
    float
        NDCG@k score
    """
    idx = np.argsort(pred_scores)[::-1][:k]
    gains = true_labels[idx]

    ideal_idx = np.argsort(true_labels)[::-1][:k]
    ideal_gains = true_labels[ideal_idx]

    if dcg(ideal_gains) == 0:
        return 0

    return dcg(gains) / (dcg(ideal_gains))


def recall_at_k(pred_scores: np.ndarray, true_labels: np.ndarray, k: int = 5) -> float:
    """
    Compute Recall at rank k

    Parameters
    ----------
    pred_scores : np.ndarray
        Predicted scores used to rank items
    true_labels : np.ndarray
        Ground-truth relevance labels
    k : int, optional
        Cutoff rank for evaluation

    Returns
    -------
    float
        Recall@k score
    """
    idx = np.argsort(pred_scores)[::-1][:k]
    gains = true_labels[idx]

    relevant_total = (true_labels > 0).sum()
    if relevant_total == 0:
        return 1.0

    return float((gains > 0).sum()) / float(relevant_total)
