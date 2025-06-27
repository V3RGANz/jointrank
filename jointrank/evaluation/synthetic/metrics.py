import numpy as np


def dcg(relevances: list[int], k: int | None = None) -> float:
    relevances_ = np.array(relevances, dtype=float)
    if k is not None:
        relevances_ = relevances_[:k]
    if relevances_.size:
        discounts = np.log2(np.arange(2, relevances_.size + 2))
        return float(np.sum(relevances_ / discounts))
    return 0

def ndcg(relevances: list[int], k: int | None = None) -> float:
    actual_dcg = dcg(relevances, k)
    ideal_dcg = dcg(sorted(relevances, reverse=True), k)
    if ideal_dcg == 0:
        return 0
    return actual_dcg / ideal_dcg

def accuracy(relevances: list[int], k: int) -> float:
    top_sorted_relevances = set(sorted(relevances, reverse=True)[:k]) - {0}  # 0 relevance should be ignored
    top_actual_relevances = relevances[:k]

    hits = len(set(top_actual_relevances) & top_sorted_relevances)
    total = len(top_sorted_relevances)

    return hits / total if total else 0
