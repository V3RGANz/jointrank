import logging

import more_itertools

from jointrank.model.reranker_base import Reranker

LOG = logging.getLogger(__name__)


class TDPartRanker(Reranker):
    """Top-Down Partitioning for Efficient List-Wise Ranking.

    https://arxiv.org/pdf/2405.14589
    Parallelized version
    """

    def __init__(self, base_ranker: Reranker, k: int = 9, window_size: int = 10, max_iters: int = 100) -> None:
        assert 1 < k < window_size, f"k={k} and window_size={window_size} not satisfy 1 < {k} < {window_size}"
        self.base_ranker = base_ranker
        self.k = k
        self.window_size = window_size
        self.max_iters = max_iters

    def _rerank_subsequences_batch(
        self, query: str, candidates: list[str], candidates_batch: list[list[int]]
    ) -> list[list[int]]:
        batch = [
            (query, [candidates[i] for i in candidate_ids])
            for candidate_ids in candidates_batch
        ]
        ranked_batch = self.base_ranker.rerank_batch(batch)
        return [
            [candidate_ids[i] for i in ranked_ids] + [c for i, c in enumerate(candidate_ids) if i not in ranked_ids]
            for ranked_ids, candidate_ids in zip(ranked_batch, candidates_batch, strict=True)
        ]

    def _pivot(self, query: str, candidates: list[str], candidate_ids: list[int]) -> tuple[list[int], list[int], bool]:
        left, right = candidate_ids[:self.window_size], candidate_ids[self.window_size:]
        left_sorted = self._rerank_subsequences_batch(query, candidates, [left])[0]
        if len(right) == 0:
            return left_sorted, right, True

        pivot = left_sorted[self.k]
        backfill = left_sorted[self.k + 1:]
        candidate_ids = left_sorted[:self.k]

        batch = [
            [pivot, *chunk]
            for chunk in more_itertools.chunked(right, self.window_size - 1)
        ]
        ranked_batch = self._rerank_subsequences_batch(query, candidates, batch)

        for chunk in ranked_batch:
            pivot_position = chunk.index(pivot)
            candidate_ids += chunk[:pivot_position]
            backfill += chunk[pivot_position + 1:]

        if len(candidate_ids) < self.k:
            return [*candidate_ids, pivot], backfill, True

        return candidate_ids, [pivot, *backfill], False

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        candidate_ids = list(range(len(candidates)))
        backfill = []

        for i in range(self.max_iters):
            candidate_ids, _backfill, is_finished = self._pivot(query, candidates, candidate_ids)
            backfill += _backfill
            LOG.info(
                "tdpart %d candidates set: %d, backfill: %d, isfinished: %s",
                i, len(candidate_ids), len(backfill), is_finished
            )
            if is_finished:
                break
        return candidate_ids + backfill

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        raise NotImplementedError
