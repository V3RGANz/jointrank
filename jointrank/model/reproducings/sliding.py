from jointrank.model.reranker_base import Reranker


class SlidingWindorReranker(Reranker):
    def __init__(self, listwise: Reranker, window_size: int, step: int) -> None:
        self.listwise = listwise
        self.window_size = window_size
        self.step = step

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        ids = list(range(len(candidates)))
        end = len(candidates)
        start = end - self.window_size
        while start > -self.step:
            start = max(0, start)
            window_ids = ids[start:end]
            window = [candidates[i] for i in window_ids]
            window_ranking = self.listwise.rerank(query, window)
            backfill = [i for i in range(len(window)) if i not in window_ranking]
            window_ranking += backfill
            window_ids_ranked = [window_ids[i] for i in window_ranking]
            ids = ids[:start] + window_ids_ranked + ids[end:]
            start = start - self.step
            end = end - self.step
        return ids

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        raise NotImplementedError
