from abc import ABC, abstractmethod


class RerankerError(Exception):
    pass


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[str]) -> list[int]: ...
    """For given query and candidates, return permutation according to the new ranking"""

    # For different models (async requests, sync requests, local models) may be different implementations
    @abstractmethod
    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]: ...


class DummyReranker(Reranker):
    def rerank(self, query: str, candidates: list[str]) -> list[int]:  # noqa: ARG002
        return list(range(len(candidates)))

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        return [list(range(len(candidates))) for _, candidates in batch]
