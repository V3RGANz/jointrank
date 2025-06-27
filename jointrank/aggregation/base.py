from abc import abstractmethod

from jointrank.blocks.comparisons import ComparisonInstance, comparisons_from_rankings


class RankAggregator:
    def score(self, rankings: list[list[int]]) -> list[float]:
        return self.score_from_comp(comparisons_from_rankings(rankings))

    @abstractmethod
    def score_from_comp(self, comparisons: list[ComparisonInstance]) -> list[float]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
