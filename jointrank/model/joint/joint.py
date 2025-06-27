import logging

from jointrank.aggregation.base import RankAggregator
from jointrank.blocks.design import BlockDesign
from jointrank.model.reranker_base import Reranker

LOG = logging.getLogger(__name__)


class JointRanker(Reranker):
    """Let's roll!"""

    def __init__(
        self,
        block_design: BlockDesign,
        base_reranker: Reranker,
        aggregator: RankAggregator,
        block_size: int,
    ) -> None:
        self.base_reranker = base_reranker
        self.aggregator = aggregator
        self.block_design = block_design
        self.block_size = block_size

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        if len(candidates) < 2:
            return list(range(len(candidates)))
        assert self.block_design.is_applicable(len(candidates), self.block_size), (
            f"{self.block_design.__class__.__name__} not applicable for (v={len(candidates)} k={self.block_size})"
        )
        blocks = self.block_design.build(len(candidates), self.block_size)
        # assert set(range(len(candidates))) == {i for b in blocks for i in b}, f"{set(range(len(candidates))) - {i for b in blocks for i in b}}"

        batch = [
            (query, [candidates[i] for i in block])
            for block in blocks
        ]

        blocks_rankings = self.base_reranker.rerank_batch(batch)

        rankings = [
            [block[i] for i in self.__backfill(block_ranking, len(block))]
            for block, block_ranking in zip(blocks, blocks_rankings, strict=True)
        ]

        scores = self.aggregator.score(rankings)
        assert len(scores) == len(candidates), f"{len(scores)} != {len(candidates)}"

        ranking, _ = zip(*sorted(
            zip(range(len(candidates)), scores, strict=True),
            key=lambda x: x[1],
            reverse=True
        ), strict=True)

        return list(ranking)

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        raise NotImplementedError

    def __backfill(self, block_ranking: list[int], block_size: int) -> list[int]:
        return block_ranking + [i for i in range(block_size) if i not in block_ranking]
