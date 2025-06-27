import random
from functools import partial

from jointrank.aggregation.base import RankAggregator
from jointrank.blocks.design import BlockDesign
from jointrank.evaluation.synthetic.metrics import accuracy, ndcg


def simulate_ideal_ranker(block: list[int], gold_relevance: list[int]) -> list[float]:
    return [float(gold_relevance[idx]) for idx in block]


def sort_block(block: list[int], scores: list[float]) -> list[int]:
    return [item for item, _ in sorted(zip(block, scores, strict=True), key=lambda x: x[1], reverse=True)]


def run_single(
    aggregator: RankAggregator,
    block_design: BlockDesign,
    total_items: int,
    block_size: int,
) -> tuple[float, float, int]:

    blocks = block_design.build(total_items, block_size)
    gold_relevances = list(reversed(range(1, total_items + 1)))

    random.shuffle(gold_relevances)

    block_scores = list(map(partial(simulate_ideal_ranker, gold_relevance=gold_relevances), blocks))

    scores = aggregator.score([
        sort_block(block, b_scores) for block, b_scores in zip(blocks, block_scores, strict=True)
    ])
    min_score = min(scores) - 1
    scores = [scores[i] if i < len(scores) else min_score for i in range(total_items)]

    ids_sorted, _ = zip(
        *sorted(zip(range(total_items), scores, strict=True), key=lambda x: x[1], reverse=True),
        strict=True
    )

    relevances = [gold_relevances[ranked] for ranked in ids_sorted]
    relevances = [2**r for r in relevances]

    acc_at_1 = accuracy(relevances, 1)
    ndcg_at_10 = ndcg(relevances, 10)

    return acc_at_1, ndcg_at_10, len(blocks)
