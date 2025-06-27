import argparse
import itertools
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd

from jointrank.aggregation.base import RankAggregator
from jointrank.aggregation.rc import RankCentralityAggregator
from jointrank.aggregation.tournament import TournamentAggregator
from jointrank.blocks.design import (
    BlockDesign,
    EquiReplicateBlockDesign,
    LatinSquarePBIBD,
    RandomizedBlockDesign,
    TriangularPBIBD,
)
from jointrank.evaluation.synthetic.run import run_single


def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    dst = cli_args().dst

    result = []

    aggregators: list[RankAggregator] = [
        RankCentralityAggregator(),
        *[TournamentAggregator(method) for method in ["elo", "eigen", "bradley", "pagerank", "wr", "newman"]]
    ]

    i = 0
    pool = Pool(10)
    replica_factor = 2
    total_sizes = [55, 100]
    block_sizes = [10, 20]

    designs: list[BlockDesign] = [
        EquiReplicateBlockDesign(replica_factor),
        RandomizedBlockDesign(replica_factor),
        TriangularPBIBD(),
        LatinSquarePBIBD()
    ]

    for total_items, block_size, design, aggregator in itertools.product(
        total_sizes,
        block_sizes,
        designs,
        aggregators,
    ):
        if not design.is_applicable(total_items, block_size):
            continue

        accuracies, ndcgs, blocks_counts = [], [], []
        args = (aggregator, design, total_items, block_size)
        argseq = itertools.repeat(args, 1000)
        for ac_at_1, ndcg_at_10, blocks_count in pool.starmap(run_single, argseq):
            accuracies.append(ac_at_1)
            ndcgs.append(ndcg_at_10)
            blocks_counts.append(blocks_count)

        assert all(bc == blocks_counts[0] for bc in blocks_counts)
        blocks_count = blocks_counts[0]

        print("iter", i, aggregator.name, type(design).__name__, blocks_count, "ndcg@10",
              int(100 * sum(ndcgs) / len(ndcgs)))
        i += 1
        ndcg_mean = sum(ndcgs) / len(ndcgs)
        ndcg_std = (sum((x - ndcg_mean) ** 2 for x in ndcgs) / len(ndcgs)) ** 0.5
        ndcg_conf_interval = 1.96 * (ndcg_std / len(ndcgs) ** 0.5)

        result.append({
            "total_items": total_items,
            "block_size": block_size,
            "blocks_count": blocks_count,
            "design": type(design).__name__,
            "aggregator": aggregator.name,
            "ndcg@10": ndcg_mean,
            "acc@1": sum(accuracies) / len(accuracies),
            "ndcg_conf": ndcg_conf_interval
        })

    parent = Path(dst).parent
    parent.mkdir(exist_ok=True, parents=True)
    result_df = pd.DataFrame(result)
    result_df.to_csv(dst, index=False)


if __name__ == "__main__":
    main()
