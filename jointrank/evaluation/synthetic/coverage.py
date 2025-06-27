import argparse
import itertools
from collections.abc import Sequence
from multiprocessing.pool import Pool
from typing import Any

import numpy as np

from jointrank.blocks.design import (
    BlockDesign,
    EquiReplicateBlockDesign,
    LatinSquarePBIBD,
    RandomizedBlockDesign,
    TriangularPBIBD,
)


def get_design_by_name(name: str, *args: Any) -> BlockDesign:
    match name:
        case "equi-replicate":
            return EquiReplicateBlockDesign(*args)
        case "random":
            return RandomizedBlockDesign(*args)
        case "latin":
            return LatinSquarePBIBD()
        case "triangular":
            return TriangularPBIBD()
        case _:
            raise ValueError

def conn_check(blocks: list[list[int]]) -> bool:
    """Check design connectivity.

    Note: for tournament graph connectivity, corresponding design connectivity is sufficient.
    """
    block_graph: dict[int, set[int]] = {}
    for (i, b1), (j, b2) in itertools.combinations(list(enumerate(blocks)), 2):
        if set(b1) & set(b2):
            block_graph.setdefault(i, set()).add(j)
            block_graph.setdefault(j, set()).add(i)
    for i, j in itertools.combinations(block_graph.keys(), 2):
        visited: set[int] = set()
        stack = list(block_graph[i])
        while stack:
            k = stack.pop(0)
            if k in visited:
                continue
            visited.add(k)
            if k == j:
                break
            stack.extend(kk for kk in block_graph[k] if kk not in visited and kk not in stack)
        if j not in visited:
            return False
    return True


def get_stats(design: BlockDesign, total_size: int, block_size: int):
    blocks = design.build(total_size, block_size)
    adj = np.zeros((total_size, total_size), dtype=np.short)
    for block in blocks:
        for x, y in itertools.combinations(block, 2):
            adj[x, y] += 1
            adj[y, x] += 1
    adj_unique = np.bool(adj)

    first_order_pairs = int(np.triu(adj_unique, k=1).sum())
    counts: list[int] = adj.sum(axis=0).tolist()
    cnk = (total_size * (total_size - 1)) // 2
    cooccurrences = []
    cooccurrences = adj[np.triu_indices(total_size, k=1)].tolist()
    cooc_arr = np.array(cooccurrences)
    mean_cooc = np.mean(cooc_arr)
    max_cooc = max(cooccurrences)
    allpairs = set()

    for x in range(total_size):
        for y in range(x + 1, total_size - 1):
            if not adj[x, y]:
                continue
            for z in range(y + 1, total_size):
                if not adj[y, z]:
                    continue
                allpairs.add((x,z))

    comp1rate = first_order_pairs / cnk
    comp2rate = len(allpairs) / cnk
    degree_min = min(counts)
    degree_max = max(counts)
    degree_mean = sum(counts) / total_size
    connectivity = conn_check(blocks)
    return blocks, comp1rate, comp2rate, degree_mean, degree_min, degree_max, mean_cooc, max_cooc, connectivity

def avg(x: Sequence[bool | int | float]) -> float:
    return sum(x) / len(x)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--block-size", type=int, required=True)
    parser.add_argument("-r", "--num-replicas", type=int, required=True)
    parser.add_argument("-v", "--total-items", type=int, required=True)
    parser.add_argument("-d", "--design", type=get_design_by_name, required=True)
    args = parser.parse_args()

    print(args.design.__class__.__name__, args.total_items, args.block_size)
    _args = (args.design, args.total_items, args.block_size)
    if not args.design.is_applicable(args.total_items, args.block_size):
        print("design is not applicable for given parameters")

    argseq = itertools.repeat(_args, 1000)

    pool = Pool(10)

    stats = [stat for blocks, *stat in pool.starmap(get_stats, argseq)]

    comp1rate, comp2rate, degree_mean, degree_min, degree_max, mean_cooc, max_cooc, connectivity = zip(*stats,
                                                                                                       strict=True)
    max_avg_degree_idx, _ = max(enumerate(degree_mean), key=lambda x: x[1])
    max_com2rate_idx, _ = max(enumerate(comp2rate), key=lambda x: x[1])
    max_com1rate_idx, _ = max(enumerate(comp1rate), key=lambda x: x[1])
    min_avg_degree_idx, _ = min(enumerate(degree_mean), key=lambda x: x[1])
    min_com2rate_idx, _ = min(enumerate(comp2rate), key=lambda x: x[1])
    min_com1rate_idx, _ = min(enumerate(comp1rate), key=lambda x: x[1])

    print("comp1rate:", avg(comp1rate))
    print("comp2rate:", avg(comp2rate))
    print("degree_mean:", avg(degree_mean))
    print("degree_min:", avg(degree_min))
    print("degree_max:", avg(degree_max))
    print("mean_cooc:", avg(mean_cooc))
    print("max_cooc:", avg(max_cooc))
    print("connectivity rate:", avg(connectivity))

    for name, idx in [
        ("max_avg_degree", max_avg_degree_idx),
        ("max_comp2rate", max_com2rate_idx),
        ("max_comp1rate", max_com1rate_idx),

        ("min_avg_degree", min_avg_degree_idx),
        ("min_comp2rate", min_com2rate_idx),
        ("min_comp1rate", min_com1rate_idx),
    ]:
        print(name, idx, "dg=", degree_mean[idx], "comp1=", comp1rate[idx], "comp2=", comp2rate[idx])

if __name__ == "__main__":
    main()
