import random
from abc import ABC, abstractmethod

import numpy as np


class BlockDesign(ABC):
    @abstractmethod
    def is_applicable(self, total_items: int, block_size: int) -> bool: ...
    @abstractmethod
    def build(self, total_items: int, block_size: int) -> list[list[int]]: ...


class EquiReplicateBlockDesign(BlockDesign):
    def __init__(self, replica_factor: int = 2) -> None:
        self.replica_factor = replica_factor

    def is_applicable(self, total_items: int, block_size: int) -> bool:
        return total_items <= block_size or (self.replica_factor * total_items) % block_size == 0

    def build(self, total_items: int, block_size: int) -> list[list[int]]:
        blocks: list[list[int]] = []

        if total_items <= block_size:
            blocks = [list(range(total_items)) for _ in range(self.replica_factor)]
            for block in blocks:
                random.shuffle(block)
            return blocks
        blocks_count = self.replica_factor * total_items // block_size
        items_pool = list(range(total_items))
        random.shuffle(items_pool)

        while(len(blocks) < blocks_count):
            if len(items_pool) >= block_size:
                blocks.append(items_pool[:block_size])
                items_pool = items_pool[block_size:]
            else:
                extra_items = random.sample(
                    list(set(range(total_items)) - set(items_pool)),
                    block_size - len(items_pool)
                )
                blocks.append(items_pool + extra_items)
                items_pool = [i for i in range(total_items) if i not in extra_items]
                random.shuffle(items_pool)

        return blocks


class RandomizedBlockDesign(BlockDesign):
    def __init__(self, total_items_factor: int = 2) -> None:
        self.total_items_factor = total_items_factor

    def is_applicable(self, total_items: int, block_size: int) -> bool:
        return self.total_items_factor * total_items % block_size == 0

    def build(self, total_items: int, block_size: int) -> list[list[int]]:
        blocks_count = (self.total_items_factor * total_items) // block_size
        return [random.sample(range(total_items), block_size) for _ in range(blocks_count)]


# For situations, where *clean* Equi-Replicate is not possible due to restriction
# in EquiReplicateBlockDesign.is_applicable
class ReplicateBlockDesign(BlockDesign):
    def __init__(self, replica_factor: int = 2) -> None:
        self.replica_factor = replica_factor

    def is_applicable(self, total_items: int, block_size: int) -> bool:  # noqa: ARG002
        return True

    def build(self, total_items: int, block_size: int) -> list[list[int]]:
        blocks: list[list[int]] = []

        if total_items <= block_size:
            blocks = [list(range(total_items)) for _ in range(self.replica_factor)]
            for block in blocks:
                random.shuffle(block)
            return blocks

        blocks_count = self.replica_factor * total_items // block_size or 1
        items_pool = list(range(total_items))
        random.shuffle(items_pool)

        while(len(blocks) < blocks_count):
            if len(items_pool) >= block_size:
                blocks.append(items_pool[:block_size])
                items_pool = items_pool[block_size:]
            else:
                extra_items = random.sample(
                    list(set(range(total_items)) - set(items_pool)),
                    block_size - len(items_pool)
                )
                blocks.append(items_pool + extra_items)
                items_pool = [i for i in range(total_items) if i not in extra_items]
                random.shuffle(items_pool)

        return blocks


class LatinSquarePBIBD(BlockDesign):
    def is_applicable(self, total_items: int, block_size: int) -> bool:
        return block_size * block_size == total_items

    def build(self, total_items: int, block_size: int) -> list[list[int]]:  # noqa: ARG002
        square = np.arange(block_size * block_size).reshape(block_size, block_size)
        return square.tolist() + square.T.tolist()


class TriangularPBIBD(BlockDesign):
    def is_applicable(self, total_items: int, block_size: int) -> bool:
        return total_items * 2 == block_size * (block_size + 1)

    def build(self, total_items: int, block_size: int) -> list[list[int]]:
        assotiation_scheme = np.full((block_size + 1, block_size + 1), 0, dtype=np.long)
        assotiation_scheme[np.triu_indices(block_size + 1, 1)] = np.arange(total_items)
        assotiation_scheme = assotiation_scheme + assotiation_scheme.T
        assotiation_scheme[np.diag_indices(block_size + 1)] = -1

        return [[b for b in block if b != -1] for block in assotiation_scheme.tolist()]


class BlockDesignCollection(BlockDesign):
    def __init__(self, *designs: BlockDesign) -> None:
        self.designs = designs

    def is_applicable(self, total_items: int, block_size: int) -> bool:
        return any(d.is_applicable(total_items, block_size) for d in self.designs)

    def build(self, total_items: int, block_size: int) -> list[list[int]]:
        design = next(d for d in self.designs if d.is_applicable(total_items, block_size))
        return design.build(total_items, block_size)
