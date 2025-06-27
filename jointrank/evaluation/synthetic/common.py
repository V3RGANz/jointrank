from typing import Any

from jointrank.blocks.design import BlockDesign, EquiReplicateBlockDesign, LatinSquarePBIBD, RandomizedBlockDesign, TriangularPBIBD


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
