import itertools
import logging
from enum import Enum

LOG = logging.getLogger(__name__)


class Verdict(int, Enum):
    Won = 1
    Draw = 0
    Lost = -1


ComparisonInstance = tuple[int, int, Verdict]


def comparisons_from_rankings(
    rankings: list[list[int]]
) -> list[ComparisonInstance]:
    comparisons: list[tuple[int, int, Verdict]] = []
    for ranking in rankings:
        for (i, first), (j, second) in itertools.combinations(list(enumerate(ranking)), 2):
            verdict = Verdict.Won if i < j else Verdict.Lost
            comparisons.append((first, second, verdict))
    return comparisons
