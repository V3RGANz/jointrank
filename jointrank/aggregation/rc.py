from typing import TypeVar

import numpy as np
from scipy.linalg import eig

from jointrank.aggregation.base import RankAggregator
from jointrank.blocks.comparisons import ComparisonInstance, Verdict

T = TypeVar("T")


# https://github.com/erensezener/rank-centrality Copyright (c) 2018 C. Eren Sezener MIT License
def extract_rc_scores(comparisons: list[tuple[T, T]], regularized: bool = True) -> dict[T, float]:  # noqa
    """
    Computes the Rank Centrality scores given a list of pairwise comparisons based on Negahban et al 2016 [1].

    Note it is assumed that the comparisons cannot result in a draw. If you want to include draws, then you can
    treat a draw between `A` and `B` as `A` winning over `B` AND `B` winning over `A`. So for a draw, you can add
    `(A, B)` and `(B, A)` to `comparisons`.

    The regularized version is also implemented. This could be useful when the number of comparisons are small
    with respect to the number of unique items. Note that for properly ranking, number of samples should be in the
    order of n logn, where n is the number of unique items.

    References

    1- Negahban, Sahand et al. “Rank Centrality: Ranking from Pairwise Comparisons.” Operations Research 65 (2017):
    266-287. DOI: https://doi.org/10.1287/opre.2016.1534

    :param comparisons: List of pairs, in `[(winnner, loser)]` format.

    :param regularized: If True, assumes a Beta prior.

    :return: A dictionary of `item -> score`
    """  # noqa


    winners, losers = zip(*comparisons, strict=True)
    unique_items = np.hstack([np.unique(winners), np.unique(losers)])

    item_to_index = {item: i for i, item in enumerate(unique_items)}

    A = np.ones((len(unique_items), len(unique_items))) * regularized  # Initializing as ones results in the Beta prior  # noqa

    for w, l in comparisons:  # noqa: E741
        A[item_to_index[l], item_to_index[w]] += 1

    A_sum = (A[np.triu_indices_from(A, 1)] + A[np.tril_indices_from(A, -1)]) + 1e-6  # to prevent division by zero  # noqa

    A[np.triu_indices_from(A, 1)] /= A_sum
    A[np.tril_indices_from(A, -1)] /= A_sum

    d_max = np.max(np.sum(A, axis=1))
    A /= d_max  # noqa

    w, v = eig(A, left=True, right=False)  # type: ignore

    max_eigv_i = np.argmax(w)
    scores = np.abs(np.real(v[:, max_eigv_i]))  # type: ignore

    return {item: scores[index] for item, index in item_to_index.items()}


class RankCentralityAggregator(RankAggregator):
    @property
    def name(self) -> str:
        return "rank-centrality"

    def score_from_comp(self, comparisons: list[ComparisonInstance]) -> list[float]:
        wl = [(x, y) if v is Verdict.Won else (y, x) for x, y, v in comparisons if v != Verdict.Draw]
        scores_dict = extract_rc_scores(wl, regularized=True)
        xs, ys, _ = zip(*comparisons, strict=True)
        total_items = max(xs + ys) + 1
        return [scores_dict.get(i, min(scores_dict.values())) for i in range(total_items)]
