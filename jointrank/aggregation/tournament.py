from typing import Any
import evalica
from jointrank.aggregation.base import RankAggregator
from jointrank.blocks.comparisons import ComparisonInstance, Verdict


class TournamentAggregator(RankAggregator):
    def __init__(self, method: str, **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs

    def score_from_comp(self, comparisons: list[ComparisonInstance]) -> list[float]:
        xs, ys, verdicts = zip(*comparisons, strict=True)
        total_items = max(xs + ys) + 1
        winners = [evalica.Winner.X if v is Verdict.Won
                   else evalica.Winner.Y if v is Verdict.Lost
                   else evalica.Winner.Draw for v in verdicts]

        match self.method:
            case "elo":
                func = evalica.elo
            case "eigen":
                func = evalica.eigen
            case "bradley":
                func = evalica.bradley_terry
            case "pagerank":
                func = evalica.pagerank
            case "wr":
                func = evalica.average_win_rate
            case "newman":
                func = evalica.newman
            case _:
                raise ValueError("unknown method: " + self.method)

        results = func(xs, ys, winners, **self.kwargs)
        scores, _ = results.scores, results.index
        min_score = min(scores)
        return [float(scores[i]) if i in scores.index else min_score for i in range(total_items)]

    @property
    def name(self) -> str:
        suffix = ""
        if self.kwargs:
            kvs = [f"{k}={v}" for k,v in self.kwargs.items()]
            suffix = "-" + "-".join(kvs)
        return "tournament-" + self.method + suffix
