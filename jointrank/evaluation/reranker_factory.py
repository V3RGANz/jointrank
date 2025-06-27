from jointrank.aggregation.tournament import TournamentAggregator
from jointrank.blocks.design import (
    BlockDesignCollection,
    EquiReplicateBlockDesign,
    LatinSquarePBIBD,
    ReplicateBlockDesign,
)
from jointrank.evaluation.cfg import RankerConfig, RunConfig
from jointrank.evaluation.data import TRECDevRerankerSample, TRECGoldRelevance
from jointrank.model.joint.joint import JointRanker
from jointrank.model.joint.listwise import ListwiseRanker
from jointrank.model.joint.listwise_logits import ListwiseRankerLogits
from jointrank.model.llm.base import LLMEngine
from jointrank.model.reproducings.rankgpt import RankGPT
from jointrank.model.reproducings.setwise import SetwiseReranker
from jointrank.model.reproducings.sliding import SlidingWindorReranker
from jointrank.model.reproducings.topdown import TDPartRanker
from jointrank.model.reproducings.tourrank import TourRanker
from jointrank.model.reranker_base import Reranker


def get_doc_relevance(gold: list[TRECGoldRelevance], qid: str, docid: str) -> int:
    g = next((g for g in gold if g.query_idx == qid and g.document_idx == docid), None)
    if g is None:
        return 0
    return g.relevance


class OracleRanker(Reranker):
    def __init__(self, dataset: list[TRECDevRerankerSample], gold: list[TRECGoldRelevance]) -> None:
        self.dataset = dataset
        self.gold = gold

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        sample = next(
            sample for sample in self.dataset
            if sample.query.content == query
            # and [doc.as_str() for doc in sample.documents] == candidates
          )
        documents = sample.documents.copy()
        documents = [
            documents.pop(next(i for i, d in enumerate(documents) if d.as_str() == candidate))
            for candidate in candidates
        ]
        # documents = [doc for candidate in candidates for doc in sample.documents if doc.as_str() == candidate]
        assert len(documents) == len(candidates), f"{len(documents)} != {len(candidates)}"
        relevances = [get_doc_relevance(self.gold, sample.query.idx, doc.idx) for doc in documents]
        return sorted(range(len(relevances)), key=lambda x: relevances[x], reverse=True)

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        raise NotImplementedError


def get_reranker(  # noqa: PLR0911
    llm: LLMEngine,
    cfg: RunConfig,
    dataset: list[TRECDevRerankerSample],
    gold: list[TRECGoldRelevance],
) -> Reranker:
    if cfg.ranker.name == "oracle":
        return OracleRanker(dataset, gold)
    if cfg.ranker.name == "listwise":
        return get_listwise(llm, cfg.ranker.listwise, logits=cfg.ranker.logits)
    if cfg.ranker.name == "sliding":
        return SlidingWindorReranker(
            get_listwise(llm, cfg.ranker.listwise, logits=cfg.ranker.logits),
            cfg.ranker.block_size,
            cfg.ranker.window_step
        )
    if cfg.ranker.name == "setwise":
        return SetwiseReranker(llm, cfg.ranker.block_size)
    if cfg.ranker.name == "tourrank":
        return TourRanker(llm, cfg.ranker.num_tournaments, shuffle=False)
    if cfg.ranker.name == "joint":
        return get_joint(llm, cfg.ranker)
    if cfg.ranker.name == "topdown":
        base = get_listwise(llm, cfg.ranker.listwise, logits=cfg.ranker.logits)
        return TDPartRanker(base, window_size=cfg.ranker.block_size)
    raise ValueError("Unknown configuration")


def get_joint(
    llm: LLMEngine,
    cfg: RankerConfig
) -> JointRanker:
        base_reranker = get_listwise(llm, cfg.listwise, logits=cfg.logits)
        if cfg.replicas != 2:
            design = BlockDesignCollection(
                EquiReplicateBlockDesign(cfg.replicas),
                ReplicateBlockDesign(cfg.replicas)
            )
        else:
            design = BlockDesignCollection(
                LatinSquarePBIBD(),
                EquiReplicateBlockDesign(cfg.replicas),
                ReplicateBlockDesign(cfg.replicas)
            )
        return JointRanker(design, base_reranker, TournamentAggregator("pagerank"), cfg.block_size)


def get_listwise(llm: LLMEngine, typ: str, *, logits: bool) -> Reranker:
    if typ == "rankgpt":
        assert not logits
        return RankGPT(llm)
    if typ == "custom":
        return ListwiseRankerLogits(llm) if logits else ListwiseRanker(llm)

    raise ValueError(typ)
