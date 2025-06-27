import json
import logging
from dataclasses import dataclass
from pathlib import Path

import ir_datasets
from pyserini.search._base import get_qrels, get_topics
from pyserini.search.lucene import LuceneSearcher

from jointrank.evaluation.cfg import DatasetConfig

LOG = logging.getLogger(__name__)


@dataclass
class TRECDocument:
    idx: str
    # url: str
    # title: str
    content: str

    def as_str(self) -> str:
        # return f"{self.url}\n{self.title}\n{self.content}"
        return self.content


@dataclass
class TRECQuery:
    idx: str
    content: str


@dataclass
class TRECDevRerankerSample:
    query: TRECQuery
    documents: list[TRECDocument]


@dataclass
class TRECGoldRelevance:
    query_idx: str
    document_idx: str
    relevance: int


def prepare_ir_datasets_data(d_cfg: DatasetConfig) -> tuple[list[TRECDevRerankerSample], list[TRECGoldRelevance]]:
    dataset = ir_datasets.load(d_cfg.dataset)
    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = TRECQuery(query.query_id, query.text)
    gold: list[TRECGoldRelevance] = []
    # print(type(dataset), dataset.has_qrels())
    for qrel in dataset.qrels_iter():
        gold.append(TRECGoldRelevance(qrel.query_id, qrel.doc_id, qrel.relevance))
    docstore = dataset.docs_store()

    to_rerank_raw: dict[str, list[tuple[str, int]]] = {}
    LOG.debug("reading reranker samples")

    for line in Path(d_cfg.run_path).read_text().splitlines():
        qid, _, docid, rank, *_ = line.split()
        to_rerank_raw.setdefault(qid, []).append((docid, int(rank)))

    to_rerank: list[TRECDevRerankerSample] = []

    for qid, candidates in to_rerank_raw.items():
        doc2rank = dict(candidates)
        # sorted_candidates = [docid for docid, _ in sorted(candidates, key=lambda x: x[1])]
        docs = [TRECDocument(d.doc_id, d.text) for d in docstore.get_many_iter(doc2rank.keys())]
        docs = sorted(docs, key=lambda x: doc2rank[x.idx])

        to_rerank.append(TRECDevRerankerSample(
            queries[qid],
            docs
        ))

    LOG.debug(
        "sample query %s %s 1st doc %s %s",
        to_rerank[0].query.idx,
        to_rerank[0].query.content,
        to_rerank[0].documents[0].idx,
        to_rerank[0].documents[0].content
    )

    return to_rerank, gold


def prepare_pyserini_index_data(
    d_cfg: DatasetConfig
) -> tuple[list[TRECDevRerankerSample], list[TRECGoldRelevance]]:
    assert d_cfg.pyserini_index is not None
    topics = get_topics(d_cfg.pyserini_index + "-test")
    qrels = get_qrels(d_cfg.pyserini_index + "-test")
    gold = []
    for qid, query_qrels in qrels.items():
        for docid, relevance in query_qrels.items():
            gold.append(TRECGoldRelevance(str(qid), str(docid), int(relevance)))

    searcher = LuceneSearcher.from_prebuilt_index(d_cfg.pyserini_index + ".flat")
    assert searcher is not None

    queries = {}
    for query_id, query in topics.items():
        queries[str(query_id)] = TRECQuery(str(query_id), query["title"])

    to_rerank_raw: dict[str, list[tuple[str, int]]] = {}
    LOG.debug("reading reranker samples")

    for line in Path(d_cfg.run_path).read_text().splitlines():
        qid, _, docid, rank, *_ = line.split()
        to_rerank_raw.setdefault(qid, []).append((docid, int(rank)))

    to_rerank: list[TRECDevRerankerSample] = []

    for qid, candidates in to_rerank_raw.items():
        doc2rank = dict(candidates)
        docs = []
        for docid in doc2rank:
            doc = searcher.doc(docid)
            assert doc is not None
            data = json.loads(doc.raw())
            text = data["text"]
            if "title" in data:
                text = f'{data["title"]}\n{text}'
            docs.append(TRECDocument(docid, text))
        docs = sorted(docs, key=lambda x: doc2rank[x.idx])

        to_rerank.append(TRECDevRerankerSample(
            queries[qid],
            docs
        ))

    return to_rerank, gold


def prepare_dataset(d_cfg: DatasetConfig) -> tuple[list[TRECDevRerankerSample], list[TRECGoldRelevance]]:
    if d_cfg.pyserini_index is not None:
        return prepare_pyserini_index_data(d_cfg)
    return prepare_ir_datasets_data(d_cfg)
