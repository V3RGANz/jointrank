import asyncio
import logging
import random

import numpy as np

from jointrank.model.llm.base import LLMEngine, Message, PromptTemplate
from jointrank.model.reranker_base import Reranker

LOG = logging.getLogger(__name__)
DocId = int


def get_prefix_role_prompt(query: str, n: int, m: int) -> list[Message]:
    return (PromptTemplate()
     .system("You are an intelligent assistant that can compare multiple documents "
             "based on their relevancy to the given query.")
     .user("I will provide you with the given query and {n} documents. \n"
           "Consider the content of all the documents comprehensively and select the {m} documents that are "
           "most relevant to the given query: {query}.")
     .assistant("Okay, please provide the documents.")
     ).format(n=str(n), query=query, m=str(m))


POST_ROLE_PROMPT = """
The Query is: {query}.
Now, you must output the top {m} documents that are most relevant to the Query using the following format strictly, and nothing else. Don't output any explanation, just the following format:
Document 3, ..., Document 1
""".strip()  # noqa: E501


def get_post_role_prompt(query: str, m: int) -> str:
    return POST_ROLE_PROMPT.format(query=query, m=m)


def sort_docs(scores: dict[DocId, int]) -> list[DocId]:
    sorted_combined = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_combined]


def parse_response(response: str, n: int, groups_docid: list[DocId]) -> list[DocId]:
    answer = next((line for line in reversed(response.splitlines()) if "Document" in line), None)
    if answer is None:
        LOG.error("Incorrect response format %s", answer)
        return groups_docid
    answer = answer.split(":")[-1]
    documents = answer.split(",")
    ranked: list[int] = []
    for doc in documents:
        if "Document" not in doc and not doc.strip().isdigit():
            LOG.warning("Unexpected format of document: '%s' in answer\n%s", doc, answer)
            continue
        try:
            if "..." in doc:
                continue
            # doc_num = int(re.sub(r"[^0-9]", " ", doc).strip().split()[0])
            doc_num = int(doc.strip().strip(".").split()[-1].strip()) - 1
            if not 0 <= doc_num < n:
                raise ValueError("Unexpected document number: %d for %d documents" % (doc_num, n))
            ranked.append(doc_num)
        except (ValueError, IndexError):
            LOG.exception("ValueError occured in score. (parse_response)")
            for j in range(1, n + 1):
                if j not in ranked:
                    ranked.append(j)
                    break

    ranked_ids: list[DocId] = []
    for doc_num in ranked:
        ranked_ids.append(groups_docid[doc_num])
    return ranked_ids


class TourRanker(Reranker):
    """TourRanker: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy.

    https://arxiv.org/pdf/2406.11678
    Rewritten implementation. Original: https://github.com/chenyiqun/TourRank/blob/e4eb7c81a988b737474cdd82798ace0cbb9c1eff/TourRank_multiprocessing.py
    """

    def __init__(self, llm: LLMEngine, num_tournaments: int = 10, *, shuffle: bool = False) -> None:
        self.llm = llm
        self.num_tournaments = num_tournaments
        self.shuffle_groupwise = shuffle

    async def run_tournaments(self, query: str, docs_id: list[DocId], all_contents: dict[DocId, str]) -> list[
        dict[DocId, int]]:
        tournaments = [
            self.tournament(y, query, docs_id, all_contents)
            for y in range(self.num_tournaments)
        ]
        return await asyncio.gather(*tournaments)

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        all_contents = dict(enumerate(candidates))
        docs_id: list[DocId] = list(all_contents.keys())

        docs_score_dicts_list = asyncio.run(self.run_tournaments(query, docs_id, all_contents))

        assert self.num_tournaments == len(docs_score_dicts_list)
        global_docs_score_dict = dict.fromkeys(docs_id, 0)
        for docs_score_dict in docs_score_dicts_list:
            for docid, score in docs_score_dict.items():
                global_docs_score_dict[docid] += score
        return sort_docs(global_docs_score_dict)

    async def group_stage(
        self,
        query: str,
        stage_docs_id: list[DocId],
        all_contents: dict[DocId, str],
        docs_score_dict: dict[DocId, int],
        group_size: int,
        winners_per_group: int
    ) -> None:
        candidates = np.arange(len(stage_docs_id))
        if self.shuffle_groupwise:
            np.random.default_rng().shuffle(candidates)
        group_matrix = candidates.reshape(group_size, -1)
        docs_groups: list[list[DocId]] = group_matrix.T.tolist()

        group_results = await asyncio.gather(*[
            self.group_processing(group, query, winners_per_group, all_contents)
            for group in docs_groups
        ])
        for group_winners in group_results:
            for winner in group_winners:
                docs_score_dict[winner] += 1

    async def knockout_stage(
        self,
        query: str,
        docs_id: list[DocId],
        all_contents: dict[DocId, str],
        docs_score_dict: dict[DocId, int],
        num_winners: int,
    ) -> None:
        for winner in await self.group_processing(docs_id, query, num_winners, all_contents):
            docs_score_dict[winner] += 1

    async def tournament(
            self, y_it: int, query: str, docs_id: list[DocId], all_contents: dict[DocId, str]
    ) -> dict[DocId, int]:
        docs_score_dict: dict[DocId, int] = dict.fromkeys(docs_id, 0)

        await self.group_stage(query, docs_id, all_contents, docs_score_dict, 20, 10)
        ranked_list = sort_docs(docs_score_dict)

        await self.group_stage(query, ranked_list[: 50], all_contents, docs_score_dict, 10, 4)
        ranked_list = sort_docs(docs_score_dict)

        await self.knockout_stage(query, ranked_list[: 20], all_contents, docs_score_dict, 10)
        ranked_list = sort_docs(docs_score_dict)

        await self.knockout_stage(query, ranked_list[: 10], all_contents, docs_score_dict, 5)
        ranked_list = sort_docs(docs_score_dict)

        await self.knockout_stage(query, ranked_list[: 5], all_contents, docs_score_dict, 2)

        LOG.info("Finished %d process.", y_it + 1)
        return docs_score_dict

    async def group_processing(
        self,
        group: list[DocId],
        query: str,
        num_winners: int,
        all_contents: dict[DocId, str],
    ) -> list[DocId]:
        group = group.copy()
        random.shuffle(group)
        return await self.elimination(query, group, all_contents, num_winners)

    def prepare_messages(
        self, query: str, group: list[DocId], all_contents: dict[DocId, str], num_winners: int
    ) -> list[Message]:
        messages = get_prefix_role_prompt(query, len(group), num_winners)
        template = PromptTemplate().add_messages(messages)
        for j, doc_id in enumerate(group, start=1):
            content = all_contents[doc_id]
            template.user(f"Document {j}: {content}")
            template.assistant(f"Received Document {j}.")
        template.user(get_post_role_prompt(query, num_winners))
        return template.messages_template

    async def elimination(
        self, query: str, group: list[DocId], all_contents: dict[DocId, str], num_winners: int
    ) -> list[DocId]:
        if len(group) < num_winners:
            LOG.warning("Group size is less than m: %d < %d. Nothing to eliminate", len(group), num_winners)
            return group
        messages = self.prepare_messages(query, group, all_contents, num_winners)
        answer = await self.llm.get_response_async(messages)
        ranked_ids = parse_response(answer, len(group), group)
        if len(ranked_ids) > num_winners:
            # LOG.warning("Response contains more documents (%d) than expected (%d)", len(ranked_ids), num_winners)
            ranked_ids = ranked_ids[:num_winners]
        return ranked_ids

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        raise NotImplementedError
