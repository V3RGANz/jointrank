import re

from ftfy import fix_text

from jointrank.model.llm.base import LLMEngine, Message, PromptTemplate
from jointrank.model.reranker_base import Reranker


def get_prefix_prompt(query: str, num: int) -> list[Message]:
    return (PromptTemplate()
     .system("You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.")
     .user("I will provide you with {num} passages, each indicated by number identifier []. \n"
           "Rank the passages based on their relevance to query: {query}.")
     .assistant("Okay, please provide the documents.")
     ).format(query=query, num=str(num))


def get_suffix_prompt(query: str, num: int) -> str:
    return (f"Search Query: {query}. \n"
            f"Rank the {num} passages above based on their relevance to the search query. "
            "The passages should be listed in descending order using identifiers. "
            "The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. "
            "Only response the ranking results, do not say any word or explain.")


def replace_number(s: str) -> str:
    return re.sub(r"\[(\d+)\]", r"(\1)", s)


class RankGPT(Reranker):
    def __init__(self, llm: LLMEngine) -> None:
        self.llm = llm

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        messages = self.prepare_messages(query, candidates)
        response = self.llm.get_response(messages)
        return self.parse_output(response, len(candidates))

    def parse_output(self, output: str, length: int) -> list[int]:
        output = re.sub(r"[^0-9]", " ", output)
        numbers = [int(x) - 1 for x in output.split()]
        numbers = list({x: 0 for x in numbers if 0 <= x < length}.keys())
        backfill = [i for i in range(length) if i not in numbers]
        return numbers + backfill

    def prepare_messages(self, query: str, candidates: list[str]) -> list[Message]:
        messages = get_prefix_prompt(query, len(candidates))
        prompt = PromptTemplate().add_messages(messages)

        for rank, candidate in enumerate(candidates, start=1):
            content = candidate.strip()
            content = fix_text(content)
            content = " ".join(content.split())
            prompt.user(f"[{rank}] {replace_number(content)}")
            prompt.assistant(f"Received passage [{rank}]")
        prompt.user(get_suffix_prompt(query, len(candidates)))
        return prompt.messages_template

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        messages_batch = [
            self.prepare_messages(query, candidates)
            for query, candidates in batch
        ]
        response_batch = self.llm.get_response_batch(messages_batch)
        return [
            self.parse_output(response, len(candidates))
            for response, (_, candidates) in zip(response_batch, batch, strict=True)
        ]
