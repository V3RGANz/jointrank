import logging
import re
from typing import ClassVar

from jointrank.model.llm.base import LLMEngine
from jointrank.model.reranker_base import Reranker

LOG = logging.getLogger(__name__)


SYS_PROMPT = ("You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage "
              "from a pool of passages based on their relevance to the query.")

USR_PROPMT = """
Given a query "{query}", which of the following passages is the most relevant one to the query?

{passages}

Output only the passage label of the most relevant passage.
""".strip()


class SetwiseReranker(Reranker):
    """Setwise Reranker.

    https://arxiv.org/pdf/2310.09497.pdf
    Adapted implementation. original:
    https://github.com/ielab/llm-rankers/blob/main/llmrankers/setwise.py
    """

    CHARACTERS: ClassVar = [
        "A", "B", "C", "D", "E", "F", "G", "H",
        "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T", "U", "V", "W"
    ]

    def __init__(self, llm: LLMEngine, num_child: int = 10, k: int = 10) -> None:
        self.llm = llm
        self.num_child = num_child
        self.k = k

    def compare(self, query: str, candidates: list[tuple[int, str]], identifiers: list[str]) -> str:
        passages = "\n\n".join([f'Passage {identifiers[i]}: "{doc[1]}"' for i, doc in enumerate(candidates)])

        output = self.llm.get_response(
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": USR_PROPMT.format(query=query, passages=passages)},
            ],
        )

        matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
        output = matches[-1][8] if matches else output.strip().removeprefix("Passage").strip()
        if output in identifiers:
            pass
        else:
            LOG.error("Unexpected output: %s", output)
            output = identifiers[0]
        return output

    def heapify(self, arr: list[tuple[int, str]], n: int, i: int, query: str) -> None:
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i, *list(range(self.num_child * i + 1, min(self.num_child * (i + 1) + 1, n)))]
            identifiers = (
                self.CHARACTERS if len(docs) <= len(self.CHARACTERS)
                else list(map(str, range(1, len(docs) + 1)))
            )
            output = self.compare(query, docs, identifiers)
            try:
                best_ind = identifiers.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heap_sort(self, arr: list[tuple[int, str]], query: str, k: int) -> None:
        n = len(arr)
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for ranked, i in enumerate(range(n - 1, 0, -1), 1):
            arr[i], arr[0] = arr[0], arr[i]
            if ranked == k:
                break
            self.heapify(arr, i, 0, query)

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        ranking = [(i, doc) for i, doc in enumerate(candidates)]

        self.heap_sort(ranking, query, self.k)
        ranking = list(reversed(ranking))

        results = [i for i, _ in ranking]
        results += [i for i in range(len(candidates)) if i not in results]

        return results

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        raise NotImplementedError
