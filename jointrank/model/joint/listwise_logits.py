import asyncio
from typing import ClassVar
from jointrank.model.llm.base import LLMEngine, Message, PromptTemplate
from jointrank.model.reranker_base import Reranker

RAW_PROMPT = """
Given a query and a documents, you must select the most relevant document for the query.

Here is a cheatsheet for the different categories of relevance:

- document is dedicated to the query and contains the exact answer, directly and comprehensively addressing it with clear and explicit information.
- document answers the query but may be indirect, incomplete, or requires some inference; relevant information might be mixed with extraneous details.
- document seems related to the query but does not answer it.
- document has nothing to do with the query.

Your input is:

QUERY: {query}
DOCUMENTS:

{snippets}

[INSTRUCTIONS]
Split this task into steps:
1. **Understand the Intent (I):** Consider the underlying intent the search, such as solving a specific problem, or answering a question.
2. **Evaluate Match (M):** Measure how well the snippet content matches the likely intent of the query.
   - Keep in mind that you are shown the snippet from the document, which may be incomplete, truncated or include auxiliary text not directly related to the main query.
   - Consider synonyms, paraphrases, and implicit information that may not directly match the query's wording but still provide relevant answers.
   - Focus on the overall context and meaning of the document. Evaluate whether the document addresses the intent of the query in a meaningful way, beyond mere keyword matching.
3. **Determine the Final Position (P):** Having relevance for each snippet established, find best order of snippets, from most relevant to least relevant.
   - If uncertain, revisit the intent and matching steps to ensure your assessment is accurate.
IMPORTANT: You MUST output ONLY index of the most relevant snippet as single character. Do not include any explanations, reasoning, or additional text in your output.
In case when there are no relevant snippets found, select any snippet.
[/INSTRUCTIONS]
All snippet ids: {snippet_ids}
The most relevant snippet is:
""".strip()  # noqa: E501

LOTGITS_PROMPT_TEMPLATE = PromptTemplate().user(RAW_PROMPT)


class ListwiseRankerLogits(Reranker):
    CHARACTERS: ClassVar = [
        "A", "B", "C", "D", "E", "F", "G", "H",
        "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T", "U", "V", "W"
    ]

    def __init__(self, llm: LLMEngine) -> None:
        self.llm = llm

    def prepare_messages(self, query: str, candidates: list[str], identifiers: list[str]) -> list[Message]:
        candidates_str = "\n\n".join(
            f"Snippet {identifiers[i]}\n```{content}```" for i, content in enumerate(candidates)
        )
        snippet_ids = " ".join(identifiers)

        return LOTGITS_PROMPT_TEMPLATE.format(query=query, snippets=candidates_str, snippet_ids=snippet_ids)

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        return asyncio.run(self._rerank_core(query, candidates))

    async def _get_scores(self, query: str, candidates: list[str]) -> list[float]:
        identifiers = (
            self.CHARACTERS if len(candidates) <= len(self.CHARACTERS)
            else list(map(str, range(1, len(candidates) + 1)))
        )
        messages = self.prepare_messages(query, candidates, identifiers)
        return await self.llm.get_logprobs(messages, identifiers)

    async def _rerank_core(self, query: str, candidates: list[str]) -> list[int]:
        scores = await self._get_scores(query, candidates)
        return [idx for idx, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    async def _rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        return await asyncio.gather(*[
            self._rerank_core(query, candidates)
            for query, candidates in batch
        ])

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        return asyncio.run(self._rerank_batch(batch))
