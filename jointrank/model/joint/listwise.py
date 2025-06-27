import logging
import re

import more_itertools

from jointrank.model.llm.base import LLMEngine, Message, PromptTemplate
from jointrank.model.reranker_base import Reranker

RAW_PROMPT = """
Given a query and a documents, you must rank documents by relevance to the query.

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
IMPORTANT: You MUST output ONLY the indices of snippets separated by space, in order of relevance. Do not include any explanations, reasoning, or additional text in your output.
Do not include irrelevant snippets.
In case when there are no relevant snippets found, answer only with "NO_RELEVANT_SNIPPETS_FOUND", without mentioning any ids
[/INSTRUCTIONS]
All snippet ids: {snippet_ids}
The order of snippets is:
""".strip()  # noqa: E501

LISTWISE_PROMPT_TEMPLATE = PromptTemplate().user(RAW_PROMPT)
LOG = logging.getLogger(__name__)


class ListwiseRanker(Reranker):
    def __init__(
            self,
            llm: LLMEngine
        ) -> None:
        self.llm = llm

    def rerank(self, query: str, candidates: list[str]) -> list[int]:
        indices: list[int] | None = None
        for attempt in range(3):
            indices = self.sningle_llm_call_and_parse(query, candidates)
            if indices is not None:
                break
            logging.info("ListwiseRanker attempt %d failed", attempt)

        if indices is None:
            LOG.info("failed to retrieve/parse LLM response, returning baseline")
            return list(range(len(candidates)))

        return indices

    def rerank_batch(self, batch: list[tuple[str, list[str]]]) -> list[list[int]]:
        batch_indices: list[list[int] | None] = [None for _ in range(len(batch))]
        for attempt in range(3):
            sample_ids_for_inference = [i for i, resp in enumerate(batch_indices) if resp is None]
            current_batch = [batch[i] for i in sample_ids_for_inference]
            current_indices = self.batch_llm_call_and_parse(current_batch)
            for i, indices in zip(sample_ids_for_inference, current_indices, strict=True):
                batch_indices[i] = indices
            if all(indices is not None for indices in batch_indices):
                break
            logging.info("ListwiseRanker [batch] attempt %d failed", attempt)

        for i, indices in enumerate(batch_indices):
            if indices is None:
                LOG.info("Failed to request/parse LLM response, returning baseline result")
                batch_indices[i] = list(range(len(batch[i][1])))
        return batch_indices  # type: ignore

    def prepare_messages(self, query: str, candidates: list[str]) -> list[Message]:
        candidates_str = "\n\n".join(f"Snippet {i + 1}\n```{content}```" for i, content in enumerate(candidates))
        snippet_ids = " ".join(str(i + 1) for i in range(len(candidates)))

        return LISTWISE_PROMPT_TEMPLATE.format(query=query, snippets=candidates_str, snippet_ids=snippet_ids)

    def parse_response(self, response: str, total_candidates: int) -> list[int]:
        if response.strip().upper() == "NO_RELEVANT_SNIPPETS_FOUND":
            return []
        formatted_response = re.sub(r"[^0-9]", " ", response).strip()
        # formatted_response = response.strip().upper().removesuffix("NO_RELEVANT_SNIPPETS_FOUND").strip()
        indices = [int(x) for x in formatted_response.split()]
        if not all(1 <= i <= total_candidates for i in indices):
            LOG.error("incorrect response for %d candidates: %s", total_candidates, formatted_response)
            indices = [i for i in indices if 1 <= i <= total_candidates]
        # assert all(1 <= i <= total_candidates for i in indices)
        if len(indices) != len(set(indices)):
            LOG.warning("result contain duplicates, dedup")
            indices = list(more_itertools.unique_everseen(indices))
        return [i - 1 for i in indices]

    def sningle_llm_call_and_parse(self, query: str, candidates: list[str]) -> list[int] | None:
        try:
            messages = self.prepare_messages(query, candidates)
            response = self.llm.get_response(messages).strip()
            return self.parse_response(response, len(candidates))
        except Exception:
            LOG.exception("sningle_llm_call_and_parse failed")

        return None

    def batch_llm_call_and_parse(self, batch: list[tuple[str, list[str]]]) -> list[list[int] | None]:
        messages_batch = [self.prepare_messages(query, candidates) for query, candidates in batch]
        responses = self.llm.get_response_batch(messages_batch)

        parsed: list[list[int] | None] = []
        for (_, candidates), response in zip(batch, responses, strict=True):
            try:
                parsed.append(self.parse_response(response, len(candidates)))
            except Exception:
                LOG.exception("Failed to parse")
                parsed.append(None)
        return parsed
