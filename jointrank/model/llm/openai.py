import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import suppress
from functools import cached_property

import tiktoken
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_token_logprob import TopLogprob

from jointrank.model.llm.base import LLMEngine, Message

LOG = logging.getLogger(__name__)

TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1


class OpenAILLMEngine(LLMEngine):
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def get_response(self, messages: list[Message]) -> str:
        return asyncio.run(self._collect_response(self._complete_chat_streaming(messages)))

    async def get_response_async(self, messages: list[Message]) -> str:
        return await self._collect_response(self._complete_chat_streaming(messages))

    def get_response_batch(self, batch: list[list[Message]]) -> list[str]:
        async def collect_batch_responses() -> list[str]:
            tasks = [self._complete_chat_streaming(messages) for messages in batch]
            return await asyncio.gather(*[self._collect_response(task) for task in tasks])

        return asyncio.run(collect_batch_responses())

    async def get_logprobs(self, messages: list[Message], tokens: list[str]) -> list[float]:
        assert len(tokens) > 0

        token_ids = []
        for token in tokens:
            tokenized = self._tokenizer.encode(token)
            token_ids.append(tokenized[0])
        assert len(set(token_ids)) == len(token_ids), "provided tokens do not satisfy prefix-free property"

        logit_bias: dict[int, int] = dict.fromkeys(token_ids, 100)

        api_response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            stream=False,
            timeout=None,
            max_tokens=1,
            n=1,
            temperature=1.0,
            logprobs=True,
            top_logprobs=20,
            logit_bias=logit_bias,  # type: ignore[arg-type]
        )

        result = [float("-inf") for _ in tokens]
        top_logprobs: list[TopLogprob] = api_response.choices[0].logprobs.content[0].top_logprobs  # type: ignore

        for log_prob in top_logprobs:
            with suppress(ValueError):
                result[tokens.index(log_prob.token)] = log_prob.logprob
        return result

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def count_tokens_messages(self, messages: list[Message]) -> int:
        total = 0
        for message in messages:
            if "name" in message:
                total += TOKENS_PER_NAME
            total += TOKENS_PER_MESSAGE
            total += self.count_tokens(message["content"])
        return total

    async def _collect_response(self, stream: AsyncGenerator[str, None]) -> str:
        response = ""
        async for chunk in stream:
            response += chunk
        return response

    async def _complete_chat_streaming(
        self,
        messages: list[Message],
    ) -> AsyncGenerator[str, None]:
        async for chunk in await self._get_stream(messages):
            if (content := chunk.choices[0].delta.content) is not None:
                yield content

    async def _get_stream(self, messages: list[Message]) -> AsyncStream[ChatCompletionChunk]:
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )


    @cached_property
    def _tokenizer(self) -> tiktoken.Encoding:
        model = self.model.split("/")[-1]
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # LOG.exception("can't find model tokenizer, using gpt-4o tokenizer")
            return tiktoken.encoding_for_model("gpt-4o")
