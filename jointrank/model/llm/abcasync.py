import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from jointrank.model.llm.base import LLMEngine, Message


class AsyncAPILLMEngine(LLMEngine, ABC):
    def get_response(self, messages: list[Message]) -> str:
        return asyncio.run(self._collect_response(self._complete_chat_streaming(messages)))

    async def get_response_async(self, messages: list[Message]) -> str:
        return await self._collect_response(self._complete_chat_streaming(messages))

    def get_response_batch(self, batch: list[list[Message]]) -> list[str]:
        async def collect_batch_responses() -> list[str]:
            tasks = [self._complete_chat_streaming(messages) for messages in batch]
            return await asyncio.gather(*[self._collect_response(task) for task in tasks])

        return asyncio.run(collect_batch_responses())

    async def _collect_response(self, stream: AsyncGenerator[str, None]) -> str:
        response = ""
        async for chunk in stream:
            response += chunk
        return response

    @abstractmethod
    async def _complete_chat_streaming(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        raise NotImplementedError
        yield ""
