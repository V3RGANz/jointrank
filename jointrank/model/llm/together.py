from collections.abc import AsyncGenerator

from together import AsyncTogether
from together.types import ChatCompletionChunk
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from jointrank.model.llm.abcasync import AsyncAPILLMEngine
from jointrank.model.llm.base import Message


class TogetherEngine(AsyncAPILLMEngine):
    def __init__(
        self,
        client: AsyncTogether | None = None,
        model: str = "mistralai/Mistral-Small-24B-Instruct-2501",
        max_tokens: int | None = None
    ) -> None:
        self.client = client or AsyncTogether()
        self.model = model
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model)
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_tokens_messages(self, messages: list[Message]) -> int:
        return len(self.tokenizer.apply_chat_template(messages))

    async def _complete_chat_streaming(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        stream: AsyncGenerator[ChatCompletionChunk, None] = await self.client.chat.completions.create(  # type: ignore
            model=self.model,
            messages=messages,
            stream=True,
            max_tokens=self.max_tokens,
        )
        async for chunk in stream:
            if chunk.choices is None or len(chunk.choices) == 0 or chunk.choices[0].delta is None:
                continue
            if (content := chunk.choices[0].delta.content) is not None:
                yield content

    async def get_logprobs(self, messages: list[Message], tokens: list[str]) -> list[float]:
        raise NotImplementedError
