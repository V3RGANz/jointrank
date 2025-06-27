from functools import cached_property

import tiktoken

from jointrank.model.llm.base import Message

TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1


class TikTokenCounter:
    def __init__(self, model: str) -> None:
        self.model = model

    @cached_property
    def _tokenizer(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self.model)

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
