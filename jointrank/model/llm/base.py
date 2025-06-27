from abc import abstractmethod
from copy import deepcopy
from typing import Required, Self, TypedDict


class Message(TypedDict, total=False):
    content: Required[str]
    role: Required[str]
    name: str
    id: str


class PromptTemplate:
    def __init__(self) -> None:
        self.messages_template: list[Message] = []

    def format(self, **kwargs: str) -> list[Message]:
        messages = deepcopy(self.messages_template)
        for m in messages:
            m["content"] = m["content"].format(**kwargs)
        return messages

    def add_messages(self, messages: list[Message]) -> Self:
        self.messages_template.extend(messages)
        return self

    def add_message(self, role: str, content: str, **kwargs: str) -> Self:
        self.messages_template.append({"role": role, "content": content, **kwargs})  # type: ignore[typeddict-item]
        return self

    def user(self, content: str) -> Self:
        return self.add_message("user", content)

    def system(self, content: str) -> Self:
        return self.add_message("system", content)

    def assistant(self, content: str) -> Self:
        return self.add_message("assistant", content)


class LLMEngine:
    @abstractmethod
    async def get_logprobs(self, messages: list[Message], tokens: list[str]) -> list[float]: ...
    @abstractmethod
    def get_response(self, messages: list[Message]) -> str: ...
    @abstractmethod
    async def get_response_async(self, messages: list[Message]) -> str: ...
    @abstractmethod
    def get_response_batch(self, batch: list[list[Message]]) -> list[str]: ...
    @abstractmethod
    def count_tokens(self, text: str) -> int: ...
    @abstractmethod
    def count_tokens_messages(self, messages: list[Message]) -> int: ...
