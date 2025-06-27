import asyncio
import logging
import threading
import time
from dataclasses import dataclass

import httpx
import openai
import together
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from jointrank.model.llm.base import LLMEngine, Message

LOG = logging.getLogger(__name__)


@dataclass
class Telemetry:
    inferences: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    inference_total_time: float = 0.
    total_time: float = 0.


class LLMEngineWithTelemetry(LLMEngine):
    async def get_response_async(self, messages: list[Message]) -> str:
        start = time.time()
        response = await self.core_engine.get_response_async(messages)
        end = time.time()
        with self.lock:
            self.telemetry.inference_total_time += end - start
            self.telemetry.inferences += 1
            self.telemetry.input_tokens += self.core_engine.count_tokens_messages(messages)
            self.telemetry.output_tokens += self.core_engine.count_tokens(response)
        return response

    def __init__(self, core_engine: LLMEngine) -> None:
        self.core_engine = core_engine
        self.telemetry = Telemetry()
        self.lock = threading.Lock()

    def get_response(self, messages: list[Message]) -> str:
        start = time.time()
        response = self.core_engine.get_response(messages)
        end = time.time()
        with self.lock:
            self.telemetry.inference_total_time += end - start
            self.telemetry.inferences += 1
            self.telemetry.input_tokens += self.core_engine.count_tokens_messages(messages)
            self.telemetry.output_tokens += self.core_engine.count_tokens(response)
        return response

    @retry(
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(LOG, log_level=logging.INFO),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((openai.APIConnectionError,
                                       openai.RateLimitError,
                                       asyncio.TimeoutError,
                                       httpx.RemoteProtocolError,
                                       openai.InternalServerError))
    )
    async def get_logprobs(self, messages: list[Message], tokens: list[str]) -> list[float]:
        start = time.time()
        response = await self.core_engine.get_logprobs(messages, tokens)
        end = time.time()
        with self.lock:
            self.telemetry.inference_total_time += end - start
            self.telemetry.inferences += 1
            self.telemetry.input_tokens += self.core_engine.count_tokens_messages(messages)
            self.telemetry.output_tokens += 1
        return response

    @retry(
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(LOG, log_level=logging.INFO),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        retry=retry_if_exception_type((
            together.error.RateLimitError,
        ))
    )
    @retry(
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(LOG, log_level=logging.INFO),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APIError,
            openai.RateLimitError,
            asyncio.TimeoutError,
            httpx.RemoteProtocolError,
            openai.InternalServerError
        ))
    )
    def get_response_batch(self, batch: list[list[Message]]) -> list[str]:
        start = time.time()
        response = self.core_engine.get_response_batch(batch)
        end = time.time()
        with self.lock:
            self.telemetry.inference_total_time += end - start
            self.telemetry.inferences += len(batch)
            self.telemetry.input_tokens += sum(self.core_engine.count_tokens_messages(messages) for messages in batch)
            self.telemetry.output_tokens += sum(self.core_engine.count_tokens(resp) for resp in response)
        return response
