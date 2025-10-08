import asyncio

from typing import AsyncGenerator
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.engine.protocol import EngineClient
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args as _build_client,
)


class WrappedvLLMEngine:
    def __init__(self, engine_args: AsyncEngineArgs) -> None:
        self._engine_args = engine_args
        self._model = engine_args.model
        self._context = None
        self._client: EngineClient | None = None

    @property
    def is_running(self) -> bool:
        return self._client is not None

    async def start(self) -> None:
        if self._client is not None:
            return
        self._context = _build_client(self._engine_args)
        self._client = await self._context.__aenter__()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model)

    async def stop(self) -> None:
        if self._client is None:
            return
        assert self._context is not None
        await self._context.__aexit__(None, None, None)
        self._client = None
        self._context = None
    
    def encode(self, text: str) -> list[int]:
        client = self._client
        assert client is not None
        return self._tokenizer.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(
        self,
        prompt: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[RequestOutput, None]:
        client = self._client
        assert client is not None
        rid = f"req-{random_uuid()}"
        return client.generate(
            TokensPrompt(prompt_token_ids=prompt),
            sampling_params,
            rid,
        )


class Executor:
    _instance: WrappedvLLMEngine | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def init(cls, engine_args: AsyncEngineArgs) -> None:
        async with cls._lock:
            if cls._instance is not None:
                raise RuntimeError("Executor already initialized")
            tmp = WrappedvLLMEngine(engine_args)
            try:
                await tmp.start()
            except Exception:
                cls._instance = None
                raise
            else:
                cls._instance = tmp

    @classmethod
    async def instance(cls) -> WrappedvLLMEngine:
        async with cls._lock:
            if cls._instance is None or not cls._instance.is_running:
                raise RuntimeError("Executor not initialized or not running")
            return cls._instance

    @classmethod
    async def stop(cls) -> None:
        async with cls._lock:
            if cls._instance is None:
                raise RuntimeError("Executor not initialized or not running")
            if cls._instance.is_running:
                await cls._instance.stop()
            cls._instance = None
