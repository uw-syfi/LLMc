import pytest
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs

from llmc.executor import Executor
from llmc.encoder import encode_text
from llmc.decoder import decode_text


async def _with_executor(func, model: str, max_logprobs: int, max_model_len: int):
    engine_args = AsyncEngineArgs(
        model=model,
        enforce_eager=True,
        max_logprobs=max_logprobs,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.5,
    )
    await Executor.init(engine_args)
    try:
        return await func()
    finally:
        try:
            await Executor.stop()
        except RuntimeError:
            pass


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("max_logprobs", [256])
@pytest.mark.parametrize("chunk_size", [256])
def test_roundtrip_small_text(model: str, max_logprobs: int, chunk_size: int):
    text = "Hello world! This is a small test."

    async def _run():
        out_bytes = b""
        async for ev in encode_text(
            text, threshold=max_logprobs, chunk_size=chunk_size
        ):
            if ev[0] == "result":
                out_bytes = ev[1]
        assert out_bytes, "encoder returned empty payload"

        out_text = ""
        async for ev in decode_text(
            out_bytes, chunk_size=chunk_size, threshold=max_logprobs
        ):
            if ev[0] == "result":
                out_text = ev[1]
        assert out_text == text

    asyncio.run(
        _with_executor(_run, model=model, max_logprobs=max_logprobs, max_model_len=4096)
    )


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("max_logprobs", [256])
@pytest.mark.parametrize("chunk_size", [256])
def test_roundtrip_unicode(model: str, max_logprobs: int, chunk_size: int):
    text = "你好，世界！مرحبا بالعالم 🌍 — Café naïve"

    async def _run():
        out_bytes = b""
        async for ev in encode_text(
            text, threshold=max_logprobs, chunk_size=chunk_size
        ):
            if ev[0] == "result":
                out_bytes = ev[1]
        assert out_bytes, "encoder returned empty payload"

        out_text = ""
        async for ev in decode_text(
            out_bytes, chunk_size=chunk_size, threshold=max_logprobs
        ):
            if ev[0] == "result":
                out_text = ev[1]
        assert out_text == text

    asyncio.run(
        _with_executor(_run, model=model, max_logprobs=max_logprobs, max_model_len=4096)
    )
