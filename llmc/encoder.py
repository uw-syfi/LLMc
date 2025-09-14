import asyncio
from typing import AsyncGenerator, Literal
from llmc.executor import Executor
from vllm.sampling_params import SamplingParams, RequestOutputKind, CompressionMode
import numpy as np
from pyfastpfor import getCodec


async def encode_to_ids(tokens: list[int], threshold: int) -> list[int]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenize=False,
        compression_mode=CompressionMode.ENCODE,
        compressed_ids=None,
        threshold=threshold,
    )

    last_output = None
    engine = await Executor.instance()
    async for req_out in engine.generate(tokens, sampling_params):
        last_output = req_out

    assert last_output is not None, "No output received from vLLM engine"
    prompt_ids = last_output.prompt_token_ids or []
    assert len(prompt_ids) >= 1, "Empty prompt_token_ids from engine"

    prompt_logprobs = last_output.prompt_logprobs or []
    compressed_ids = [int(prompt_ids[0])]
    for i in range(1, len(prompt_ids)):
        logprob_dict = prompt_logprobs[i]
        token_id = int(prompt_ids[i])
        if logprob_dict is None:
            raise RuntimeError(f"Missing prompt logprobs at position {i}")
        info = logprob_dict.get(token_id)
        if info is not None and info.rank is not None and info.rank <= threshold:
            compressed_ids.append(int(info.rank) - 1)
        else:
            compressed_ids.append(token_id + threshold)

    return compressed_ids


def compress_ids(ids: list[int]) -> np.ndarray:
    ids_np = np.array(ids, dtype=np.uint32, order="C")
    buffer_np = np.zeros(ids_np.size * 2, dtype=np.uint32, order="C")
    codec = getCodec("simdbinarypacking")
    compressed_size = codec.encodeArray(
        ids_np, len(ids_np), buffer_np[1:], len(buffer_np[1:])
    )
    buffer_np[0] = len(ids)
    return buffer_np[: compressed_size + 1]


async def encode_chunk(tokens: list[int], threshold: int = 256) -> np.ndarray:
    compressed_ids = await encode_to_ids(tokens, threshold)
    return compress_ids(compressed_ids)


EncodeResult = tuple[Literal["total"], int] \
    | tuple[Literal["finished"], int] \
    | tuple[Literal["result"], list[np.ndarray]]


async def encode_text(
    text: str, threshold: int = 256, chunk_size: int = 4096
) -> AsyncGenerator[
    EncodeResult,
    None,
]:
    engine = await Executor.instance()
    tokens = engine.encode(text)
    async_results: list[asyncio.Task[np.ndarray]] = []
    for i in range(0, len(tokens), chunk_size):
        async_results.append(
            asyncio.create_task(
                encode_chunk(
                    tokens[i : min(i + chunk_size, len(tokens))],
                    threshold,
                )
            )
        )
    total = (len(tokens) + chunk_size - 1) // chunk_size
    yield ("total", total)
    counter = 0
    for future in asyncio.as_completed(async_results):
        await future
        counter += 1
        yield ("finished", counter)
    yield ("result", [future.result() for future in async_results])
