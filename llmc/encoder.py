import asyncio
import torch
from typing import AsyncGenerator, Literal
from llmc.executor import Executor
from vllm.sampling_params import SamplingParams, RequestOutputKind, CompressionMode
import numpy as np
import brotli
from leb128 import u as uleb
import logging


async def encode_chunk(tokens: list[int], threshold: int, task_id: int = 0) -> list[int]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
        detokenize=False,
        compression_mode=CompressionMode.ENCODE,
        compressed_ids=tokens,
        threshold=threshold,
    )

    last_output = None
    engine = await Executor.instance()
    logging.debug(f"Encoding chunk {task_id} with {len(tokens)} tokens")
    async for req_out in engine.generate(tokens, sampling_params):
        last_output = req_out
    logging.debug(f"Finished encoding chunk {task_id}")
    assert last_output is not None, "No output received from vLLM engine"
    prompt_ids = last_output.prompt_token_ids or []
    assert len(prompt_ids) >= 1, "Empty prompt_token_ids from engine"

    compressed_ids = [int(prompt_ids[0])]
    prompt_logprobs = last_output.prompt_logprobs
    assert prompt_logprobs is not None
    logprob_token_ids, logprobs, _ = prompt_logprobs
    # The original rank that vLLM returns does not work.
    # I don't know why.
    logprob_token_ids = logprob_token_ids[:, 1:]
    logprobs = logprobs[:, 1:]
    prompt_ids = torch.tensor(prompt_ids[1:], dtype=logprob_token_ids.dtype, device=logprob_token_ids.device)
    index = torch.argmax((logprob_token_ids == prompt_ids.unsqueeze(1)).to(torch.int32), dim=1).tolist()
    for i in range(len(prompt_ids)):
        prompt_id = prompt_ids[i]
        rank = index[i]
        if logprob_token_ids[i, rank] == prompt_id:
            compressed_ids.append(rank)
        else:
            compressed_ids.append(int(prompt_id) + threshold)

    return compressed_ids


def _encode_varint_lib(arr: np.ndarray) -> bytes:
    a = np.asarray(arr)
    return b"".join(uleb.encode(int(x)) for x in a.flat)


def varint_brotli_compress(arr: np.ndarray) -> bytes:
    return brotli.compress(_encode_varint_lib(arr), quality=11)


EncodeResult = tuple[Literal["total"], int] \
    | tuple[Literal["finished"], int] \
    | tuple[Literal["result"], bytes]


async def encode_text(
    text: str, threshold: int, chunk_size: int
) -> AsyncGenerator[
    EncodeResult,
    None,
]:
    engine = await Executor.instance()
    tokens = engine.encode(text)
    async_results: list[asyncio.Task[list[int]]] = []
    for i in range(0, len(tokens), chunk_size):
        async_results.append(
            asyncio.create_task(
                encode_chunk(
                    tokens[i : min(i + chunk_size, len(tokens))],
                    threshold,
                    i,
                )
            )
        )
    yield ("total", len(async_results))
    counter = 0
    for future in asyncio.as_completed(async_results):
        await future
        counter += 1
        yield ("finished", counter)
    concat_ids = np.concatenate([future.result() for future in async_results])
    yield ("result", varint_brotli_compress(concat_ids))
