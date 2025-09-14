import asyncio
import numpy as np
from typing import AsyncGenerator, Literal

from llmc.executor import Executor
from vllm.sampling_params import SamplingParams, RequestOutputKind, CompressionMode
from pyfastpfor import getCodec


def decompress_to_ids(data: np.ndarray) -> list[int]:
    size = data[0]
    buffer_np = np.zeros(size * 2, dtype=np.uint32, order="C")
    codec = getCodec("simdbinarypacking")
    decompressed_size = codec.decodeArray(
        data[1:], len(data[1:]), buffer_np, len(buffer_np)
    )
    return buffer_np[:decompressed_size].tolist()


ChunkDecodeResult = (
    tuple[Literal["total"], int]
    | tuple[Literal["delta"], int]
    | tuple[Literal["result"], str]
)


async def decode_chunk_streaming(data: np.ndarray, threshold: int = 256) -> AsyncGenerator[
    ChunkDecodeResult,
    None,
]:
    ids = decompress_to_ids(data)
    if not ids:
        yield ("result", "")
        return

    first_token_id = int(ids[0])
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=len(ids) - 1,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
        compression_mode=CompressionMode.DECODE,
        compressed_ids=ids[1:],
        threshold=threshold,
    )
    yield ("total", len(ids))
    yield ("delta", 1)

    engine = await Executor.instance()
    token_ids = [first_token_id]
    async for req_out in engine.generate([first_token_id], sampling_params):
        yield ("delta", len(req_out.outputs[0].token_ids))
        token_ids.extend(req_out.outputs[0].token_ids)

    yield ("result", engine.decode(token_ids))


async def decode_chunk(data: np.ndarray, threshold: int = 256) -> str:
    ids = decompress_to_ids(data)
    if not ids:
        return ""

    first_token_id = int(ids[0])
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=len(ids) - 1,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
        compression_mode=CompressionMode.DECODE,
        compressed_ids=ids[1:],
        threshold=threshold,
    )

    engine = await Executor.instance()
    last_output = None
    async for req_out in engine.generate([first_token_id], sampling_params):
        last_output = req_out

    assert last_output is not None
    return engine.decode(last_output.outputs[0].token_ids)


DecodeResult = tuple[Literal["total"], int] \
    | tuple[Literal["finished"], int] \
    | tuple[Literal["result"], str]


async def decode_text_streaming(
    data: list[np.ndarray],
    threshold: int = 256,
) -> AsyncGenerator[
    DecodeResult,
    None,
]:
    queue: asyncio.Queue[tuple[int, ChunkDecodeResult]] = asyncio.Queue()
    async def enqueue(idx: int, gen: AsyncGenerator[ChunkDecodeResult, None]):
        async for result in gen:
            await queue.put((idx, result))
    tasks = [
        asyncio.create_task(
            enqueue(idx, decode_chunk_streaming(chunk, threshold))
        )
        for idx, chunk in enumerate(data)
    ]
    unfinished_tasks = len(tasks)
    finished_tokens = 0
    total_tokens = []
    final = ["" for _ in range(len(tasks))]
    while unfinished_tasks > 0:
        idx, result = await queue.get()
        if result[0] == "total":
            total_tokens.append(result[1])
            if len(total_tokens) == len(tasks):
                yield ("total", sum(total_tokens))
        elif result[0] == "delta":
            finished_tokens += result[1]
            yield ("finished", finished_tokens)
        elif result[0] == "result":
            final[idx] = result[1]
            unfinished_tasks -= 1
    yield ("result", "".join(final))


async def decode_text(
    data: list[np.ndarray],
    threshold: int = 256,
) -> AsyncGenerator[
    DecodeResult,
    None,
]:
    async_results: list[asyncio.Task[str]] = []
    for chunk in data:
        async_results.append(asyncio.create_task(decode_chunk(chunk, threshold)))
    yield ("total", len(data))
    counter = 0
    for future in asyncio.as_completed(async_results):
        await future
        yield ("finished", counter)
        counter += 1
    results = [future.result() for future in async_results]
    yield ("result", "".join(results))
