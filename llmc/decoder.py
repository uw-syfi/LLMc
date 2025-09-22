import asyncio
from typing import AsyncGenerator, Literal

from llmc.executor import Executor
from vllm.sampling_params import SamplingParams, RequestOutputKind, CompressionMode
import io
import brotli
from leb128 import u as uleb


def decompress_to_ids(data: bytes) -> list[int]:
    r = io.BytesIO(brotli.decompress(data))
    vals: list[int] = []
    while True:
        try:
            v, _ = uleb.decode_reader(r)
            vals.append(int(v))
        except EOFError:
            break
    return vals


ChunkDecodeResult = (
    tuple[Literal["total"], int]
    | tuple[Literal["delta"], int]
    | tuple[Literal["result"], str]
)


async def decode_chunk_streaming(data: bytes, threshold: int = 256) -> AsyncGenerator[
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


async def decode_chunk(ids: list[int], threshold: int = 256) -> str:
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
    return engine.decode(list(last_output.outputs[0].token_ids))


DecodeResult = tuple[Literal["total"], int] \
    | tuple[Literal["finished"], int] \
    | tuple[Literal["result"], str]


async def decode_text(
    data: bytes,
    chunk_size: int,
    threshold: int,
) -> AsyncGenerator[
    DecodeResult,
    None,
]:
    async_results: list[asyncio.Task[str]] = []
    ids = decompress_to_ids(data)
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:min(start + chunk_size, len(ids))]
        async_results.append(asyncio.create_task(decode_chunk(chunk, threshold)))
    yield ("total", len(data))
    counter = 0
    for future in asyncio.as_completed(async_results):
        await future
        yield ("finished", counter)
        counter += 1
    results = [future.result() for future in async_results]
    yield ("result", "".join(results))
