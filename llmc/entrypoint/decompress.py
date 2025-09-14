import gzip
import pickle

from vllm.engine.arg_utils import AsyncEngineArgs

from llmc.executor import Executor
from tqdm import tqdm
from llmc.decoder import decode_text


async def decompress(
    *,
    input_path: str,
    output_path: str,
    model: str,
    threshold: int,
    chunk_size: int,
    gpu_memory_utilization: float = 0.5,
) -> None:
    engine_args = AsyncEngineArgs(
        model=model,
        enforce_eager=True,
        max_logprobs=threshold,
        max_model_len=chunk_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    await Executor.init(engine_args)
    try:
        with gzip.open(input_path, "rb") as f:
            chunks = pickle.load(f)
        output: str | None = None
        bar: tqdm | None = None
        async for result in decode_text(chunks, threshold=threshold):
            if result[0] == "total":
                bar = tqdm(total=result[1], desc="Decoding", unit="token")
            elif result[0] == "finished":
                if bar is not None:
                    bar.update(result[1])
            elif result[0] == "result":
                assert bar is not None
                bar.close()
                output = result[1]
        assert output is not None
        with open(output_path, "w") as f:
            f.write(output)
    finally:
        await Executor.stop()
