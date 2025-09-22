from pathlib import Path
from tqdm import tqdm

from vllm.engine.arg_utils import AsyncEngineArgs

from llmc.executor import Executor
from llmc.encoder import encode_text


def _read_text(path: str) -> str:
    if path == "-":
        import sys
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


async def compress(
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
        gpu_memory_utilization=gpu_memory_utilization,
    )
    await Executor.init(engine_args)
    try:
        text = _read_text(input_path)
        original_size_bytes = len(text.encode("utf-8"))
        output: bytes | None = None
        bar: tqdm | None = None
        async for result in encode_text(text, threshold=threshold, chunk_size=chunk_size):
            if result[0] == "total":
                bar = tqdm(total=result[1], desc="Encoding", unit="chunk")
            elif result[0] == "finished":
                assert bar is not None
                bar.update(result[1])
            elif result[0] == "result":
                assert bar is not None
                bar.close()
                output = result[1]
        assert output is not None
        with open(output_path, "wb") as f:
            f.write(output)
        size_bytes = Path(output_path).stat().st_size
        ratio = size_bytes / max(original_size_bytes, 1)
        print(f"Compression ratio: {ratio:.6f}")

    finally:
        await Executor.stop()
