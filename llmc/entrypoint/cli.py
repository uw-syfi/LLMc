import argparse
import asyncio
import os

import uvicorn

from llmc.entrypoint.compress import compress as compress_task
from llmc.entrypoint.decompress import decompress as decompress_task


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmc", description="LLM compression toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    p_compress = sub.add_parser("compress", help="Compress a text file")
    p_compress.add_argument("input", help="Input text file path ('-' for stdin)")
    p_compress.add_argument("output", help="Output .pkl.gz path")
    p_compress.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model name")
    p_compress.add_argument("--threshold", type=_positive_int, default=256, help="Threshold")
    p_compress.add_argument("--chunk-size", type=_positive_int, default=4096, dest="chunk_size", help="Chunk size")
    p_compress.add_argument("--gpu-mem", type=float, default=0.5, dest="gpu_memory_utilization", help="GPU memory utilization (0-1)")

    # decompress
    p_decompress = sub.add_parser("decompress", help="Decompress a .pkl.gz file")
    p_decompress.add_argument("input", help="Input .pkl.gz path")
    p_decompress.add_argument("output", help="Output text file ('-' for stdout)")
    p_decompress.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model name")
    p_decompress.add_argument("--threshold", type=_positive_int, default=256, help="Threshold")
    p_decompress.add_argument("--chunk-size", type=_positive_int, default=4096, dest="chunk_size", help="Chunk size")
    p_decompress.add_argument("--gpu-mem", type=float, default=0.5, dest="gpu_memory_utilization", help="GPU memory utilization (0-1)")

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI server")
    p_serve.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model name")
    p_serve.add_argument("--max-threshold", type=_positive_int, default=256, help="Max threshold")
    p_serve.add_argument("--max-chunk-size", type=_positive_int, default=4096, help="Max chunk size")
    p_serve.add_argument("--gpu-mem", type=float, default=0.5, dest="gpu_memory_utilization", help="GPU memory utilization (0-1)")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host to bind")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to bind")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "compress":
        asyncio.run(
            compress_task(
                input_path=args.input,
                output_path=args.output,
                model=args.model,
                threshold=args.threshold,
                chunk_size=args.chunk_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )
        return

    if args.command == "decompress":
        asyncio.run(
            decompress_task(
                input_path=args.input,
                output_path=args.output,
                model=args.model,
                threshold=args.threshold,
                chunk_size=args.chunk_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        )
        return

    if args.command == "serve":
        # Set env vars consumed by llmc.entrypoint.server
        os.environ["LLMC_MODEL"] = args.model
        os.environ["LLMC_MAX_MODEL_LEN"] = str(args.max_chunk_size)
        os.environ["LLMC_MAX_THRESHOLD"] = str(args.max_threshold)
        os.environ["LLMC_GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)

        uvicorn.run("llmc.entrypoint.server:app", host=args.host, port=args.port, reload=False)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
