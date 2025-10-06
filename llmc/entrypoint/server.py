import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

from vllm.engine.arg_utils import AsyncEngineArgs

from llmc.executor import Executor
from llmc.encoder import encode_text
from llmc.decoder import decode_text


def _get_engine_args_from_env() -> AsyncEngineArgs:
    model = os.environ.get("LLMC_MODEL", "")
    if not model:
        raise RuntimeError("LLMC_MODEL is required")
    max_model_len = int(os.environ.get("LLMC_MAX_MODEL_LEN", "32768"))
    gpu_memory_utilization = float(os.environ.get("LLMC_GPU_MEMORY_UTILIZATION", "0.5"))
    threshold = int(os.environ.get("LLMC_MAX_THRESHOLD", "256"))
    return AsyncEngineArgs(
        model=model,
        enforce_eager=True,
        max_logprobs=threshold,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


app = FastAPI(title="LLMC Server", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    engine_args = _get_engine_args_from_env()
    await Executor.init(engine_args)
    web_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web")
    web_dir = os.path.abspath(web_dir)
    if os.path.isdir(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")


@app.on_event("shutdown")
async def _shutdown() -> None:
    try:
        await Executor.stop()
    except RuntimeError:
        pass


@app.get("/")
async def root_index() -> RedirectResponse:
    return RedirectResponse(url="/web/")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/metrics")
async def metrics() -> Response:
    lines = [
        "# HELP app_up 1 if the app process is up",
        "# TYPE app_up gauge",
        "app_up 1",
        "# HELP app_info Static info about the app",
        "# TYPE app_info gauge",
        f'app_info{{version="{app.version}"}} 1',
    ]
    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/compress")
async def compress_endpoint(
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    threshold: int = Form(ge=1),
    chunk_size: int = Form(ge=1),
) -> Response:
    if not text and not file:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'file'.")
    if text and file:
        raise HTTPException(
            status_code=400, detail="Provide only one of 'text' or 'file'."
        )

    if file is not None:
        data = await file.read()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Uploaded file must be UTF-8 text."
            ) from exc
    assert text is not None

    original_size_bytes = len(text.encode("utf-8"))
    payload: bytes = b""
    async for result in encode_text(text, threshold=threshold, chunk_size=chunk_size):
        if result[0] == "result":
            payload = result[1]
            break
    size_bytes = len(payload)
    ratio = size_bytes / max(original_size_bytes, 1)
    headers = {
        "Content-Disposition": "attachment; filename=compressed.llmc",
        "X-Original-Size": str(original_size_bytes),
        "X-Compressed-Size": str(size_bytes),
        "X-Compression-Ratio": f"{ratio:.6f}",
    }
    return Response(content=payload, media_type="application/octet-stream", headers=headers)


@app.post("/decompress")
async def decompress_endpoint(
    file: UploadFile = File(...),
    threshold: int = Form(ge=1),
    chunk_size: int = Form(ge=1),
    download: bool = Form(default=False),
    filename: str | None = Form(default=None),
) -> Response:
    data = await file.read()
    text = ""
    async for result in decode_text(data, chunk_size=chunk_size, threshold=threshold):
        if result[0] == "result":
            text = result[1]
            break
    if download:
        name = filename or "decompressed.txt"
        headers = {"Content-Disposition": f"attachment; filename={name}"}
        return Response(
            content=text, media_type="text/plain; charset=utf-8", headers=headers
        )
    return PlainTextResponse(text)
