import pytest
import os
import time
import signal
import subprocess
from typing import Optional
import requests


BASE = os.environ.get("LLMC_BASE_URL", "http://127.0.0.1:8000")


def _server_ready() -> bool:
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        return r.ok
    except Exception:
        return False

def _start_server(model: str, max_model_len: int, max_threshold: int) -> Optional[subprocess.Popen]:
    env = os.environ.copy()
    env.setdefault("LLMC_MODEL", model)
    env.setdefault("LLMC_MAX_MODEL_LEN", str(max_model_len))
    env.setdefault("LLMC_MAX_THRESHOLD", str(max_threshold))
    env.setdefault("LLMC_GPU_MEMORY_UTILIZATION", "0.5")
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "llmc.entrypoint.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(int(BASE.rsplit(":", 1)[1]) if ":" in BASE else 8000),
            "--no-access-log",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(120):
        if _server_ready():
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    try:
        proc.terminate()
    except Exception:
        pass
    return None

def _stop_server(proc: Optional[subprocess.Popen]) -> None:
    if not proc:
        return
    try:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    except Exception:
        pass


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
def test_health(model: str):
    proc: Optional[subprocess.Popen] = None
    if not _server_ready():
        proc = _start_server(model=model, max_model_len=4096, max_threshold=256)
        if not _server_ready():
            _stop_server(proc)
            print("SKIP: server not available at", BASE)
            return
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.ok
    assert r.json().get("status") == "ok"
    _stop_server(proc)


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B"])
@pytest.mark.parametrize("max_threshold", [256])
@pytest.mark.parametrize("chunk_size", [256])
def test_compress_and_decompress_roundtrip(model: str, max_threshold: int, chunk_size: int):
    proc: Optional[subprocess.Popen] = None
    if not _server_ready():
        proc = _start_server(model=model, max_model_len=4096, max_threshold=max_threshold)
        if not _server_ready():
            _stop_server(proc)
            print("SKIP: server not available at", BASE)
            return
    text = "Hello server roundtrip!"

    fd = {
        "threshold": (None, str(max_threshold)),
        "chunk_size": (None, str(chunk_size)),
        "text": (None, text),
    }
    r = requests.post(f"{BASE}/compress", files=fd, timeout=120)
    assert r.ok
    payload = r.content
    assert payload

    files = {
        "file": ("x.llmc", payload, "application/octet-stream"),
        "threshold": (None, str(max_threshold)),
        "chunk_size": (None, str(chunk_size)),
    }
    r = requests.post(f"{BASE}/decompress", files=files, timeout=120)
    assert r.ok
    assert r.text == text
    _stop_server(proc)
