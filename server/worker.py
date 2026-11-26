"""Render-facing FastAPI server for the DOTNXS Inkling chat website."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictInt

from dotnxs.streamed_inkling import StreamedInklingRunner

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("dotnxs.server")

NXS_REPO_ID = os.getenv("NXS_REPO_ID", "rmaaorg/Inkling-2-Mini-NXS")
NXS_FILENAME = os.getenv("NXS_FILENAME", "Inkling-2-Mini.nxs")
CACHE_BYTES = int(os.getenv("DOTNXS_CACHE_BYTES", str(16 * 1024 * 1024)))
STORAGE_CACHE_BYTES = int(os.getenv("DOTNXS_STORAGE_CACHE_BYTES", str(5 * 1024 * 1024 * 1024)))
STORAGE_CACHE_RESERVE_BYTES = int(os.getenv("DOTNXS_STORAGE_CACHE_RESERVE_BYTES", str(256 * 1024 * 1024)))
STORAGE_CACHE_DIR = os.getenv("DOTNXS_STORAGE_CACHE_DIR") or None
MAX_CONTEXT_TOKENS = int(os.getenv("DOTNXS_MAX_CONTEXT_TOKENS", "64"))
MAX_NEW_TOKENS = int(os.getenv("DOTNXS_MAX_NEW_TOKENS", "12"))
CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://dotnxs.rmaa.pk,https://dotnxs.onrender.com,https://rmaaorg-dotnxs-demo.static.hf.space,https://rmaaorg-dotnxs-demo.hf.space,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]


class ChatRequest(BaseModel):
    input_ids: list[StrictInt] = Field(min_length=1, max_length=MAX_CONTEXT_TOKENS)


async def _load_runner(app: FastAPI) -> None:
    try:
        logger.info("Opening NXS archive %s/%s", NXS_REPO_ID, NXS_FILENAME)
        runner = await asyncio.to_thread(
            StreamedInklingRunner.from_hf_repo,
            NXS_REPO_ID,
            NXS_FILENAME,
            cache_bytes=CACHE_BYTES,
            max_context_tokens=MAX_CONTEXT_TOKENS,
            load_tokenizer=False,
            storage_cache_dir=STORAGE_CACHE_DIR,
            storage_cache_bytes=STORAGE_CACHE_BYTES,
            storage_cache_reserve_bytes=STORAGE_CACHE_RESERVE_BYTES,
            token=os.getenv("HF_TOKEN") or None,
        )
        app.state.runner = runner
        logger.info("Streamed Inkling runner is ready")
    except Exception as exc:
        app.state.runner_error = str(exc)
        logger.exception("Inkling runner initialization failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runner = None
    app.state.runner_error = None
    app.state.generation_lock = threading.Lock()
    app.state.runner_task = asyncio.create_task(_load_runner(app))
    yield
    task = app.state.runner_task
    if not task.done():
        task.cancel()
    if app.state.runner is not None:
        await asyncio.to_thread(app.state.runner.close)


app = FastAPI(
    title="DOTNXS Inkling API",
    version="0.1.0",
    description="Text-only Inkling 2 Mini inference streamed from the single NXS archive.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=3600,
)


@app.get("/health")
async def health():
    if app.state.runner is not None:
        return {
            "status": "ready",
            "input_protocol": "token_ids_v1",
            "model": NXS_REPO_ID,
            "model_revision": getattr(app.state.runner.weights.archive, "source_revision", None),
            "cache_limit_bytes": CACHE_BYTES,
            "storage_cache_limit_bytes": app.state.runner.storage_cache.limit_bytes,
            "storage_cache_used_bytes": app.state.runner.storage_cache.used_bytes,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
        }
    if app.state.runner_error:
        return {"status": "error", "detail": app.state.runner_error[:500]}
    return {"status": "loading", "model": NXS_REPO_ID}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _chat_response(payload: ChatRequest):
    runner = app.state.runner
    if runner is None:
        detail = app.state.runner_error or "Inkling is still loading; retry shortly."
        raise HTTPException(status_code=503, detail=detail, headers={"Retry-After": "30"})
    vocab_size = int(runner.model.config.vocab_size)
    if any(token_id < 0 or token_id >= vocab_size for token_id in payload.input_ids):
        raise HTTPException(status_code=422, detail="An input token ID is outside the model vocabulary.")
    if not app.state.generation_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The CPU model is busy; retry when the current reply finishes.")

    input_ids = payload.input_ids
    output_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def generate() -> None:
        try:
            for token_id in runner.stream_token_ids(input_ids, max_new_tokens=MAX_NEW_TOKENS):
                if stop_event.is_set():
                    break
                output_queue.put(("token", {"id": token_id}))
            output_queue.put(("done", {}))
        except Exception as exc:
            logger.exception("Inkling generation failed")
            output_queue.put(("error", {"message": str(exc)[:500]}))
        finally:
            output_queue.put(None)

    def start_thread() -> threading.Thread:
        thread = threading.Thread(target=generate, name="dotnxs-generation", daemon=True)
        thread.start()
        return thread

    async def event_stream():
        thread = start_thread()
        try:
            yield _sse("status", {"status": "generating"})
            while True:
                item = await asyncio.to_thread(output_queue.get)
                if item is None:
                    break
                event, data = item
                yield _sse(event, data)
                if event == "error":
                    break
        finally:
            stop_event.set()
            app.state.generation_lock.release()
            if not thread.is_alive():
                thread.join(timeout=0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/chat")
async def chat(payload: ChatRequest):
    return await _chat_response(payload)


@app.get("/v1/chat-stream")
async def chat_stream(payload: str = Query(..., max_length=12000)):
    try:
        parsed = ChatRequest.model_validate_json(payload)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid encoded chat payload.") from exc
    return await _chat_response(parsed)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.worker:app",
        host="127.0.0.1",
        port=int(os.getenv("DOTNXS_WORKER_PORT", "8001")),
    )
