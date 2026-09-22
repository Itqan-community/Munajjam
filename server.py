import asyncio
import gc
import os
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Munajjam API Server")

# Allow connections from any frontend (supports Colab & Cloudflare tunnel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dictionary to store background job state
jobs: dict = {}
# Single-thread executor to prevent concurrent GPU execution / VRAM thrashing
_executor = ThreadPoolExecutor(max_workers=1)


# Integration boundary for HybridQuranAligner (Issue #120).
# DO NOT modify this function's signature — it will be replaced entirely by #120.
# This stub fails explicitly if #120 is not available; no fallback to legacy code.
async def _run_hybrid_alignment(
    job_id: str,
    audio_path: str,
    params: dict,
    progress_callback: Callable[[int, int], None],
) -> Any:
    """
    Integration point for HybridQuranAligner (Issue #120).
    This stub will be REPLACED by #120 implementation.
    Fails explicitly if #120 not available — no legacy fallback.
    """
    try:
        from munajjam.core.hybrid_aligner import HybridQuranAligner
    except ImportError:
        raise ImportError(
            "HybridQuranAligner not available. Requires Issue #120."
        )

    # Call signature, return type, internal flow DEFINED BY #120.
    # This stub does not guess them. Real impl replaces entire function.
    raise NotImplementedError("Waiting for HybridQuranAligner from Issue #120")


# Response formatters (isolated, adaptable for #120 integration)

def _safe_get(obj: Any, *names: str):
    """Safely get first non-None attribute from obj, returning None if all missing."""
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return None


def _format_ayah_segment(result: Any) -> dict:
    """Format a single ayah alignment result for API response."""
    # Safe attribute access with explicit None checks (0.0 is valid timestamp).
    # Adapts to whatever #120 returns — single change point.
    ayah_number = _safe_get(result, "ayah_number")
    if ayah_number is None:
        ayah = getattr(result, "ayah", None)
        if ayah is not None:
            ayah_number = getattr(ayah, "ayah_number", None)

    start_time = _safe_get(result, "start_time", "start")
    end_time = _safe_get(result, "end_time", "end")
    text = _safe_get(result, "text", "transcribed_text")
    if text is None:
        ayah = getattr(result, "ayah", None)
        if ayah is not None:
            text = getattr(ayah, "text", None)

    words = getattr(result, "words", None)
    formatted_words = None
    if words:
        formatted_words = [
            {
                "word": getattr(w, "word", ""),
                "start": getattr(w, "start", 0.0),
                "end": getattr(w, "end", 0.0),
                "probability": getattr(w, "probability", 0.0),
            }
            for w in words
        ]

    return {
        "ayah_number": ayah_number,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "words": formatted_words,
        "pause_duration": getattr(result, "pause_duration", None),
        "is_breath_boundary": getattr(result, "is_breath_boundary", False),
    }


def _format_breath_groups(breath_boundaries: list) -> list[dict]:
    """Format breath boundaries for API response."""
    # Schema TBD by #120; assumes start_sec/end_sec/duration_sec (matching BreathBoundary).
    if not breath_boundaries:
        return []
    formatted = []
    for b in breath_boundaries:
        start = _safe_get(b, "start_sec", "start")
        end = _safe_get(b, "end_sec", "end")
        duration = _safe_get(b, "duration_sec", "duration")
        if start is not None and end is not None:
            formatted.append({"start_sec": start, "end_sec": end, "duration_sec": duration})
    return formatted


def _extract_breath_boundaries(alignment_result: Any) -> list:
    """Extract breath boundaries from alignment result."""
    # INTEGRATION BOUNDARY: #120 return structure not yet defined.
    # Placeholder — replaced when #120 lands. Currently returns empty list.
    # When #120 implemented, replace with actual extraction logic.
    # DO NOT assume final API (e.g., breath_boundaries attribute).
    return []


def _build_success_response(alignment_result: Any) -> dict:
    """Build success response matching Issue #121 schema: {data: [...], breath_groups: [...]}."""
    ayahs = []
    if hasattr(alignment_result, "__iter__"):
        for r in alignment_result:
            ayahs.append(_format_ayah_segment(r))

    # Breath groups — integration point for #120
    breath_boundaries = _extract_breath_boundaries(alignment_result)

    return {"data": ayahs, "breath_groups": _format_breath_groups(breath_boundaries)}


# Background job processing

def _process_alignment_job(
    job_id: str,
    audio_path: str,
    surah_id: int,
    method: str,
    riwaya: str,
    chunk_duration: float,
    min_silence_ms: int,
    min_speech_ms: int,
    pad_ms: int,
    repetition_attach: bool,
) -> None:
    """Background task: transcribe + align + store result."""
    params = {
        "surah_id": surah_id,
        "method": method,
        "riwaya": riwaya,
        "chunk_duration": chunk_duration,
        "min_silence_ms": min_silence_ms,
        "min_speech_ms": min_speech_ms,
        "pad_ms": pad_ms,
        "repetition_attach": repetition_attach,
    }

    def progress_cb(current: int, total: int):
        jobs[job_id]["progress"] = int((current / total) * 100)
        jobs[job_id]["message"] = f"Processing ayah {current}/{total}"

    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 0
        jobs[job_id]["message"] = "Starting alignment..."

        # Call integration boundary — real impl from #120
        alignment_result = asyncio.run(
            _run_hybrid_alignment(
                job_id=job_id,
                audio_path=audio_path,
                params=params,
                progress_callback=progress_cb,
            )
        )

        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Alignment completed"
        response = _build_success_response(alignment_result)
        jobs[job_id]["data"] = response["data"]
        jobs[job_id]["breath_groups"] = response["breath_groups"]

    except ImportError as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = "Alignment engine not available"
    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Alignment failed: {e}"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        gc.collect()


# Endpoints

@app.post("/align/job/{surah_id}")
async def create_alignment_job(
    surah_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form("hybrid"),
    riwaya: str = Form("hafs"),
    chunk_duration: float = Form(30.0),
    min_silence_ms: int = Form(300),
    min_speech_ms: int = Form(100),
    pad_ms: int = Form(100),
    repetition_attach: bool = Form(False),
) -> JSONResponse:
    # ---- Validation ----
    if surah_id < 1 or surah_id > 114:
        return JSONResponse(
            {"status": "error", "message": "surah_id must be 1-114"}, status_code=400
        )
    if method != "hybrid":
        return JSONResponse(
            {"status": "error", "message": "method must be 'hybrid'"}, status_code=400
        )
    if riwaya not in ("hafs", "warsh"):
        return JSONResponse(
            {"status": "error", "message": "riwaya must be 'hafs' or 'warsh'"}, status_code=400
        )
    if chunk_duration <= 0:
        return JSONResponse(
            {"status": "error", "message": "chunk_duration must be > 0"}, status_code=400
        )
    if min_silence_ms < 100:
        return JSONResponse(
            {"status": "error", "message": "min_silence_ms must be >= 100"}, status_code=400
        )
    if min_speech_ms < 100:
        return JSONResponse(
            {"status": "error", "message": "min_speech_ms must be >= 100"}, status_code=400
        )
    if pad_ms < 0:
        return JSONResponse(
            {"status": "error", "message": "pad_ms must be >= 0"}, status_code=400
        )

    # ---- Save audio ----
    job_id = str(uuid.uuid4())
    os.makedirs("temp_audio", exist_ok=True)
    file_location = os.path.join("temp_audio", f"{job_id}_{surah_id}.mp3")

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"Failed to save audio: {e}"}, status_code=500
        )

    # ---- Initialize job ----
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "data": None,
        "error": None,
    }

    # ---- Queue background task ----
    background_tasks.add_task(
        lambda: _executor.submit(
            _process_alignment_job,
            job_id,
            file_location,
            surah_id,
            method,
            riwaya,
            chunk_duration,
            min_silence_ms,
            min_speech_ms,
            pad_ms,
            repetition_attach,
        )
    )

    return JSONResponse(
        {"status": "queued", "job_id": job_id, "message": "Job started"}
    )


@app.get("/align/status/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(
            {"status": "error", "message": "Job not found"}, status_code=404
        )

    if job["status"] == "success":
        return JSONResponse({"status": "success", "progress": 100, "message": "Completed", "data": job["data"], "breath_groups": job.get("breath_groups", [])})
    elif job["status"] == "error":
        return JSONResponse(
            {"status": "error", "progress": job["progress"], "message": job["message"], "error": job["error"]},
            status_code=500,
        )
    else:
        return JSONResponse(
            {"status": job["status"], "progress": job["progress"], "message": job["message"]}
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}