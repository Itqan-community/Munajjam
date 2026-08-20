import gc
import os
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from munajjam.config import get_settings
from munajjam.transcription.whisperFactory import WhisperBackend, WhisperFactory

app = FastAPI(title="Munajjam API Server")

# Allow connections from any frontend (supports Colab & Cloudflare tunnel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Valid model sizes supported by WhisperX
VALID_MODEL_SIZES = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
}

# In-memory dictionary to store background job state
jobs: dict = {}
# Single-thread executor to prevent concurrent GPU execution / VRAM thrashing
_executor = ThreadPoolExecutor(max_workers=1)

print(
    "Initializing global WhisperX transcriber (models will be loaded lazily on first request)..."
)
global_transcriber = WhisperFactory().create_whisper(backend=WhisperBackend.WHISPERX)


def _run_job(
    job_id: str, file_location: str, surah_number: int, model_size: str | None = None
) -> None:
    """
    Background job function to execute audio transcription and ayah alignment.

    Args:
        job_id: Unique identifier for the alignment job.
        file_location: Path to temporary audio file.
        surah_number: Surah number (1-114).
        model_size: Optional WhisperX model size requested by caller.
    """
    try:
        jobs[job_id]["status"] = "processing"

        # Resolve model size for every job (defaults to configuration setting)
        target_model_size = model_size or get_settings().whisperx_model_size
        if hasattr(global_transcriber, "set_model_name"):
            print(
                f"[Job {job_id[:8]}] Resolving WhisperX model size to: {target_model_size}"
            )
            global_transcriber.set_model_name(target_model_size)

        print(
            f"[Job {job_id[:8]}] Started processing Surah {surah_number} with WhisperX ({global_transcriber.model_name})"
        )

        # Transcribe and align
        segments = global_transcriber.transcribe(file_location, surah_id=surah_number)

        response_data = []
        for segment in segments:
            ayah_data = {
                "ayah_number": segment.id,
                "start_time": segment.start,
                "end_time": segment.end,
            }
            if getattr(segment, "pause_duration", None) is not None:
                ayah_data["pause_duration"] = segment.pause_duration
            if getattr(segment, "is_breath_boundary", None) is not None:
                ayah_data["is_breath_boundary"] = segment.is_breath_boundary
            if getattr(segment, "words", None):
                ayah_data["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in segment.words
                ]
            response_data.append(ayah_data)

        jobs[job_id] = {"status": "success", "data": response_data, "error": None}
        print(f"[Job {job_id[:8]}] Completed successfully")

    except Exception as e:
        traceback.print_exc()
        jobs[job_id] = {"status": "error", "data": None, "error": str(e)}
        print(f"[Job {job_id[:8]}] Error: {e!s}")

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
        gc.collect()


@app.post("/align/{surah_number}")
async def align_audio(
    surah_number: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    riwaya: str = Form("hafs"),
    model_size: str | None = Form(None),
) -> JSONResponse:
    """
    Upload an audio file and initiate background audio-to-ayah alignment.

    Args:
        surah_number: Quran Surah number (1-114).
        background_tasks: FastAPI background task manager.
        file: Uploaded audio file (.mp3, .wav, etc.).
        riwaya: Quranic Riwaya ("hafs", "warsh").
        model_size: Optional WhisperX model size (tiny, base, small, medium, large-v1, large-v2, large-v3).

    Returns:
        JSONResponse containing job status and job_id.
    """
    if model_size and model_size not in VALID_MODEL_SIZES:
        return JSONResponse(
            {
                "status": "error",
                "message": f"Invalid model_size: '{model_size}'. Must be one of {sorted(VALID_MODEL_SIZES)}",
            },
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    os.makedirs("temp_audio", exist_ok=True)
    file_location = os.path.join("temp_audio", f"{job_id}_{surah_number}.mp3")

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {"status": "queued", "data": None, "error": None}

    background_tasks.add_task(
        lambda: _executor.submit(
            _run_job, job_id, file_location, surah_number, model_size
        )
    )

    return JSONResponse(
        {
            "status": "queued",
            "job_id": job_id,
            "message": "بدأت المهمة وسيتم فحصها تلقائياً.",
        }
    )


@app.get("/align/status/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    """
    Check the status and result of a background alignment job.

    Args:
        job_id: Unique job identifier returned by POST /align/{surah_number}.

    Returns:
        JSONResponse with job status (queued, processing, success, error) and data.
    """
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(
            {"status": "error", "message": "المهمة غير موجودة"}, status_code=404
        )

    if job["status"] == "success":
        return JSONResponse({"status": "success", "data": job["data"]})
    elif job["status"] == "error":
        return JSONResponse(
            {"status": "error", "message": job["error"]}, status_code=500
        )
    else:
        return JSONResponse(
            {"status": job["status"], "message": "المعالجة مستمرة، يرجى الانتظار..."}
        )


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
