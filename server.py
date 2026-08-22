import gc
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from munajjam.config import get_settings
from munajjam.transcription.ctc_segmentation import FastConformerCTCTranscriber
from munajjam.transcription.whisperFactory import WhisperBackend, WhisperFactory
from munajjam.transcription.whisperx import Whisperx

app = FastAPI(title="Munajjam API Server")

# Enable CORS for external web clients and Google Colab / tunnel frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported WhisperX model sizes
VALID_MODEL_SIZES = {
    "tiny",
    "base",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
}

# Supported alignment modes (issue #104).
# "whisperx" is the existing default; "ctc_segmentation" selects the new
# FastConformer CTC pipeline.
VALID_ALIGNMENT_MODES = {"whisperx", "ctc_segmentation"}

# Dictionary for storing background job states
jobs: dict = {}
# Single-threaded ThreadPoolExecutor to prevent GPU memory race conditions
_executor = ThreadPoolExecutor(max_workers=1)

print(
    "Initializing global WhisperX transcriber (models will be loaded lazily on first request)..."
)
global_transcriber = WhisperFactory().create_whisper(backend=WhisperBackend.WHISPERX)

# The CTC transcriber is also created lazily — no ONNX/tokenizer is loaded here.
_ctc_transcriber: FastConformerCTCTranscriber | None = None


def _get_ctc_transcriber() -> FastConformerCTCTranscriber:
    """Lazy singleton for the FastConformer CTC segmentation backend."""
    global _ctc_transcriber
    if _ctc_transcriber is None:
        settings = get_settings()
        if not settings.fastconformer_model_path:
            raise ValueError(
                "MUNAJJAM_FASTCONFORMER_MODEL_PATH is not set. "
                "Provide the path to the exported ONNX graph."
            )
        if not settings.fastconformer_tokenizer_model_path:
            raise ValueError(
                "MUNAJJAM_FASTCONFORMER_TOKENIZER_MODEL_PATH is not set. "
                "Provide the path to the SentencePiece tokenizer.model."
            )
        _ctc_transcriber = cast(
            FastConformerCTCTranscriber,
            WhisperFactory().create_whisper(backend=WhisperBackend.CTC_SEGMENTATION),
        )
    return _ctc_transcriber


def _run_job(
    job_id: str,
    file_location: str,
    surah_number: int,
    model_size: str | None = None,
) -> None:
    """Background job: WhisperX alignment (default mode)."""
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
            f"[Job {job_id[:8]}] Started processing Surah {surah_number} with WhisperX"
            f" ({cast(Whisperx, global_transcriber).model_name})"
        )

        segments = global_transcriber.transcribe(file_location, surah_id=surah_number)

        response_data = _build_response(segments)

        jobs[job_id] = {"status": "success", "data": response_data, "error": None}
        print(f"[Job {job_id[:8]}] Completed successfully")

    except Exception as e:
        import traceback

        traceback.print_exc()
        jobs[job_id] = {"status": "error", "data": None, "error": str(e)}
        print(f"[Job {job_id[:8]}] Error: {e!s}")

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
        gc.collect()


def _run_ctc_job(
    job_id: str,
    file_location: str,
    surah_number: int,
) -> None:
    """Background job: FastConformer CTC segmentation (issue #104)."""
    try:
        jobs[job_id]["status"] = "processing"

        print(f"[Job {job_id[:8]}] Started CTC segmentation of Surah {surah_number}")

        transcriber = _get_ctc_transcriber()
        segments = transcriber.transcribe(file_location, surah_id=surah_number)

        response_data = _build_response(segments)

        jobs[job_id] = {"status": "success", "data": response_data, "error": None}
        print(f"[Job {job_id[:8]}] Completed CTC segmentation successfully")

    except Exception as e:
        import traceback

        traceback.print_exc()
        jobs[job_id] = {"status": "error", "data": None, "error": str(e)}
        print(f"[Job {job_id[:8]}] CTC segmentation error: {e!s}")

    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
        gc.collect()


def _build_response(segments) -> list[dict]:
    """Build the response payload from segments (shared by both backends)."""
    response_data = []
    for segment in segments:
        ayah_data = {
            "ayah_number": segment.id,
            "start_time": segment.start,
            "end_time": segment.end,
        }
        if getattr(segment, "words", None):
            ayah_data["words"] = [
                {"word": w.word, "start": w.start, "end": w.end} for w in segment.words
            ]
        response_data.append(ayah_data)
    return response_data


@app.post("/align/{surah_number}")
async def align_audio(
    surah_number: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    riwaya: str = Form("hafs"),
    model_size: str | None = Form(None),
    alignment_mode: str = Form("whisperx"),
) -> JSONResponse:
    """
    Upload an audio file and initiate background audio-to-ayah alignment.

    Args:
        surah_number: Quran Surah number (1-114).
        background_tasks: FastAPI background task manager.
        file: Uploaded audio file (.mp3, .wav, etc.).
        riwaya: Quranic Riwaya ("hafs", "warsh").
        model_size: Optional WhisperX model size (tiny, base, small, medium,
                   large-v1, large-v2, large-v3). Ignored for ctc_segmentation.
        alignment_mode: Alignment backend — ``"whisperx"`` (default, existing
                       behavior) or ``"ctc_segmentation"`` (issue #104
                       FastConformer CTC pipeline).

    Returns:
        JSONResponse containing job status and job_id.
    """
    if alignment_mode not in VALID_ALIGNMENT_MODES:
        return JSONResponse(
            {
                "status": "error",
                "message": (
                    f"Invalid alignment_mode: '{alignment_mode}'. "
                    f"Must be one of {sorted(VALID_ALIGNMENT_MODES)}"
                ),
            },
            status_code=400,
        )

    if (
        alignment_mode == "whisperx"
        and model_size
        and model_size not in VALID_MODEL_SIZES
    ):
        return JSONResponse(
            {
                "status": "error",
                "message": (
                    f"Invalid model_size: '{model_size}'. "
                    f"Must be one of {sorted(VALID_MODEL_SIZES)}"
                ),
            },
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    os.makedirs("temp_audio", exist_ok=True)
    file_location = os.path.join("temp_audio", f"{job_id}_{surah_number}.mp3")

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {"status": "queued", "data": None, "error": None}

    if alignment_mode == "ctc_segmentation":
        background_tasks.add_task(
            lambda: _executor.submit(_run_ctc_job, job_id, file_location, surah_number)
        )
    else:
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
async def get_job_status(job_id: str):
    """
    مسار للتحقق من حالة المهمة
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
async def health():
    return {"status": "ok"}
