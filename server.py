import os
import uuid
import shutil
import gc
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from munajjam.transcription.whisperFactory import WhisperFactory, WhisperBackend

app = FastAPI(title="Munajjam API Server")

# السماح بالاتصال من أي واجهة (للتوافق مع Colab)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قاموس لتخزين حالة المهام في الخلفية
jobs: dict = {}
# ThreadPoolExecutor بمسار واحد لمنع تداخل عمليات كرت الشاشة (GPU)
_executor = ThreadPoolExecutor(max_workers=1)

print(
    "Initializing global WhisperX transcriber (models will be loaded lazily on first request)..."
)
global_transcriber = WhisperFactory().create_whisper(
    backend=WhisperBackend.WHISPERX, model_name="large-v2"
)


def _run_job(job_id: str, file_location: str, surah_number: int):
    """
    مهمة خلفية تقوم بالنسخ الصوتي والمزامنة ثم تحديث حالة المهمة.
    """
    try:
        jobs[job_id]["status"] = "processing"

        print(
            f"[Job {job_id[:8]}] Started processing Surah {surah_number} with WhisperX"
        )

        # استخدام WhisperX للحصول على تزمين على مستوى الكلمات (من الذاكرة المؤقتة)
        segments = global_transcriber.transcribe(file_location, surah_id=surah_number)

        response_data = []
        for segment in segments:
            ayah_data = {
                "ayah_number": segment.id,
                "start_time": segment.start,
                "end_time": segment.end,
            }
            if getattr(segment, "words", None):
                ayah_data["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in segment.words
                ]
            response_data.append(ayah_data)

        jobs[job_id] = {"status": "success", "data": response_data, "error": None}
        print(f"[Job {job_id[:8]}] Completed successfully")

    except Exception as e:
        import traceback

        traceback.print_exc()
        jobs[job_id] = {"status": "error", "data": None, "error": str(e)}
        print(f"[Job {job_id[:8]}] Error: {str(e)}")

    finally:
        # تنظيف الموارد الخاصة بالملف المؤقت فقط (مع إبقاء النماذج بالذاكرة)
        if os.path.exists(file_location):
            os.remove(file_location)
        gc.collect()


@app.post("/align/{surah_number}")
async def align_audio(
    surah_number: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    riwaya: str = Form("hafs"),
):
    """
    مسار لاستقبال الملفات وبدء المزامنة
    """
    job_id = str(uuid.uuid4())
    os.makedirs("temp_audio", exist_ok=True)
    file_location = os.path.join("temp_audio", f"{job_id}_{surah_number}.mp3")

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    jobs[job_id] = {"status": "queued", "data": None, "error": None}

    # تشغيل المعالجة في الخلفية
    background_tasks.add_task(
        lambda: _executor.submit(_run_job, job_id, file_location, surah_number)
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
