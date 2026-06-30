import os
import uuid
import shutil
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from munajjam.transcription import WhisperTranscriber
from munajjam.data import load_surah_ayahs
from munajjam.core import align
from munajjam.formatters import format_alignment_results

app = FastAPI(title="Munajjam API Server")

# السماح بالاتصال من أي واجهة (للتوافق مع Colab)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قاموس لتخزين حالة المهام في الخلفية
jobs: dict = {}
# ThreadPoolExecutor بمسار واحد لمنع تداخل عمليات كرت الشاشة (GPU)
_executor = ThreadPoolExecutor(max_workers=1)

def _run_job(job_id: str, file_location: str, surah_number: int):
    """
    مهمة خلفية تقوم بالنسخ الصوتي والمزامنة ثم تحديث حالة المهمة.
    تحتفظ بنفس طرق التزمين الأساسية الخاصة بمكتبة منجم.
    """
    try:
        jobs[job_id]["status"] = "processing"

        print(f"[Job {job_id[:8]}] Started processing Surah {surah_number}")

        # 1. النسخ الصوتي باستخدام إعدادات منجم
        with WhisperTranscriber() as transcriber:
            segments = transcriber.transcribe(file_location, surah_id=surah_number)
        
        # 2. تحميل الآيات الخاصة بالسورة
        ayahs = load_surah_ayahs(surah_number)
        
        # 3. المزامنة باستخدام خوارزميات منجم
        results = align(file_location, segments, ayahs)
        
        # 4. تنسيق المخرجات
        output = format_alignment_results(
            results=results,
            surah_id=surah_number
        )
        
        # الواجهة الأمامية تتوقع البيانات ضمن مفتاح data
        response_data = output.to_dict()["results"]

        jobs[job_id] = {
            "status": "success",
            "data": response_data,
            "error": None
        }
        print(f"[Job {job_id[:8]}] Completed successfully")

    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id] = {
            "status": "error",
            "data": None,
            "error": str(e)
        }
        print(f"[Job {job_id[:8]}] Error: {str(e)}")

    finally:
        # تنظيف الموارد
        if os.path.exists(file_location):
            os.remove(file_location)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@app.post("/align/{surah_number}")
async def align_audio(
    surah_number: int, 
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    riwaya: str = Form("hafs")
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

    return JSONResponse({
        "status": "queued",
        "job_id": job_id,
        "message": "بدأت المهمة وسيتم فحصها تلقائياً."
    })


@app.get("/align/status/{job_id}")
async def get_job_status(job_id: str):
    """
    مسار للتحقق من حالة المهمة
    """
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "error", "message": "المهمة غير موجودة"}, status_code=404)

    if job["status"] == "success":
        return JSONResponse({
            "status": "success",
            "data": job["data"]
        })
    elif job["status"] == "error":
        return JSONResponse({"status": "error", "message": job["error"]}, status_code=500)
    else:
        return JSONResponse({
            "status": job["status"],
            "message": "المعالجة مستمرة، يرجى الانتظار..."
        })


@app.get("/health")
async def health():
    return {"status": "ok"}
