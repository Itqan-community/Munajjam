"""
سكربت التحميل الشامل لنماذج منصة مُنجّم للتزمين القرآني من Hugging Face
Munajjam Master ONNX Models Downloader
المستودع الرسمي: https://huggingface.co/Alimalas/munajjam-onnx-models
"""

import argparse
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ID = "Alimalas/munajjam-onnx-models"

MODELS = [
    {
        "name": "نموذج تقطيع الأنفاس القرآني والوقف (Recitation Segmenter v2 - ONNX DirectML)",
        "folder": "model_segmenter",
        "key_file": "model.onnx",
        "pattern": "model_segmenter/*",
    },
    {
        "name": "محرك التزمين الصوتي الفونيمي (Zipformer v3 - ONNX)",
        "folder": "model_zipformer",
        "key_file": "zipformer_p_arabic_v3.onnx",
        "pattern": "model_zipformer/*",
    },
    {
        "name": "كاشف النشاط الصوتي والعزل (Silero VAD - ONNX)",
        "folder": "model_vad",
        "key_file": "silero_vad.onnx",
        "pattern": "model_vad/*",
    },
    {
        "name": "نموذج التدقيق والمطابقة الصوتية (Wav2Vec2 Arabic - ONNX)",
        "folder": "model_wav2vec2",
        "key_file": "model.onnx",
        "pattern": "model_wav2vec2/*",
    },
]


def download_all_models(target_base_dir: str, force: bool = False) -> bool:
    print("=" * 70)
    print("      🌟 مُنجّم: بدء تنزيل نماذج الذكاء الاصطناعي (ONNX DirectML) 🌟")
    print(f"📦 المستودع الرسمي المفتوح: https://huggingface.co/{REPO_ID}")
    print(f"📁 مجلد النماذج: {os.path.abspath(target_base_dir)}")
    print("=" * 70)
    print()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ مكتبة huggingface_hub غير مثبتة في بيئة بايثون.")
        print("   يرجى تثبيتها عبر: pip install huggingface_hub")
        return False

    os.makedirs(target_base_dir, exist_ok=True)
    success_count = 0
    total_models = len(MODELS)

    for idx, item in enumerate(MODELS, start=1):
        target_folder = os.path.join(target_base_dir, item["folder"])
        key_file_path = os.path.join(target_folder, item["key_file"])

        print(f"[{idx}/{total_models}] {item['name']}")
        print(f"📁 المجلد المحلي: {target_folder}")

        if os.path.exists(key_file_path) and not force:
            print("   ✅ النموذج موجود مسبقاً، تم التحقق والتخطي.")
            print()
            success_count += 1
            continue

        print("   ⏳ جاري التحميل من Hugging Face...")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                allow_patterns=[item["pattern"]],
                local_dir=target_base_dir,
                local_dir_use_symlinks=False,
            )
            if os.path.exists(key_file_path):
                size_mb = os.path.getsize(key_file_path) / (1024 * 1024)
                print(f"   ✅ اكتمل التحميل بنجاح! ({item['key_file']} - {size_mb:.2f} MB)")
                success_count += 1
            else:
                print("   ⚠️ تم انتهاء النقل ولكن لم يتم العثور على الملف الرئيسي.")
        except Exception as e:
            print(f"   ❌ تعذر التحميل: {e}")

        print()

    print("=" * 70)
    if success_count == total_models:
        print("🎉 اكتمل فحص وتنزيل كافة النماذج بنجاح تام وبصيغة ONNX المسرعة عتادياً!")
        print("👉 يمكنك الآن إطلاق خادم الذكاء الاصطناعي وتشغيل منصة مُنجّم مباشرة.")
    else:
        print(f"⚠️ اكتمل تنزيل {success_count} من أصل {total_models} نماذج.")
        print("يرجى التأكد من اتصال الإنترنت وإعادة المحاولة إن لزم الأمر.")
    print("=" * 70)
    return success_count == total_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all Munajjam ONNX models")
    default_dir = os.path.join(os.getcwd(), "munajjam", "models")
    parser.add_argument("--output-dir", default=default_dir, help="Directory to store models")
    parser.add_argument("--force", action="store_true", help="Force re-download if models exist")
    args = parser.parse_args()

    download_all_models(target_base_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
