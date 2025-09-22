import csv
import os
import json


SURAH_NAMES = {
    1: "الفاتحة", 2: "البقرة", 3: "آل_عمران", 4: "النساء", 5: "المائدة",
    6: "الأنعام", 7: "الأعراف", 8: "الأنفال", 9: "التوبة", 10: "يونس",
    11: "هود", 12: "يوسف", 13: "الرعد", 14: "إبراهيم", 15: "الحجر",
    16: "النحل", 17: "الإسراء", 18: "الكهف", 19: "مريم", 20: "طه",
    21: "الأنبياء", 22: "الحج", 23: "المؤمنون", 24: "النور", 25: "الفرقان",
    26: "الشعراء", 27: "النمل", 28: "القصص", 29: "العنكبوت", 30: "الروم",
    31: "لقمان", 32: "السجدة", 33: "الأحزاب", 34: "سبأ", 35: "فاطر",
    36: "يس", 37: "الصافات", 38: "ص", 39: "الزمر", 40: "غافر",
    41: "فصلت", 42: "الشورى", 43: "الزخرف", 44: "الدخان", 45: "الجاثية",
    46: "الأحقاف", 47: "محمد", 48: "الفتح", 49: "الحجرات", 50: "ق",
    51: "الذاريات", 52: "الطور", 53: "النجم", 54: "القمر", 55: "الرحمن",
    56: "الواقعة", 57: "الحديد", 58: "المجادلة", 59: "الحشر", 60: "الممتحنة",
    61: "الصف", 62: "الجمعة", 63: "المنافقون", 64: "التغابن", 65: "الطلاق",
    66: "التحريم", 67: "الملك", 68: "القلم", 69: "الحاقة", 70: "المعارج",
    71: "نوح", 72: "الجن", 73: "المزمل", 74: "المدثر", 75: "القيامة",
    76: "الإنسان", 77: "المرسلات", 78: "النبأ", 79: "النازعات", 80: "عبس",
    81: "التكوير", 82: "الإنفطار", 83: "المطففين", 84: "الإنشقاق", 85: "البروج",
    86: "الطارق", 87: "الأعلى", 88: "الغاشية", 89: "الفجر", 90: "البلد",
    91: "الشمس", 92: "الليل", 93: "الضحى", 94: "الشرح", 95: "التين",
    96: "العلق", 97: "القدر", 98: "البينة", 99: "الزلزلة", 100: "العاديات",
    101: "القارعة", 102: "التكاثر", 103: "العصر", 104: "الهمزة", 105: "الفيل",
    106: "قريش", 107: "الماعون", 108: "الكوثر", 109: "الكافرون", 110: "النصر",
    111: "المسد", 112: "الإخلاص", 113: "الفلق", 114: "الناس"
}


def load_current_config():
    with open("current_config.json", encoding="utf-8") as f:
        data = json.load(f)
    return data["RECITER_NAME"], data["RECITATION_UUID"]


def get_log_file_path(sura_id: int) -> str:
    reciter_name, recitation_uuid = load_current_config()
    surah_name = SURAH_NAMES.get(sura_id, f"Unknown-{sura_id}")
    
    base_dir = f"{reciter_name}-{recitation_uuid}"
    surah_dir = os.path.join(base_dir, f"{sura_id}-{surah_name}")
    os.makedirs(surah_dir, exist_ok=True)  # ✅ يعمل الفولدر لو مش موجود
    
    return os.path.join(surah_dir, "Logging.csv")

# -----------------------
# Logging for each surah
# -----------------------
def init_logging_file(sura_id: int) -> str:
    log_file = get_log_file_path(sura_id)
    with open(log_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sura_id", "ayah_index", "ayah_text",
            "model_text", "start_time", "end_time",
            "similarity_score", "status", "notes"
        ])
    return log_file

# -----------------------
# add to logging
# -----------------------
def log_result(
    sura_id: int,
    ayah_index: int,
    ayah_text: str,
    model_text: str,
    start_time: float,
    end_time: float,
    similarity_score: float,
    status: str,
    notes: str = ""
):
    log_file = get_log_file_path(sura_id)
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            sura_id, ayah_index, ayah_text, model_text,
            start_time, end_time, similarity_score,
            status, notes
        ])
