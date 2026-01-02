"""
Check which surah audio files are missing from the audio folder.

Usage:
    python check_missing.py
"""

from pathlib import Path

# Surah names (1-114)
SURAH_NAMES = {
    1: "الفاتحة", 2: "البقرة", 3: "آل عمران", 4: "النساء", 5: "المائدة",
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
    81: "التكوير", 82: "الانفطار", 83: "المطففين", 84: "الانشقاق", 85: "البروج",
    86: "الطارق", 87: "الأعلى", 88: "الغاشية", 89: "الفجر", 90: "البلد",
    91: "الشمس", 92: "الليل", 93: "الضحى", 94: "الشرح", 95: "التين",
    96: "العلق", 97: "القدر", 98: "البينة", 99: "الزلزلة", 100: "العاديات",
    101: "القارعة", 102: "التكاثر", 103: "العصر", 104: "الهمزة", 105: "الفيل",
    106: "قريش", 107: "الماعون", 108: "الكوثر", 109: "الكافرون", 110: "النصر",
    111: "المسد", 112: "الإخلاص", 113: "الفلق", 114: "الناس",
}

AUDIO_FOLDER = Path("Quran/badr_alturki_audio")
OUTPUT_FOLDER = Path("output")
TOTAL_SURAHS = 114


def main():
    print("\n" + "=" * 60)
    print("  QURAN AUDIO FILES STATUS CHECK")
    print("=" * 60)
    
    # Check which audio files exist
    existing_audio = set()
    if AUDIO_FOLDER.exists():
        for f in AUDIO_FOLDER.glob("*.wav"):
            try:
                surah_id = int(f.stem)
                existing_audio.add(surah_id)
            except ValueError:
                pass
    
    # Check which output files exist
    existing_output = set()
    if OUTPUT_FOLDER.exists():
        for f in OUTPUT_FOLDER.glob("surah_*.json"):
            try:
                surah_id = int(f.stem.replace("surah_", ""))
                existing_output.add(surah_id)
            except ValueError:
                pass
    
    # Calculate missing
    all_surahs = set(range(1, TOTAL_SURAHS + 1))
    missing_audio = all_surahs - existing_audio
    missing_output = existing_audio - existing_output  # Has audio but no output
    
    # Summary
    print(f"\n📁 Audio folder: {AUDIO_FOLDER}")
    print(f"📂 Output folder: {OUTPUT_FOLDER}")
    print(f"\n{'─' * 60}")
    print(f"📊 SUMMARY")
    print(f"{'─' * 60}")
    print(f"   Total surahs in Quran: {TOTAL_SURAHS}")
    print(f"   ✅ Audio files present: {len(existing_audio)}")
    print(f"   ❌ Audio files missing: {len(missing_audio)}")
    print(f"   📝 Already processed: {len(existing_output)}")
    print(f"   🔄 Ready to process: {len(missing_output)}")
    
    # List missing audio files
    if missing_audio:
        print(f"\n{'─' * 60}")
        print(f"❌ MISSING AUDIO FILES ({len(missing_audio)} surahs)")
        print(f"{'─' * 60}")
        for surah_id in sorted(missing_audio):
            name = SURAH_NAMES.get(surah_id, "Unknown")
            print(f"   {surah_id:03d}.wav - {name}")
        
        # Quick copy list
        print(f"\n📋 Quick copy list:")
        missing_files = [f"{s:03d}.wav" for s in sorted(missing_audio)]
        print(f"   {', '.join(missing_files)}")
    else:
        print(f"\n🎉 All 114 surah audio files are present!")
    
    # List pending processing
    if missing_output:
        print(f"\n{'─' * 60}")
        print(f"🔄 PENDING PROCESSING ({len(missing_output)} surahs)")
        print(f"{'─' * 60}")
        for surah_id in sorted(missing_output):
            name = SURAH_NAMES.get(surah_id, "Unknown")
            print(f"   {surah_id:03d} - {name}")
        print(f"\n   Run 'python batch_process.py' to process these!")
    elif len(existing_output) == TOTAL_SURAHS:
        print(f"\n🎉 All 114 surahs have been processed!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
