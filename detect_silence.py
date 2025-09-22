from pydub import AudioSegment, silence
import json

def detect_silence(AUDIO_PATH):
    print("Detecting silences...")
    audio = AudioSegment.from_file(AUDIO_PATH, format="mp3")

    silences = silence.detect_silence(audio, min_silence_len=500, silence_thresh=-40)
    silences = [(round(start/1000,2), round(end/1000,2)) for start, end in silences]

    # حفظ النتائج
    with open("silences.json", "w", encoding="utf-8") as f:
        json.dump(silences, f, ensure_ascii=False, indent=2)

    print("Silences saved to silences.json")
