# align_segments.py
import json
import pandas as pd
from difflib import SequenceMatcher
from logging_utils import init_logging_file, log_result, SURAH_NAMES
import argparse
import os
from clean import remove_repetition_with_gemini

print("Running Alignment")

# -------------------
# Parse args
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--sura_id", type=int, required=True, help="Sura ID to align")
args = parser.parse_args()
current_sura_id = args.sura_id

# -------------------
# Init logging
# -------------------
log_file = init_logging_file(current_sura_id)
print(f"Logging initialized at {log_file}")

# -------------------
# Helpers
# -------------------
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def has_close_silence(time, silences, tol=0.3):
    return any(abs(time - s[0]) <= tol for s in silences)



# -------------------
# Load data
# -------------------
if not os.path.exists("segments.json"):
    raise FileNotFoundError("segments.json not found!")
segments = json.load(open("segments.json", encoding="utf-8"))

if not os.path.exists("silences.json"):
    raise FileNotFoundError("silences.json not found!")
with open("silences.json", encoding="utf-8") as f:
    silences = json.load(f)
silences = [(s[0], s[1]) for s in silences]

df = pd.read_csv("Quran Ayas List.csv")
surah = df[df["sura_id"] == current_sura_id]

aligned = []
seg_idx = 0

# -------------------
# Skip basmala if not sura 1
# -------------------
if seg_idx < len(segments):
    first_seg_text = segments[seg_idx].get("text", "")
    if "بِسْمِ " in first_seg_text and current_sura_id != 1:
        print(f"Skipping basmala segment because sura_id={current_sura_id} is not 1")
        seg_idx += 1

# -------------------
# Alignment loop
# -------------------
for _, row in surah.iterrows():
    verse_text = row["text"].strip()
    verse_id = row["id"]
    sura_id = row["sura_id"]
    index = row["index"]

    collected_text = ""
    start_time = None
    end_time = None

    # Special case for isti3aza
    if index == 1 and seg_idx < len(segments):
        text = segments[seg_idx]["text"].strip()
        if text in [
            "أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ",
            "أعوذ بالله من الشيطان الرجيم",
            "وعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ"
        ]:
            print("Skipping isti3aza segment")
            seg_idx += 1

    match_found = False
    while seg_idx < len(segments):
        seg = segments[seg_idx]
        seg_text = seg.get("text", "").strip()

        if start_time is None:
            start_time = seg.get("start", 0.0)

        if seg_text:
            collected_text += " " + seg_text
        end_time = seg.get("end", start_time)

        cleaned_text = remove_repetition_with_gemini(collected_text.strip())
        sim_original = similarity(verse_text, cleaned_text)

        print(f"Verse {index}: similarity={sim_original:.2f}")
        print(f"Original : {verse_text}")
        print(f"Cleaned  : {cleaned_text}")

        if sim_original >= 0.6:
            status = "Success"
            notes = ""
            aligned.append({
                "id": int(verse_id),
                "sura_id": int(sura_id),
                "index": int(index),
                "text": verse_text,
                "start": round(start_time, 2),
                "end": round(end_time, 2)
            })
            log_result(
                sura_id, index, verse_text, cleaned_text,
                round(start_time, 2), round(end_time, 2),
                round(sim_original, 3), status, notes
            )
            print(f"Logged verse {index}")
            seg_idx += 1
            match_found = True
            break
        else:
            if has_close_silence(end_time, silences):
                status = "Fail"
                notes = "Stopped at silence"
                log_result(
                    sura_id, index, verse_text, cleaned_text,
                    round(start_time, 2), round(end_time, 2),
                    round(sim_original, 3), status, notes
                )
                print(f"Stopped at silence, logged verse {index}")
                seg_idx += 1
                match_found = True
                break
            else:
                seg_idx += 1
                if seg_idx >= len(segments):
                    status = "Fail"
                    notes = "End of segments"
                    log_result(
                        sura_id, index, verse_text, cleaned_text,
                        round(start_time, 2), round(end_time, 2),
                        round(sim_original, 3), status, notes
                    )
                    print(f"End of segments, logged verse {index}")
                    match_found = True
                continue

    if not match_found:
        status = "Fail"
        notes = "No match found"
        log_result(
            sura_id, index, verse_text, "",
            0, 0,
            0, status, notes
        )
        print(f"No match found for verse {index}, logged as fail")

# -------------------
# Save aligned.json
# -------------------
with open("aligned.json", "w", encoding="utf-8") as f:
    json.dump(aligned, f, ensure_ascii=False, indent=2)

print(f"Saved {len(aligned)} ayah to aligned.json")
