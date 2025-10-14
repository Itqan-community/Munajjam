import os
import json
import csv
import re
from difflib import SequenceMatcher
from rapidfuzz import fuzz
import argparse
from .logging_utils import init_logging_file, log_result, SURAH_NAMES
from . import save_to_db as db
from collections import Counter



def normalize_arabic(text):
    """Normalize Arabic text by removing diacritics and standardizing letters"""
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def similarity(a, b):
    """Compute similarity ratio between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def get_first_last_words(text, n=1):
    """Return first n and last n words from text"""
    words = normalize_arabic(text).split()
    first = " ".join(words[:n]) if len(words) >= n else " ".join(words)
    last = " ".join(words[-n:]) if len(words) >= n else " ".join(words)
    return first, last

def remove_overlap(text1, text2):
    """Merge text2 into text1 while removing overlapping words"""
    words1 = normalize_arabic(text1).split()
    words2 = text2.split()
    count1 = Counter(words1)
    cleaned_words2 = []
    for word in words2:
        if count1[normalize_arabic(word)] > 0:
            count1[normalize_arabic(word)] -= 1
            continue
        cleaned_words2.append(word)
    if not cleaned_words2:
        return text1.strip(), True
    return text1.strip() + " " + " ".join(cleaned_words2).strip(), True

def load_config(config_file=os.path.join("data", "current_config.json")):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get("SURAH_UUID"), config.get("RECITER_NAME")
    except FileNotFoundError:
        print(f"Error: {config_file} not found.")
        return None, None


def clean_and_merge_segments(sura_id_filter):
    log_file = init_logging_file(sura_id_filter)
    print(f"Logging initialized at {log_file}")
    surah_uuid, reciter_name = load_config()
    if not surah_uuid or not reciter_name:
        return

    # Load transcribed segments for the current sura
    segments_file = os.path.join("data", "segments", f"{sura_id_filter}_segments.json")
    try:
        with open(segments_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    except FileNotFoundError:
        print(f"Error: {segments_file} not found.")
        return

    # Load ayahs
    ayahs = []
    try:
        with open(os.path.join("data", "Quran Ayas List.csv"), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if sura_id_filter is None or row['sura_id'] == str(sura_id_filter):
                    ayahs.append({
                        "id": int(row["id"]),
                        "sura_id": int(row["sura_id"]),
                        "index": int(row["index"]),
                        "text": row["text"],
                    })
    except FileNotFoundError:
        print("Error: Quran Ayas List.csv.csv not found.")
        return

    # Database setup
    conn = db.create_connection(os.path.join("data", "quran.db"))
    if conn is not None:
        db.create_table(conn)
    else:
        print("Error! Cannot create the database connection.")
        return

    cleaned_segments = []
    i = 0
    ayah_index = 0

    while i < len(segments) and ayah_index < len(ayahs):
        start_time = segments[i]['start']
        merged_text = segments[i]['text']
        end_time = segments[i]['end']
        overlap_flag = False

        while True:
            # Compute full similarity with the ayah
            full_sim = similarity(normalize_arabic(merged_text), normalize_arabic(ayahs[ayah_index]["text"]))
            words_in_ayah = ayahs[ayah_index ]["text"].strip().split()
            N_CHECK = 1 
            if len(words_in_ayah) >= 3:
                N_CHECK = 3
            elif len(words_in_ayah) == 2:
                N_CHECK = 2
            # Also check similarity of last two words
            _, seg_last2 = get_first_last_words(merged_text, n=N_CHECK)
            _, ayah_last2 = get_first_last_words(ayahs[ayah_index]["text"], n=N_CHECK)
            last_words_sim = similarity(seg_last2, ayah_last2)

            print("\n--- Comparing segment with ayah ---")
            print(f"Segment ID {segments[i]['id']} text: {merged_text}")
            print(f"Ayah index {ayah_index} text   : {ayahs[ayah_index]['text']}")
            print(f"Full similarity : {full_sim:.3f}")
            print(f"Last 2 words similarity : {last_words_sim:.3f}")

            # Decide if reached end of ayah
            if last_words_sim >= 0.6:
                print("✅ Reached end of ayah based on last words similarity.")
                # Save to DB and cleaned_segments
                ayah_data = (
                    surah_uuid,
                    ayahs[ayah_index]['sura_id'],
                    ayahs[ayah_index]['index'],
                    start_time,
                    end_time,
                    reciter_name
                )
                db.insert_ayah_timestamp(conn, ayah_data)
                cleaned_segments.append({
                    "id": len(cleaned_segments) + 1,
                    "sura_id": ayahs[ayah_index]['sura_id'],
                    "ayah_index": ayah_index,
                    "start": start_time,
                    "end": end_time,
                    "transcribed_text": merged_text,
                    "corrected_text": ayahs[ayah_index]["text"],
                    "start_id": segments[i]['id'],
                    "end_id": segments[i]['id'],
                    "uuid": surah_uuid
                })
                ayah_index += 1
                break

            # Check if next segment starts a new ayah
            if i + 1 < len(segments) :
                if  ayah_index+1 < len(ayahs):
                    words_in_next_ayah = ayahs[ayah_index + 1]["text"].strip().split()
                    N_CHECK = 1 
                    if len(words_in_next_ayah) >= 3:
                        N_CHECK = 3
                    elif len(words_in_next_ayah) == 2:
                        N_CHECK = 2
                    next_first_seg, _ = get_first_last_words(segments[i+1]["text"], n=N_CHECK)
                    next_first_ayah, _ = get_first_last_words(ayahs[ayah_index + 1]["text"], n=N_CHECK)
                    sim=similarity(next_first_seg, next_first_ayah)
                    if ayah_index + 1 < len(ayahs) and sim > 0.6:
                        print(f"⚠️ Found next ayah start (using {N_CHECK} words),with similarity ={sim} Finalizing current ayah.")
                        ayah_data = (
                            surah_uuid,
                            ayahs[ayah_index]['sura_id'],
                            ayahs[ayah_index]['index'],
                            start_time,
                            end_time,
                            reciter_name
                        )
                        db.insert_ayah_timestamp(conn, ayah_data)
                        cleaned_segments.append({
                            "id": len(cleaned_segments) + 1,
                            "sura_id": ayahs[ayah_index]['sura_id'],
                            "ayah_index": ayah_index,
                            "start": start_time,
                            "end": end_time,
                            "transcribed_text": merged_text,
                            "corrected_text": ayahs[ayah_index]["text"],
                            "start_id": segments[i]['id'],
                            "end_id": segments[i]['id'],
                            "uuid": surah_uuid
                        })
                        ayah_index += 1
                        break

                # Merge next segment if not end
                print(f"🔄 Merging segment {segments[i]['id']} with next segment {segments[i+1]['id']}")
                merged_text, overlap_found = remove_overlap(merged_text, segments[i+1]['text'])
                if overlap_found:
                    overlap_flag = True
                end_time = segments[i+1]['end']
                i += 1
            else:
                # End of segments
                print("❌ End of segments reached. Finalizing last ayah.")
                ayah_data = (
                    surah_uuid,
                    ayahs[ayah_index]['sura_id'],
                    ayahs[ayah_index]['index'],
                    start_time,
                    end_time,
                    reciter_name
                )
                db.insert_ayah_timestamp(conn, ayah_data)
                cleaned_segments.append({
                    "id": len(cleaned_segments) + 1,
                    "sura_id": ayahs[ayah_index]['sura_id'],
                    "ayah_index": ayah_index,
                    "start": start_time,
                    "end": end_time,
                    "transcribed_text": merged_text,
                    "corrected_text": ayahs[ayah_index]["text"],
                    "start_id": segments[i]['id'],
                    "end_id": segments[i]['id'],
                    "uuid": surah_uuid
                })
                ayah_index += 1
                break

        i += 1

    conn.close()

    # Save corrected segments
    output_file = os.path.join("data", "corrected_segments", f"corrected_segments_{sura_id_filter}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_segments, f, ensure_ascii=False, indent=4)

    print(f"\n✅ All segments for Sura {sura_id_filter} processed and saved to {output_file}.")

def main():
    print("Aligning segments...")
    parser = argparse.ArgumentParser(description="Align Quranic segments with original ayahs and save to DB.")
    parser.add_argument("--sura_id", type=int, required=True, help="Sura ID to align")
    args = parser.parse_args()
    current_sura_id = args.sura_id
    clean_and_merge_segments(current_sura_id)

if __name__ == "__main__":
    main()
