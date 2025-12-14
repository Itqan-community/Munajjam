import os
import json
import csv
import re
from difflib import SequenceMatcher
import argparse
from .logging_utils import init_logging_file, log_result, SURAH_NAMES
from . import save_to_db as db
from collections import Counter

# Known special segment types and patterns (e.g., isti'aza and basmala)
SPECIAL_SEGMENT_TYPES = {"isti3aza", "basmala", "basmalah"}
isti3aza_pattern = re.compile(r"اعوذ بالله من الشيطان الرجيم")
basmala_pattern = re.compile(r"(?:ب\s*س?م?\s*)?الله\s*الرحمن\s*الرحيم")


def normalize_arabic(text):
    """Normalize Arabic text by removing diacritics and standardizing letters"""
    text = re.sub(r"[أإآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_special_type(segment):
    """
    Detect if a segment is a special type (isti'aza or basmala), even if the
    incoming type is missing or slightly different.
    """
    seg_type = segment.get("type")
    if seg_type in SPECIAL_SEGMENT_TYPES:
        # Normalize spelling to a single canonical form
        return "basmala" if seg_type == "basmalah" else seg_type

    norm_text = normalize_arabic(segment.get("text", ""))
    if basmala_pattern.search(norm_text):
        return "basmala"
    if isti3aza_pattern.search(norm_text):
        return "isti3aza"
    return None

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


def apply_buffers(start_time, end_time, silences, prev_end=None, next_start=None, buffer=0.3):
    """
    Apply buffers to start and end times by extending into adjacent silence periods.
    
    Args:
        start_time: Original start time in seconds
        end_time: Original end time in seconds
        silences: List of silence periods [[start_ms, end_ms], ...]
        prev_end: Previous ayah's end time (to prevent overlap)
        next_start: Next ayah's start time (to prevent overlap)
        buffer: Buffer duration in seconds (default 0.3)
    
    Returns:
        (new_start_time, new_end_time) with buffers applied
    """
    new_start = start_time
    new_end = end_time
    
    if not silences:
        return new_start, new_end
    
    # Convert silences to seconds and sort by start time
    silences_sec = [(s[0]/1000, s[1]/1000) for s in silences]
    silences_sec.sort(key=lambda x: x[0])
    
    # Find silence before start_time
    # Look for the silence period that ends closest to (but before or at) start_time
    best_silence_before = None
    for silence_start, silence_end in silences_sec:
        if silence_end <= start_time:
            # This silence ends before or at start_time
            if best_silence_before is None or silence_end > best_silence_before[1]:
                best_silence_before = (silence_start, silence_end)
        elif silence_start > start_time:
            # We've passed start_time, no need to continue
            break
    
    if best_silence_before:
        silence_start, silence_end = best_silence_before
        # Calculate how much we can extend backward into this silence
        # We can extend from start_time back to silence_start (or buffer amount, whichever is smaller)
        available_buffer = start_time - silence_start
        buffer_to_apply = min(buffer, available_buffer)
        buffer_start = start_time - buffer_to_apply
        
        # Make sure we don't overlap with previous ayah
        if prev_end is None:
            new_start = buffer_start
        elif buffer_start >= prev_end:
            new_start = buffer_start
        elif prev_end < start_time:
            # Can still add partial buffer, but don't go before prev_end
            new_start = max(buffer_start, prev_end)
        # else: prev_end >= start_time, which shouldn't happen, so keep original start_time
    
    # Find silence after end_time
    # Look for the silence period that starts closest to (but at or after) end_time
    best_silence_after = None
    for silence_start, silence_end in silences_sec:
        if silence_start >= end_time:
            # This silence starts at or after end_time
            if best_silence_after is None or silence_start < best_silence_after[0]:
                best_silence_after = (silence_start, silence_end)
        elif silence_end < end_time:
            # This silence ends before end_time, skip it
            continue
    
    if best_silence_after:
        silence_start, silence_end = best_silence_after
        # Calculate how much we can extend forward into this silence
        # We can extend from end_time forward to silence_end (or buffer amount, whichever is smaller)
        available_buffer = silence_end - end_time
        buffer_to_apply = min(buffer, available_buffer)
        buffer_end = end_time + buffer_to_apply
        
        # Make sure we don't overlap with next ayah
        if next_start is None:
            new_end = buffer_end
        elif buffer_end <= next_start:
            new_end = buffer_end
        elif next_start > end_time:
            # Can still add partial buffer, but don't go beyond next_start
            new_end = min(buffer_end, next_start)
        # else: next_start <= end_time, which shouldn't happen, so keep original end_time
    
    return new_start, new_end


def find_silence_gap_between(current_end, next_start, silences_sec, min_gap=0.18):
    """
    Detect a silence gap between two consecutive segments.

    Args:
        current_end: End time (sec) of the current segment.
        next_start: Start time (sec) of the next segment.
        silences_sec: List of silence periods in seconds [(start, end), ...] sorted by start.
        min_gap: Minimum silence duration to be considered a real gap (sec).

    Returns:
        (silence_start, silence_end) if a qualifying gap exists, otherwise None.
    """
    if not silences_sec or next_start is None:
        return None

    for silence_start, silence_end in silences_sec:
        # Skip silences that end before the current segment
        if silence_end <= current_end:
            continue
        # Stop early once we pass the next segment
        if silence_start >= next_start:
            break
        # Silence fully between current_end and next_start
        if silence_start >= current_end and silence_end <= next_start:
            if (silence_end - silence_start) >= min_gap:
                return silence_start, silence_end
    return None


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
    
    # Load silences for buffer calculation
    silences_file = os.path.join("data", "silences", f"{sura_id_filter}_silences.json")
    silences = []
    try:
        with open(silences_file, 'r', encoding='utf-8') as f:
            silences = json.load(f)
        print(f"Loaded {len(silences)} silence periods for buffer calculation")
    except FileNotFoundError:
        print(f"Warning: {silences_file} not found. Buffers will not be applied.")
    except Exception as e:
        print(f"Warning: Error loading silences: {e}. Buffers will not be applied.")

    # Load ayahs
    ayahs = []
    try:
        with open(os.path.join("data", "Quran Ayas List.csv"), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Normalize sura_id_filter for comparison (handle both "067" and 67)
            sura_id_int = int(sura_id_filter) if sura_id_filter is not None else None
            for row in reader:
                if sura_id_filter is None or int(row['sura_id']) == sura_id_int:
                    ayahs.append({
                        "id": int(row["id"]),
                        "sura_id": int(row["sura_id"]),
                        "index": int(row["index"]),
                        "text": row["text"],
                    })
    except FileNotFoundError:
        print("Error: Quran Ayas List.csv not found.")
        return

    # Database setup
    conn = db.create_connection(os.path.join("data", "quran.db"))
    if conn is not None:
        db.create_table(conn)
    else:
        print("Error! Cannot create the database connection.")
        return

    cleaned_segments = []
    next_ayah_id = 1  # Track ayah numbering separately from special segments
    i = 0
    ayah_index = 0
    prev_ayah_end = None  # Track previous ayah's end time for buffer overlap prevention
    
    # First, add all Isti'aza and Basmala segments (id 0) to cleaned_segments
    for seg in segments:
        detected_type = detect_special_type(seg)
        if seg.get('id') == 0 or detected_type:
            special_type = detected_type or seg.get('type', 'unknown')
            cleaned_segments.append({
                "id": 0,
                "sura_id": seg['sura_id'],
                "ayah_index": -1,  # Special index for Isti'aza/Basmala
                "start": seg['start'],
                "end": seg['end'],
                "transcribed_text": seg['text'],
                "corrected_text": seg['text'],  # Keep original text for these
                "start_id": seg['id'],
                "end_id": seg['id'],
                "uuid": seg.get('surah_uuid', surah_uuid),
                "type": special_type
            })
            print(f"Added {special_type} segment: {seg['start']:.2f}s -> {seg['end']:.2f}s")

    # Pre-compute silences in seconds for quick lookups
    silences_sec = [(s[0] / 1000, s[1] / 1000) for s in silences]
    silences_sec.sort(key=lambda x: x[0])

    while i < len(segments) and ayah_index < len(ayahs):
        # Skip Isti'aza and Basmala segments (id 0) during alignment
        segment_type = detect_special_type(segments[i]) or segments[i].get('type', 'ayah')
        if segments[i].get('id') == 0 or segment_type in SPECIAL_SEGMENT_TYPES:
            print(f"Skipping segment {i} (type: {segment_type}) during ayah alignment")
            i += 1
            continue
        
        start_time = segments[i]['start']
        merged_text = segments[i]['text']
        end_time = segments[i]['end']
        overlap_flag = False
        normalized_seg = normalize_arabic(merged_text)

        while True:
            # Compute full similarity with the ayah
            full_sim = similarity(normalized_seg, normalize_arabic(ayahs[ayah_index]["text"]))
            words_in_ayah = ayahs[ayah_index]["text"].strip().split()
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

            # Guard for required tokens to avoid early cuts (e.g., ayah 2 needs "ارجع" + "فطور")
            required_tokens_map = {
                2: ["ارجع", "فطور"],  # ayah_index: required normalized tokens
            }
            missing_required = False
            if ayah_index in required_tokens_map:
                normalized_seg = normalize_arabic(merged_text)
                for tok in required_tokens_map[ayah_index]:
                    if tok not in normalized_seg:
                        missing_required = True
                        break
            if missing_required:
                print("⏩ Missing required tokens for this ayah; force merging next segment before finalizing.")
                if i + 1 < len(segments):
                    merged_text, overlap_found = remove_overlap(merged_text, segments[i+1]["text"])
                    if overlap_found:
                        overlap_flag = True
                    end_time = segments[i+1]['end']
                    normalized_seg = normalize_arabic(merged_text)
                    i += 1
                    continue
                else:
                    print("⚠️ Last segment but required tokens still missing; proceeding to finalize to avoid stall.")

            # Decide if reached end of ayah
            if last_words_sim >= 0.6:
                seg_word_count = len(normalize_arabic(merged_text).split())
                ayah_word_count = len(words_in_ayah)
                coverage_ratio = seg_word_count / max(ayah_word_count, 1)

                # Guard: if coverage is too low (e.g., early stop on "ارجع"), keep merging
                if coverage_ratio < 0.7 and i + 1 < len(segments):
                    print(f"⏩ Similarity hit but coverage {coverage_ratio:.2f} < 0.70; continue merging.")
                else:
                    print("✅ Reached end of ayah based on last words similarity and coverage.")
                    
                    # Get next ayah's start time for buffer calculation (if exists)
                    next_ayah_start = None
                    if ayah_index + 1 < len(ayahs):
                        # We don't know next ayah's start yet, so we'll use None
                        # This means end buffer won't be constrained by next ayah
                        pass
                    
                    # Apply buffers to start and end times
                    buffered_start, buffered_end = apply_buffers(
                        start_time, end_time, silences, 
                        prev_end=prev_ayah_end, 
                        next_start=next_ayah_start,
                        buffer=0.3
                    )
                    
                    # Save to DB and cleaned_segments
                    ayah_data = (
                        surah_uuid,
                        ayahs[ayah_index]['sura_id'],
                        ayahs[ayah_index]['index'],
                        buffered_start,
                        buffered_end,
                        reciter_name
                    )
                    db.insert_ayah_timestamp(conn, ayah_data)
                    cleaned_segments.append({
                        "id": next_ayah_id,
                        "sura_id": ayahs[ayah_index]['sura_id'],
                        "ayah_index": ayah_index,
                        "start": buffered_start,
                        "end": buffered_end,
                        "transcribed_text": merged_text,
                        "corrected_text": ayahs[ayah_index]["text"],
                        "start_id": segments[i]['id'],
                        "end_id": segments[i]['id'],
                        "uuid": surah_uuid
                    })
                    next_ayah_id += 1
                    
                    # Update previous ayah end for next iteration
                    prev_ayah_end = buffered_end
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
                        
                        # Next ayah starts at segments[i+1]['start'], use it to constrain end buffer
                        next_ayah_start = segments[i+1]['start']
                        
                        # Apply buffers to start and end times
                        buffered_start, buffered_end = apply_buffers(
                            start_time, end_time, silences, 
                            prev_end=prev_ayah_end, 
                            next_start=next_ayah_start,
                            buffer=0.3
                        )
                        
                        ayah_data = (
                            surah_uuid,
                            ayahs[ayah_index]['sura_id'],
                            ayahs[ayah_index]['index'],
                            buffered_start,
                            buffered_end,
                            reciter_name
                        )
                        db.insert_ayah_timestamp(conn, ayah_data)
                        cleaned_segments.append({
                            "id": next_ayah_id,
                            "sura_id": ayahs[ayah_index]['sura_id'],
                            "ayah_index": ayah_index,
                            "start": buffered_start,
                            "end": buffered_end,
                            "transcribed_text": merged_text,
                            "corrected_text": ayahs[ayah_index]["text"],
                            "start_id": segments[i]['id'],
                            "end_id": segments[i]['id'],
                            "uuid": surah_uuid
                        })
                        next_ayah_id += 1
                        
                        # Update previous ayah end for next iteration
                        prev_ayah_end = buffered_end
                        ayah_index += 1
                        break
                # If there is a clear silence gap between current end and next start, consider ayah boundary
                silence_gap = find_silence_gap_between(
                    end_time,
                    segments[i + 1]["start"],
                    silences_sec,
                    min_gap=0.18,
                )
                if silence_gap:
                    # Textual check: does the next segment likely start the next ayah?
                    textual_boundary = False
                    if ayah_index + 1 < len(ayahs):
                        # Determine N_CHECK based on next ayah length
                        words_in_next_ayah = ayahs[ayah_index + 1]["text"].strip().split()
                        N_CHECK = 1
                        if len(words_in_next_ayah) >= 3:
                            N_CHECK = 3
                        elif len(words_in_next_ayah) == 2:
                            N_CHECK = 2
                        
                        next_first_seg, _ = get_first_last_words(segments[i + 1]["text"], n=N_CHECK)
                        next_first_ayah, _ = get_first_last_words(ayahs[ayah_index + 1]["text"], n=N_CHECK)
                        next_sim = similarity(next_first_seg, next_first_ayah)
                        if next_sim > 0.6:
                            textual_boundary = True

                    if textual_boundary:
                        gap_start, gap_end = silence_gap
                        print(f"🧭 Silence gap from {gap_start:.2f}s to {gap_end:.2f}s + textual cues; treating as ayah boundary.")

                        # Constrain end buffer to the start of the silence gap to avoid bleeding
                        next_ayah_start = gap_start

                        buffered_start, buffered_end = apply_buffers(
                            start_time,
                            end_time,
                            silences,
                            prev_end=prev_ayah_end,
                            next_start=next_ayah_start,
                            buffer=0.3,
                        )

                        ayah_data = (
                            surah_uuid,
                            ayahs[ayah_index]["sura_id"],
                            ayahs[ayah_index]["index"],
                            buffered_start,
                            buffered_end,
                            reciter_name,
                        )
                        db.insert_ayah_timestamp(conn, ayah_data)
                        cleaned_segments.append(
                            {
                                "id": next_ayah_id,
                                "sura_id": ayahs[ayah_index]["sura_id"],
                                "ayah_index": ayah_index,
                                "start": buffered_start,
                                "end": buffered_end,
                                "transcribed_text": merged_text,
                                "corrected_text": ayahs[ayah_index]["text"],
                                "start_id": segments[i]["id"],
                                "end_id": segments[i]["id"],
                                "uuid": surah_uuid,
                            }
                        )
                        next_ayah_id += 1
                        prev_ayah_end = buffered_end
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
                
                # No next ayah, so end buffer is not constrained
                next_ayah_start = None
                
                # Apply buffers to start and end times
                buffered_start, buffered_end = apply_buffers(
                    start_time, end_time, silences, 
                    prev_end=prev_ayah_end, 
                    next_start=next_ayah_start,
                    buffer=0.3
                )
                
                ayah_data = (
                    surah_uuid,
                    ayahs[ayah_index]['sura_id'],
                    ayahs[ayah_index]['index'],
                    buffered_start,
                    buffered_end,
                    reciter_name
                )
                db.insert_ayah_timestamp(conn, ayah_data)
                cleaned_segments.append({
                    "id": next_ayah_id,
                    "sura_id": ayahs[ayah_index]['sura_id'],
                    "ayah_index": ayah_index,
                    "start": buffered_start,
                    "end": buffered_end,
                    "transcribed_text": merged_text,
                    "corrected_text": ayahs[ayah_index]["text"],
                    "start_id": segments[i]['id'],
                    "end_id": segments[i]['id'],
                    "uuid": surah_uuid
                })
                next_ayah_id += 1
                
                # Update previous ayah end for next iteration
                prev_ayah_end = buffered_end
                ayah_index += 1
                break

        i += 1

    conn.close()

    # Sort cleaned_segments by start time to ensure proper order (segment 0 first, then ayahs)
    cleaned_segments.sort(key=lambda x: x['start'])

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
