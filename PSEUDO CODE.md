# DATA FLOW SUMMARY

    Audio File (WAV)
        ↓
    [TRANSCRIBE]
        ↓
    Segments JSON (timestamped text chunks)
        ↓
    [ALIGN with Reference Ayahs, Remove Overlaps, Correct segments]
        ↓
    Corrected Segments JSON (ayah timestamps)
        ↓
    [SAVE TO DATABASE]
        ↓
    SQLite Database (final ayah timestamps)

## Prepare audio data

    1. Download Quran surahs, surah by surah.
    2. Standardize file names (from 1 to 114), remove leading zeros (001 -> 1), to match "sura_id" in "Quran Ayas List.csv".
    3. Convert to wav, and add them to Quran/audio folder.

## MAIN PROGRAM

    START

    1. Load Whisper AI model (Tarteel Arabic Quran model)
        - Load model configuration from model_config.json
        - Support for both faster-whisper and transformers models
        - Model caching to avoid reloading
        - Device selection: CUDA > MPS (Apple Silicon) > CPU
    2. Get list of audio files to process (currently: ["067.wav"])
    3. Set reciter name = "badr al-turki"

    FOR EACH sura audio file:

        4. Generate unique UUID for this surah.
        5. Save UUID and reciter name to data/current_config.json
        6. Get expected number of ayahs from "Quran Ayas List.csv" for this sura

        7. Set attempt = 1
        8. Set success = FALSE

        WHILE attempt <= MAX_RETRIES (3) AND success == FALSE:

            // --- TRANSCRIPTION PHASE ---
            9. Call TRANSCRIBE_AUDIO(audio_file, processor, model, device)

            // --- ALIGNMENT PHASE ---
            10. Try:
                    Call ALIGN_SEGMENTS(sura_id)
                Catch Exception:
                    Print error and retry

            // --- VALIDATION PHASE ---
            11. Load corrected_segments_{sura_id}.json
            12. Count ayah segments (exclude id=0 and ayah_index=-1)
            13. Compare count with expected_ayahs

            IF aligned_count == expected_ayahs:
                success = TRUE
                Print "✅ All {total_ayas} ayat aligned successfully"
            ELSE:
                Print "⚠️ Only {aligned_count} ayah segments found, expected {expected_ayahs}. Retrying..."
                attempt = attempt + 1
                Sleep 1 second

        IF success == FALSE:
            Print "❌ Failed after {MAX_RETRIES} attempts, skipping this sura"

        14. Run save_to_db.py to commit to database (via subprocess)

    END FOR

    END PROGRAM

# FUNCTION: TRANSCRIBE_AUDIO(audio_file, processor, model, device)

    INPUT: Path to WAV audio file, model components (processor/model/device)
    OUTPUT: segments.json, silences.json

    1. Load UUID from data/current_config.json
    2. Load audio using Pydub (for silence detection)
    3. Load audio using Librosa (for AI model, 16kHz sample rate)
    4. Extract sura_id from filename (e.g., "067.wav" -> "067")

    5. Detect silent parts:
        - Threshold: -30dB
        - Minimum length: 300ms
        - Result: List of [start_ms, end_ms] silence periods
    6. Detect non-silent parts (speech chunks):
        - Same parameters as silence detection
        - Result: List of [start_ms, end_ms] speech periods

    7. Initialize empty segments list
    8. Check if using faster-whisper (processor is None)

    FOR EACH speech chunk (idx, [start_ms, end_ms]):

        9. Convert timestamps to sample indices:
            start_sample = (start_ms / 1000) * sample_rate
            end_sample = (end_ms / 1000) * sample_rate
        10. Extract audio segment samples
        11. Skip if segment is empty

        12. IF using faster-whisper:
                - Save segment to temporary WAV file
                - Call model.transcribe(tmp_file, beam_size=1, language="ar")
                - Extract text from first segment result
                - Delete temporary file
            ELSE (transformers):
                - Prepare input: processor(segment, sampling_rate=16000, return_tensors="pt")
                - Move to device (CUDA/MPS/CPU)
                - Run model.generate(max_new_tokens=128, num_beams=1)
                - Decode: processor.batch_decode(ids, skip_special_tokens=True)

        13. Normalize Arabic text (standardize letters, remove diacritics)

        14. Check for special segment types:
            - IF normalized text matches isti3aza_pattern:
                segment_type = "isti3aza"
                segment_id = 0
                Print "Segment {idx} (Isti'aza): {start}s -> {end}s {text}"
            - ELSE IF normalized text matches basmala_pattern:
                segment_type = "basmala"
                segment_id = 0
                Print "Segment {idx} (Basmala): {start}s -> {end}s {text}"
            - ELSE:
                segment_type = "ayah"
                segment_id = idx
                Print "Segment {idx}: {start}s -> {end}s {text}"

        15. Create segment object:
            {
                "id": segment_id (0 for special, idx for ayahs),
                "sura_id": sura_id (integer),
                "surah_uuid": UUID,
                "start": start_ms / 1000 (seconds, rounded to 2 decimals),
                "end": end_ms / 1000 (seconds, rounded to 2 decimals),
                "text": text.strip(),
                "type": segment_type ("isti3aza", "basmala", or "ayah")
            }

        16. Add segment to list (including special segments)

    END FOR

    17. Create directories: data/segments, data/silences
    18. Save all segments to data/segments/{sura_id}_segments.json
    19. Save all silences to data/silences/{sura_id}_silences.json

    RETURN segments, silences

# FUNCTION: ALIGN_SEGMENTS(sura_id)

    INPUT: Sura ID number (string, e.g., "067")
    OUTPUT: corrected_segments_{sura_id}.json with ayah timestamps

    1. Initialize logging file for this sura
    2. Load UUID and reciter name from data/current_config.json
    3. Load transcribed segments from data/segments/{sura_id}_segments.json
    4. Load silences from data/silences/{sura_id}_silences.json (for buffer calculation)
    5. Load reference ayahs from "data/Quran Ayas List.csv" (filter by sura_id)
    6. Connect to SQLite database (data/quran.db)
    7. Create database table if not exists

    8. Initialize:
        - i = 0 (segment index)
        - ayah_index = 0
        - cleaned_segments = empty list
        - next_ayah_id = 1 (separate counter for ayahs)
        - prev_ayah_end = None (track previous ayah for buffer overlap prevention)

    9. Pre-process special segments:
        FOR EACH segment in segments:
            detected_type = DETECT_SPECIAL_TYPE(segment)
            IF segment.id == 0 OR detected_type is not None:
                special_type = detected_type OR segment.type OR 'unknown'
                Add to cleaned_segments with:
                    - id = 0
                    - ayah_index = -1 (special marker)
                    - type = special_type
                    - All original segment fields (start, end, text, etc.)
                Print "Added {special_type} segment: {start}s -> {end}s"
        END FOR

    10. Convert silences to seconds and sort by start time:
        silences_sec = [(s[0]/1000, s[1]/1000) for s in silences]
        Sort silences_sec by start time

    WHILE i < total_segments AND ayah_index < total_ayahs:

        // Skip special segments during alignment
        11. segment_type = DETECT_SPECIAL_TYPE(segments[i]) OR segments[i].get('type', 'ayah')
            IF segments[i].id == 0 OR segment_type in SPECIAL_SEGMENT_TYPES:
                Print "Skipping segment {i} (type: {segment_type}) during ayah alignment"
                i = i + 1
                Continue to next iteration

        12. start_time = segments[i].start
        13. merged_text = segments[i].text
        14. end_time = segments[i].end
        15. overlap_flag = False
        16. normalized_seg = NORMALIZE_ARABIC(merged_text)

        LOOP FOREVER (until break):

            // --- CALCULATE SIMILARITIES ---
            17. Calculate full_similarity between normalized_seg and NORMALIZE_ARABIC(current_ayah.text)
            18. words_in_ayah = current_ayah.text.strip().split()
            
            19. Determine N_CHECK based on current ayah word count:
                IF len(words_in_ayah) >= 3: N_CHECK = 3
                ELSE IF len(words_in_ayah) == 2: N_CHECK = 2
                ELSE: N_CHECK = 1

            20. Get last N_CHECK words from merged_text
            21. Get last N_CHECK words from current ayah
            22. Calculate last_words_similarity

            23. Print comparison info:
                "--- Comparing segment with ayah ---"
                "Segment ID {id} text: {merged_text}"
                "Ayah index {ayah_index} text: {ayah_text}"
                "Full similarity: {full_sim:.3f}"
                "Last {N_CHECK} words similarity: {last_words_sim:.3f}"

            // --- REQUIRED TOKENS GUARD ---
            24. required_tokens_map = {2: ["ارجع", "فطور"], ...}
                IF ayah_index in required_tokens_map:
                    normalized_seg = NORMALIZE_ARABIC(merged_text)
                    missing_required = False
                    FOR EACH token in required_tokens_map[ayah_index]:
                        IF token not in normalized_seg:
                            missing_required = True
                            BREAK
                    
                    IF missing_required:
                        Print "⏩ Missing required tokens; force merging next segment"
                        IF i + 1 < len(segments):
                            merged_text, overlap_found = REMOVE_OVERLAP(merged_text, segments[i+1].text)
                            IF overlap_found: overlap_flag = True
                            end_time = segments[i+1].end
                            normalized_seg = NORMALIZE_ARABIC(merged_text)
                            i = i + 1
                            Continue loop (go back to step 17)
                        ELSE:
                            Print "⚠️ Last segment but required tokens still missing; proceeding"

            // --- CHECK 1: Last words match + coverage check ---
            25. IF last_words_similarity >= 0.6:
                    seg_word_count = len(NORMALIZE_ARABIC(merged_text).split())
                    ayah_word_count = len(words_in_ayah)
                    coverage_ratio = seg_word_count / max(ayah_word_count, 1)

                    // Guard: if coverage is too low, keep merging
                    IF coverage_ratio < 0.7 AND i + 1 < len(segments):
                        Print "⏩ Similarity hit but coverage {coverage_ratio:.2f} < 0.70; continue merging"
                        // Fall through to merging logic
                    ELSE:
                        Print "✅ Reached end of ayah based on last words similarity and coverage"
                        
                        // Get next ayah's start time (if exists) - initially None
                        next_ayah_start = None
                        
                        // Apply buffers to start and end times
                        buffered_start, buffered_end = APPLY_BUFFERS(
                            start_time, end_time, silences,
                            prev_end=prev_ayah_end,
                            next_start=next_ayah_start,
                            buffer=0.3
                        )
                        
                        // Save to database
                        ayah_data = (UUID, sura_id, ayah_index, buffered_start, buffered_end, reciter_name)
                        INSERT_AYAH_TIMESTAMP(conn, ayah_data)
                        
                        // Add to cleaned_segments
                        cleaned_segments.append({
                            "id": next_ayah_id,
                            "sura_id": sura_id,
                            "ayah_index": ayah_index,
                            "start": buffered_start,
                            "end": buffered_end,
                            "transcribed_text": merged_text,
                            "corrected_text": current_ayah.text,
                            "start_id": segments[i].id,
                            "end_id": segments[i].id,
                            "uuid": UUID
                        })
                        
                        next_ayah_id = next_ayah_id + 1
                        prev_ayah_end = buffered_end
                        ayah_index = ayah_index + 1
                        BREAK (exit inner loop)

            // --- CHECK 2: Next segment starts next ayah? ---
            26. IF i + 1 < len(segments):
                    IF ayah_index + 1 < len(ayahs):
                        // Calculate N_CHECK for next ayah
                        words_in_next_ayah = next_ayah.text.strip().split()
                        N_CHECK = 1
                        IF len(words_in_next_ayah) >= 3: N_CHECK = 3
                        ELSE IF len(words_in_next_ayah) == 2: N_CHECK = 2

                        next_first_seg, _ = GET_FIRST_LAST_WORDS(segments[i+1].text, n=N_CHECK)
                        next_first_ayah, _ = GET_FIRST_LAST_WORDS(next_ayah.text, n=N_CHECK)
                        sim = SIMILARITY(next_first_seg, next_first_ayah)
                        
                        IF sim > 0.6:
                            Print "⚠️ Found next ayah start (using {N_CHECK} words), similarity={sim}. Finalizing current ayah"
                            
                            // Next ayah starts at segments[i+1].start
                            next_ayah_start = segments[i+1].start
                            
                            // Apply buffers
                            buffered_start, buffered_end = APPLY_BUFFERS(
                                start_time, end_time, silences,
                                prev_end=prev_ayah_end,
                                next_start=next_ayah_start,
                                buffer=0.3
                            )
                            
                            // Save to database and cleaned_segments
                            (same as CHECK 1)
                            
                            next_ayah_id = next_ayah_id + 1
                            prev_ayah_end = buffered_end
                            ayah_index = ayah_index + 1
                            BREAK (exit inner loop)

                    // --- CHECK 3: Silence gap detection ---
                    27. silence_gap = FIND_SILENCE_GAP_BETWEEN(
                            end_time, segments[i+1].start, silences_sec, min_gap=0.18)
                    
                    28. IF silence_gap is not None:
                            // Verify with textual check
                            textual_boundary = False
                            IF ayah_index + 1 < len(ayahs):
                                // Determine N_CHECK for next ayah
                                words_in_next_ayah = next_ayah.text.strip().split()
                                N_CHECK = 1
                                IF len(words_in_next_ayah) >= 3: N_CHECK = 3
                                ELSE IF len(words_in_next_ayah) == 2: N_CHECK = 2
                                
                                next_first_seg, _ = GET_FIRST_LAST_WORDS(segments[i+1].text, n=N_CHECK)
                                next_first_ayah, _ = GET_FIRST_LAST_WORDS(next_ayah.text, n=N_CHECK)
                                next_sim = SIMILARITY(next_first_seg, next_first_ayah)
                                IF next_sim > 0.6:
                                    textual_boundary = True

                            IF textual_boundary:
                                gap_start, gap_end = silence_gap
                                Print "🧭 Silence gap from {gap_start:.2f}s to {gap_end:.2f}s + textual cues; treating as ayah boundary"
                                
                                // Constrain end buffer to gap start
                                next_ayah_start = gap_start
                                
                                buffered_start, buffered_end = APPLY_BUFFERS(
                                    start_time, end_time, silences,
                                    prev_end=prev_ayah_end,
                                    next_start=next_ayah_start,
                                    buffer=0.3
                                )
                                
                                // Save to database and cleaned_segments
                                (same as CHECK 1)
                                
                                next_ayah_id = next_ayah_id + 1
                                prev_ayah_end = buffered_end
                                ayah_index = ayah_index + 1
                                BREAK (exit inner loop)

                    // --- No boundary detected, merge segments ---
                    29. Print "🔄 Merging segment {segments[i].id} with next segment {segments[i+1].id}"
                        merged_text, overlap_found = REMOVE_OVERLAP(merged_text, segments[i+1].text)
                        IF overlap_found: overlap_flag = True
                        end_time = segments[i+1].end
                        i = i + 1
                        Continue loop (go back to step 17)

            // --- CHECK 4: No more segments? ---
            30. ELSE (no next segment):
                    Print "❌ End of segments reached. Finalizing last ayah"
                    
                    // No next ayah, so end buffer is not constrained
                    next_ayah_start = None
                    
                    // Apply buffers
                    buffered_start, buffered_end = APPLY_BUFFERS(
                        start_time, end_time, silences,
                        prev_end=prev_ayah_end,
                        next_start=next_ayah_start,
                        buffer=0.3
                    )
                    
                    // Save to database and cleaned_segments
                    (same as CHECK 1)
                    
                    next_ayah_id = next_ayah_id + 1
                    prev_ayah_end = buffered_end
                    ayah_index = ayah_index + 1
                    BREAK (exit inner loop)

        END LOOP

        31. i = i + 1

    END WHILE

    32. Close database connection
    33. Sort cleaned_segments by start time (ensures special segments first, then ayahs)
    34. Save cleaned_segments to data/corrected_segments/corrected_segments_{sura_id}.json
    35. Print "✅ All segments for Sura {sura_id} processed and saved"

    RETURN

# KEY HELPER FUNCTIONS

    NORMALIZE_ARABIC(text)
        INPUT: Arabic text string
        OUTPUT: Normalized text string
        
        1. Replace أ, إ, آ with ا (all alef variants)
        2. Replace ى with ي (ya variants)
        3. Replace ة with ه (ta marbuta to ha)
        4. Remove all punctuation and diacritics: re.sub(r"[^\w\s]", "", text)
        5. Remove extra whitespace: re.sub(r"\s+", " ", text).strip()
        RETURN normalized_text

    DETECT_SPECIAL_TYPE(segment)
        INPUT: Segment dictionary with 'type' and 'text' fields
        OUTPUT: "isti3aza", "basmala", or None
        
        1. seg_type = segment.get("type")
        2. IF seg_type in SPECIAL_SEGMENT_TYPES {"isti3aza", "basmala", "basmalah"}:
            IF seg_type == "basmalah": RETURN "basmala"
            ELSE: RETURN seg_type
        
        3. norm_text = NORMALIZE_ARABIC(segment.get("text", ""))
        4. IF basmala_pattern.search(norm_text):
            RETURN "basmala"
        5. IF isti3aza_pattern.search(norm_text):
            RETURN "isti3aza"
        6. RETURN None

    SIMILARITY(a, b)
        INPUT: Two text strings
        OUTPUT: Similarity ratio (0.0 to 1.0)
        
        1. Use SequenceMatcher(None, a, b).ratio()
        RETURN ratio

    GET_FIRST_LAST_WORDS(text, n=1)
        INPUT: Text string, number of words n
        OUTPUT: (first_n_words, last_n_words)
        
        1. words = NORMALIZE_ARABIC(text).split()
        2. first = " ".join(words[:n]) IF len(words) >= n ELSE " ".join(words)
        3. last = " ".join(words[-n:]) IF len(words) >= n ELSE " ".join(words)
        RETURN first, last

    REMOVE_OVERLAP(text1, text2)
        INPUT: Two text strings to merge
        OUTPUT: (merged_text, overlap_found boolean)
        
        1. words1 = NORMALIZE_ARABIC(text1).split()
        2. words2 = text2.split() (keep original for output)
        3. count1 = Counter(words1) (word frequency map)
        4. cleaned_words2 = empty list
        
        5. FOR EACH word in words2:
            IF count1[NORMALIZE_ARABIC(word)] > 0:
                count1[NORMALIZE_ARABIC(word)] -= 1
                Continue (skip overlapping word)
            ELSE:
                Append word to cleaned_words2
        
        6. IF cleaned_words2 is empty:
            RETURN text1.strip(), True
        
        7. merged_text = text1.strip() + " " + " ".join(cleaned_words2).strip()
        RETURN merged_text, True

    APPLY_BUFFERS(start_time, end_time, silences, prev_end=None, next_start=None, buffer=0.3)
        INPUT: 
            - start_time, end_time: Original timestamps in seconds
            - silences: List of [start_ms, end_ms] silence periods
            - prev_end: Previous ayah's end time (to prevent overlap)
            - next_start: Next ayah's start time (to prevent overlap)
            - buffer: Buffer duration in seconds (default 0.3)
        OUTPUT: (new_start_time, new_end_time) with buffers applied
        
        1. new_start = start_time
        2. new_end = end_time
        
        3. IF silences is empty: RETURN new_start, new_end
        
        4. Convert silences to seconds and sort:
            silences_sec = [(s[0]/1000, s[1]/1000) for s in silences]
            Sort silences_sec by start time
        
        // Extend start backward into preceding silence
        5. best_silence_before = None
        6. FOR EACH (silence_start, silence_end) in silences_sec:
            IF silence_end <= start_time:
                IF best_silence_before is None OR silence_end > best_silence_before[1]:
                    best_silence_before = (silence_start, silence_end)
            ELSE IF silence_start > start_time:
                BREAK (passed start_time)
        
        7. IF best_silence_before is not None:
            silence_start, silence_end = best_silence_before
            available_buffer = start_time - silence_start
            buffer_to_apply = min(buffer, available_buffer)
            buffer_start = start_time - buffer_to_apply
            
            IF prev_end is None:
                new_start = buffer_start
            ELSE IF buffer_start >= prev_end:
                new_start = buffer_start
            ELSE IF prev_end < start_time:
                new_start = max(buffer_start, prev_end)
            // else: prev_end >= start_time (shouldn't happen), keep original
        
        // Extend end forward into following silence
        8. best_silence_after = None
        9. FOR EACH (silence_start, silence_end) in silences_sec:
            IF silence_start >= end_time:
                IF best_silence_after is None OR silence_start < best_silence_after[0]:
                    best_silence_after = (silence_start, silence_end)
            ELSE IF silence_end < end_time:
                Continue (skip silences before end_time)
        
        10. IF best_silence_after is not None:
            silence_start, silence_end = best_silence_after
            available_buffer = silence_end - end_time
            buffer_to_apply = min(buffer, available_buffer)
            buffer_end = end_time + buffer_to_apply
            
            IF next_start is None:
                new_end = buffer_end
            ELSE IF buffer_end <= next_start:
                new_end = buffer_end
            ELSE IF next_start > end_time:
                new_end = min(buffer_end, next_start)
            // else: next_start <= end_time (shouldn't happen), keep original
        
        RETURN new_start, new_end

    FIND_SILENCE_GAP_BETWEEN(current_end, next_start, silences_sec, min_gap=0.18)
        INPUT: 
            - current_end: End time (sec) of current segment
            - next_start: Start time (sec) of next segment
            - silences_sec: List of (start, end) tuples in seconds, sorted by start
            - min_gap: Minimum silence duration to qualify (default 0.18 sec)
        OUTPUT: (silence_start, silence_end) tuple or None
        
        1. IF silences_sec is empty OR next_start is None:
            RETURN None
        
        2. FOR EACH (silence_start, silence_end) in silences_sec:
            IF silence_end <= current_end:
                Continue (skip silences before current segment)
            IF silence_start >= next_start:
                BREAK (passed next segment)
            
            // Silence fully between current_end and next_start
            IF silence_start >= current_end AND silence_end <= next_start:
                IF (silence_end - silence_start) >= min_gap:
                    RETURN (silence_start, silence_end)
        
        3. RETURN None

    LOAD_MODEL()
        INPUT: None (reads from model_config.json)
        OUTPUT: (processor, model, device) or (model, device) for faster-whisper
        
        1. Load model_config.json to get selected model and config
        2. Check model cache - if same model already loaded, RETURN cached model
        3. Determine model type (faster-whisper or transformers)
        
        4. IF faster-whisper:
            - Import WhisperModel from faster_whisper
            - device = "cuda" IF CUDA available ELSE "cpu"
            - compute_type = "float16" IF CUDA ELSE "int8"
            - model = WhisperModel(model_id, device=device, compute_type=compute_type)
            - Cache model
            - RETURN (model, device)
        
        5. ELSE (transformers):
            - processor = AutoProcessor.from_pretrained(model_id)
            - Determine device: CUDA > MPS (Apple Silicon) > CPU
            - torch_dtype = float16 IF CUDA ELSE float32
            - model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype, low_cpu_mem_usage=True)
            - Move model to device
            - model.eval()
            - IF torch.compile available AND device != MPS:
                model = torch.compile(model, mode="reduce-overhead")
            - Cache model
            - RETURN (processor, model, device)

# DATABASE OPERATIONS

    CREATE_CONNECTION(db_file="data/quran.db")
        INPUT: Database file path
        OUTPUT: SQLite connection object
        
        1. RETURN sqlite3.connect(db_file)

    CREATE_TABLE(conn)
        INPUT: SQLite connection
        OUTPUT: None (creates table if not exists)
        
        1. Execute CREATE TABLE IF NOT EXISTS ayah_timestamps:
            - id INTEGER PRIMARY KEY
            - recitation_uuid TEXT NOT NULL
            - surah_num INTEGER NOT NULL
            - ayah_num INTEGER NOT NULL
            - start_time REAL NOT NULL
            - end_time REAL NOT NULL
            - reciter_name TEXT
        2. Commit transaction

    INSERT_AYAH_TIMESTAMP(conn, ayah_data)
        INPUT: 
            - conn: SQLite connection
            - ayah_data: Tuple (recitation_uuid, surah_num, ayah_num, start_time, end_time, reciter_name)
        OUTPUT: None (inserts record)
        
        1. Execute INSERT INTO ayah_timestamps VALUES (?, ?, ?, ?, ?, ?)
        2. Commit transaction

# LOGGING SYSTEM

    INIT_LOGGING_FILE(sura_id)
        INPUT: Sura ID number
        OUTPUT: Path to log file
        
        1. Load reciter_name and surah_uuid from current_config.json
        2. Get surah_name from SURAH_NAMES mapping
        3. Create directory: data/{reciter_name-uuid}/{sura_id-surah_name}/
        4. Create CSV file: Logging.csv with headers:
            ["sura_id", "ayah_index", "ayah_text", "model_text", 
             "start_time", "end_time", "similarity_score", 
             "status", "notes", "overlap_status"]
        5. RETURN log file path

    LOG_RESULT(sura_id, ayah_index, ayah_text, model_text, 
               start_time, end_time, similarity_score, 
               status, notes="", overlap_status="")
        INPUT: All ayah alignment details
        OUTPUT: None (appends to log file)
        
        1. Get log file path for sura_id
        2. Append row to CSV with all provided data
