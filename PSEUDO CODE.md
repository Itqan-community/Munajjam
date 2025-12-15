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

    1.Download Quran surahs , surah by surah.
    2.Standardize file names (from 1 to 114), remove leading zeros (001 -> 1), to match "sura_id" in "Quran Ayas List.csv".
    3.convert to wav, and add them to quran_wav folder.

## MAIN PROGRAM

    START

    1. Load Whisper AI model (Tarteel Arabic Quran model)
    2. Get list of audio files to process (suras 6, 5, 4, 3)
    3. Set reciter name = "عمر النبراوي"

    FOR EACH sura audio file:

        4. Generate unique UUID for this surah.
        5. Save UUID and reciter name to config.json to extract and use them in the other scripts.
        6. Get expected number of ayahs from "Quran Ayas List.csv" for this sura

        7. Set attempt = 1
        8. Set success = FALSE

        WHILE attempt <= 3 AND success == FALSE:

            // --- TRANSCRIPTION PHASE ---
            9. Call TRANSCRIBE_AUDIO(audio_file)

            // --- ALIGNMENT PHASE ---
            10. Call ALIGN_SEGMENTS(sura_id)

            // --- VALIDATION PHASE ---
            11. Load corrected_segments.json
            12. Count how many ayahs were aligned

            IF aligned_count == expected_ayahs:
                success = TRUE
                Print "Success! All ayahs aligned"
            ELSE:
                Print "Incomplete alignment, retrying..."
                attempt = attempt + 1

        IF success == FALSE:
            Print "Failed after 3 attempts, skipping this sura"

        13. Run save_to_db.py to commit to database

    END FOR

    END PROGRAM

# FUNCTION: TRANSCRIBE_AUDIO(audio_file)

    INPUT: Path to WAV audio file
    OUTPUT: segments.json, silences.json

    1. Load UUID from config.json
    2. Load audio using Pydub (for silence detection)
    3. Load audio using Librosa (for AI model, 16kHz)
    4. Extract sura_id from filename (e.g., "75.wav" -> 75)

    5. Detect silent parts (threshold: -30dB, min length: 300ms)
    6. Detect non-silent parts (speech chunks)

    7. Initialize empty segments list

    FOR EACH speech chunk:

        8. Extract audio samples for this chunk
        9. Skip if chunk is empty

        10. Prepare audio for model
        11. Run Whisper model inference
        12. Decode output to Arabic text

        13. Normalize Arabic text (standardize letters, remove diacritics)

        14. IF text contains "أعوذ بالله من الشيطان الرجيم":
                Skip this segment (it's Isti'aza, not ayah)
                Continue to next chunk

        15. IF text contains "بسم الله الرحمن الرحيم" AND sura_id != 1:
                Skip this segment (Basmala not part of ayah text)
                Continue to next chunk

        16. Create segment object:
            - id, sura_id, UUID
            - start time (seconds)
            - end time (seconds)
            - transcribed text

        17. Add segment to list
        18. Print segment info

    END FOR

    19. Save all segments to segments/sura_id_segments.json
    20. Save all silences to silences/sura_id_silences.json

    RETURN segments, silences

# FUNCTION: ALIGN_SEGMENTS(sura_id)

    INPUT: Sura ID number
    OUTPUT: corrected_segments.json with ayah timestamps

    1. Load UUID and reciter name from config.json
    2. Load transcribed segments from segments.json
    3. Load silences from silences.json (for buffer calculation)
    4. Load reference ayahs from "Quran Ayas List.csv" (filter by sura_id)
    5. Connect to SQLite database (quran.db)

    6. Initialize:
        - i = 0 (segment index)
        - ayah_index = 0
        - cleaned_segments = empty list
        - next_ayah_id = 1 (separate counter for ayahs)
        - prev_ayah_end = None (track previous ayah for buffer overlap prevention)

    7. Pre-process special segments:
        FOR EACH segment in segments:
            IF segment.id == 0 OR segment.type in ["isti3aza", "basmala"]:
                Add to cleaned_segments with:
                    - id = 0
                    - ayah_index = -1 (special marker)
                    - type = detected type
                Print "Added [type] segment: [start]s -> [end]s"
        END FOR

    8. Convert silences to seconds and sort by start time

    WHILE i < total_segments AND ayah_index < total_ayahs:

        // Skip special segments during alignment
        9. IF current segment is special (id=0 or type in special types):
                i = i + 1
                Continue to next iteration

        10. start_time = segments[i].start
        11. merged_text = segments[i].text
        12. end_time = segments[i].end

        LOOP FOREVER (until break):

            // --- CALCULATE SIMILARITIES ---
            13. Calculate full_similarity between merged_text and current ayah

            14. Determine N_CHECK based on current ayah word count:
                IF ayah has 3+ words: N = 3
                ELSE IF ayah has 2 words: N = 2
                ELSE: N = 1

            15. Get last N words from merged_text
            16. Get last N words from current ayah
            17. Calculate last_words_similarity

            18. Print comparison info

            // --- REQUIRED TOKENS GUARD ---
            19. IF current ayah has required tokens (e.g., ayah 2 needs ["ارجع", "فطور"]):
                    IF any required token is missing from merged_text:
                        Force merge next segment
                        Continue loop

            // --- CHECK 1: Last words match + coverage check ---
            20. IF last_words_similarity >= 0.6:
                    Calculate coverage_ratio = len(merged_words) / len(ayah_words)

                    IF coverage_ratio >= 0.7 OR no more segments:
                        // Apply smart buffer system
                        Determine next_ayah_start (if next segment exists)

                        Call APPLY_BUFFERS(start_time, end_time, silences,
                                          prev_ayah_end, next_ayah_start, buffer=0.3)
                        Get buffered_start, buffered_end

                        Finalize current ayah:
                            - Save to database with buffered timestamps
                            - Add to cleaned_segments list with id=next_ayah_id

                        next_ayah_id = next_ayah_id + 1
                        prev_ayah_end = buffered_end
                        ayah_index = ayah_index + 1
                        BREAK (exit inner loop)
                    ELSE:
                        Print "Coverage too low, continue merging"
                        // Fall through to merging logic

            // --- CHECK 2: Next segment starts next ayah? ---
            21. IF next segment exists AND next ayah exists:

                    // Calculate N_CHECK for next ayah
                    22. Determine N_CHECK based on next ayah word count

                    23. Get first N words from next segment
                    24. Get first N words from next ayah
                    25. Calculate first_words_similarity

                    26. IF first_words_similarity > 0.6:
                            Print "Next segment starts next ayah"

                            // Apply smart buffer system
                            next_ayah_start = next_segment.start
                            Call APPLY_BUFFERS(start_time, end_time, silences,
                                              prev_ayah_end, next_ayah_start, buffer=0.3)
                            Get buffered_start, buffered_end

                            Finalize current ayah with buffered timestamps
                            next_ayah_id = next_ayah_id + 1
                            prev_ayah_end = buffered_end
                            ayah_index = ayah_index + 1
                            BREAK (exit inner loop)

                    // --- CHECK 3: Silence gap detection ---
                    27. Call FIND_SILENCE_GAP(end_time, next_segment.start, silences, min_gap=0.18)

                    28. IF silence gap found:
                            // Verify with textual check
                            IF next segment likely starts next ayah (similarity > 0.6):
                                Print "Silence gap + textual cues = ayah boundary"

                                // Constrain end buffer to gap start
                                Call APPLY_BUFFERS with next_start = gap_start

                                Finalize current ayah with buffered timestamps
                                next_ayah_id = next_ayah_id + 1
                                prev_ayah_end = buffered_end
                                ayah_index = ayah_index + 1
                                BREAK (exit inner loop)

                    // --- No boundary detected, merge segments ---
                    29. Merge next segment text into merged_text
                        Call REMOVE_OVERLAP to clean duplicate words
                        Update end_time to next segment's end
                        i = i + 1
                        Continue loop (go back to step 13)

            // --- CHECK 4: No more segments? ---
            30. ELSE (no next segment):
                    Print "End of segments reached"

                    // Apply buffer without next constraint
                    Call APPLY_BUFFERS(start_time, end_time, silences,
                                      prev_ayah_end, None, buffer=0.3)

                    Force finalize current ayah with buffered timestamps
                    next_ayah_id = next_ayah_id + 1
                    prev_ayah_end = buffered_end
                    ayah_index = ayah_index + 1
                    BREAK (exit inner loop)

        END LOOP

        31. i = i + 1

    END WHILE

    32. Sort cleaned_segments by start time (ensures special segments first, then ayahs)
    33. Close database connection
    34. Save cleaned_segments to corrected_segments_sura_id.json
    35. Print completion message

    RETURN

# KEY HELPER FUNCTIONS

    NORMALIZE_ARABIC(text)
        1. Replace أ, إ, آ with ا
        2. Replace ى with ي
        3. Replace ة with ه
        4. Remove all punctuation and diacritics
        5. Remove extra whitespace
        RETURN normalized_text

    DETECT_SPECIAL_TYPE(segment)
        1. Check if segment.type in ["isti3aza", "basmala", "basmalah"]
        2. If yes, normalize spelling and return canonical type
        3. Otherwise, check text patterns:
           - IF matches "اعوذ بالله من الشيطان الرجيم": return "isti3aza"
           - IF matches "(?:ب\s*س?م?\s*)?الله\s*الرحمن\s*الرحيم": return "basmala"
        4. Return None if not special

    APPLY_BUFFERS(start_time, end_time, silences, prev_end, next_start, buffer=0.3)
        INPUT: Original timestamps, silence periods, constraints, buffer duration
        OUTPUT: Buffered timestamps

        1. Convert silences from milliseconds to seconds and sort
        2. new_start = start_time, new_end = end_time

        // Extend start backward into preceding silence
        3. Find best silence period that ends before/at start_time
        4. IF found:
               Calculate available_buffer = start_time - silence_start
               buffer_to_apply = min(buffer, available_buffer)
               buffer_start = start_time - buffer_to_apply

               IF prev_end is None OR buffer_start >= prev_end:
                   new_start = buffer_start
               ELSE IF prev_end < start_time:
                   new_start = max(buffer_start, prev_end)

        // Extend end forward into following silence
        5. Find best silence period that starts at/after end_time
        6. IF found:
               Calculate available_buffer = silence_end - end_time
               buffer_to_apply = min(buffer, available_buffer)
               buffer_end = end_time + buffer_to_apply

               IF next_start is None OR buffer_end <= next_start:
                   new_end = buffer_end
               ELSE IF next_start > end_time:
                   new_end = min(buffer_end, next_start)

        RETURN new_start, new_end

    FIND_SILENCE_GAP(current_end, next_start, silences_sec, min_gap=0.18)
        INPUT: End of current segment, start of next segment, silence periods
        OUTPUT: (gap_start, gap_end) or None

        1. FOR EACH silence in silences_sec:
               IF silence ends before current_end: skip
               IF silence starts after next_start: break

               IF silence is fully between current_end and next_start:
                   IF (silence_end - silence_start) >= min_gap:
                       RETURN (silence_start, silence_end)

        2. RETURN None

    CALCULATE_SIMILARITY(text1, text2)
        1. Normalize both texts
        2. Use SequenceMatcher to compare
        3. Return ratio (0.0 to 1.0)

    GET_FIRST_LAST_WORDS(text, n)
        1. Normalize text
        2. Split into words
        3. Extract first n words
        4. Extract last n words
        RETURN first_words, last_words

    REMOVE_OVERLAP(text1, text2)
        1. Split text1 into words, count occurrences
        2. Split text2 into words
        3. For each word in text2:
           IF word exists in text1:
               Decrement counter and skip (already present)
           ELSE:
               Keep it
        4. Append remaining words to text1
        RETURN merged_text, overlap_found
