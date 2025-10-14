
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
    3. Load reference ayahs from "Quran Ayas List.csv" (filter by sura_id)
    4. Connect to SQLite database (quran.db)

    5. Initialize:
    - i = 0 (segment index)
    - ayah_index = 0
    - cleaned_segments = empty list

    WHILE i < total_segments AND ayah_index < total_ayahs:
        
        6. start_time = segments[i].start
        7. merged_text = segments[i].text
        8. end_time = segments[i].end
        
        LOOP FOREVER (until break):
            
            // --- CALCULATE SIMILARITIES ---
            9. Calculate full_similarity between merged_text and current ayah
            
            10. Determine N_CHECK:
                IF next ayah has 3+ words: N = 3
                ELSE IF next ayah has 2 words: N = 2
                ELSE: N = 1
            
            11. Get last N words from merged_text
            12. Get last N words from current ayah
            13. Calculate last_words_similarity
            
            14. Print comparison info
            
            // --- CHECK 1: Last words match? ---
            15. IF last_words_similarity >= 0.7:
                    Finalize current ayah:
                        - Save to database
                        - Add to cleaned_segments list
                    ayah_index = ayah_index + 1
                    BREAK (exit inner loop)
            
            // --- CHECK 2: Next segment starts next ayah? ---
            16. IF next segment exists AND next ayah exists:
                    
                    17. Get first N words from next segment
                    18. Get first N words from next ayah
                    19. Calculate first_words_similarity
                    
                    20. IF first_words_similarity > 0.8:
                            Print "Stop! Next segment starts next ayah"
                            Finalize current ayah WITHOUT merging
                            ayah_index = ayah_index + 1
                            BREAK (exit inner loop)
                    
                    21. ELSE:
                            Merge next segment text into merged_text
                            Remove overlapping words
                            Update end_time to next segment's end
                            i = i + 1
                            Continue loop (go back to step 9)
            
            // --- CHECK 3: No more segments? ---
            22. ELSE (no next segment):
                    Print "End of segments reached"
                    Force finalize current ayah
                    ayah_index = ayah_index + 1
                    BREAK (exit inner loop)
        
        END LOOP
        
        23. i = i + 1

    END WHILE

    24. Close database connection
    25. Save cleaned_segments to corrected_segments_sura_id.json
    26. Print completion message

    RETURN


# KEY HELPER FUNCTIONS
    NORMALIZE_ARABIC(text)
    1. Replace أ, إ, آ with ا
    2. Replace ى with ي
    3. Replace ة with ه
    4. Remove all punctuation and diacritics
    5. Remove extra whitespace
    RETURN normalized_text
        
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
        Skip it (already present)
    ELSE:
        Keep it
    4. Append remaining words to text1
    RETURN merged_text
