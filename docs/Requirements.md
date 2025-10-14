#Quran Recitation Synchronization 

## Objective 

Build an AI system that automatically synchronizes Quranic verses (ayahs) with the recitation audio by generating accurate timestamps for the start and end of each ayah. 

## Glossary 

- Audio Dataset: ADS. 

## Input 

Audio Input 

Quran recitation recordings (e.g., .mp3 / .wav). 

Different reciters and recitation styles (speed, tajweed). 

Plain audio divided by Surah. 

## Output 

Database (SQLite) 

TABLE ayah_timestamps  

id 

Recitation _uuid 

surah_num 

ayah_num 

start_time 

 

end_time 

reciter_name 

 

### User Story 

As a Quranic Content Creator, 

I want to Quran Audio to be mapped out with Quran Script chunked by Ayah,  

So that I would be able to add new recitations tracks to the database without having to do this manually. 

 

### Acceptance Criteria 

Make sure that the user can provide us a new path for new recitation so that the system can start detecting in the next run that a new recitation has been added and that it needs to be processed. 

Make sure that the processing of the recitation is well illustrated as an algorithm and is easy to maintain and extend: 

Make sure that the system automatically scrapes the folder that contains the audio files. 

Make sure that the system starts processing one Surah at a time. 

Make sure that the system starts processing one Ayah per current Surah at a time. 

Make sure that the system doesn’t skip an Ayah before it saves the data of the processing of the current one properly if not, the system should not continue processing otherwise we would be waiting while the data is not being saved. 

Make sure to create a test function to check the current index of the processed Ayahs as quantification algorithm to double check that the data of the current ayah has been saved. 

Make sure that the algorithm moves to the next Surah once we double check that the current Surah’s processing has been done successfully by comparing the number of created timestamps (Surah’s Records count in Ayahs_Timestamp database entity which is the total number of Ayah’s of the currently under-processing Surah in the Original Quran Script. This will double check in quantified approach that the processing is being done successfully. 

Make sure that there is a logging algorithm that logges all processing footprint in a file that is being divided by Surah per each recitation. 

Make sure that the logging files are being organized by directories in the following structure: Reciration_Name-UUID - Directory. Surah_Number-Surah_Name – Directory. Logging File.csv 

Make sure that the logging file internal structure (Schema) follows a specific small data structure not just a list of operations in raw text. 

Make sure that the system can use word-level timestamps from the ASR output for precise boundary detection. 

Make sure that Ayah based audio files (if the Quran Audio Library is divided by Ayah's each Ayah in a separate audio clip) that it gets mapped directly to Ayah's because it is already chunked before. (Future Version) 

Make sure that the alignment is performed by sequentially matching transcribed words to the reference words of the Ayah. 

Make sure that normalization of both ASR and canonical text is performed before alignment to ensure correct word matching despite typographical differences (misspellings). 

Make sure that the alignment loop handles overlap detection and removal simultaneously by strictly comparing the current transcribed word against the expected word and skipping any redundant transcribed words (overlap/noise) if a sequence mismatch is detected. 

 

#Steps 

Download Surahs 

Download each Surah separately (quran/001_al_fatiha.mp3, quran/002_al_baqara.mp3, …). 

Extract Ayah-Quran-Script's Features: 

Hold first and last words as Ayah Boundaries. 

Loop and Hold the list of words within the Ayah in a separate dictionary object with Key: Word String Value: AdsOccursCount. 

Detect Silence 

Run silence detection on each Surah. 

Long silence is usually a sign of an Ayah ending. 

Save silence intervals into JSON. 

Run ASR Model 

Use a model to transcribe Surah audio. Output: transcription + Word-Level Timestamps. 

Detect Overlap Algorithm: 

Loop on words of the segment and do the following: 

If a word is being found increment the no# of occurrences by one (if there is an overlap the count will be more than one for the overlapped words). 

Keep processing all words until we hit the last word of the Ayah. 

Add up the data of the overlapping words to a data file/database. 

 

Hold segment[-1]  

Check if there is more than one segment, if no --> skip 

If yes : 

Hold segment[-2] 

Loop on each segment and store occ_cnt of each word  

Store in overlap_db (surah_id, most_likely, ayah_num, json overlapped_segments_content) 

Does overlapping have gaps? Like triangle relationship 

E.g. Segment 1 overlaps with segment 3. 

E.g Segment 1 overlaps with segment 2. 

E.g Segment 2 overlaps with segment 3. 

 

Normalization 

Normalize both ASR text and Canonical Ayah text before comparison (Remove diacritics, unify Alef forms, remove extra spaces). This step corrects for typographical issues and ensures a high similarity score for correct matches. 

Alignment 

For Surah-level audio:  

Sequential Match: Iterate through the canonical words and attempt to match them one-by-one with the transcribed words. 

Overlap/Misalignment Handling: If the transcribed word does not match the expected canonical word, it is treated as noise or a redundant/overlapped word and is skipped (the transcribed word pointer advances, but the canonical pointer does not). 

Boundary Commitment: The Ayah's final start_time is the start of the first successfully matched transcribed word, and the end_time is the end of the last successfully matched transcribed word. 

Similarity Check 

Save aligned Ayahs to aligned.json with start and end derived from word-level boundaries. 

Save Results 

Save aligned Ayahs into aligned.json: 

"id": 7, 

    "sura_id": 1, 

    "index": 7, 

    "text": "صِرَٰطَ ٱلَّذِينَ أَنۡعَمۡتَ عَلَيۡهِمۡ غَيۡرِ ٱلۡمَغۡضُوبِ عَلَيۡهِمۡ وَلَا ٱلضَّآلِّينَ", 

    "start": 35.2, 

    "end": 49.7 

 
 

Save to Database 

Store each Ayah alignment in an SQLite table. 

 

 

### Validation Step 

For each Surah, check that: 

Number of processed Ayahs (with timestamps) == number of Ayahs in Quran CSV. 

If mismatch → flag the Surah for manual review. 

Generate Logging.csv (per Surah, per Reciter) 

Columns: 

ayah_index → Ayah order inside the Surah (1,2,3…). 

ayah_text → The original Ayah text from the Quran CSV. 

model_text → The text recognized/generated by the ASR model. 

start_time → Timestamp (seconds) when Ayah starts. 

end_time → Timestamp (seconds) when Ayah ends. 

similarity_score → Matching score between ayah_text and model_text. 

status → Success if timestamps + similarity are valid, Fail otherwise. 

overlap_status → detected and removed/ no overlap. 

notes → Extra comments (e.g., “low similarity”, “missing timestamps”). 

 

 

 _____________________________________________

 ## Requirements

- Make sure that the user can provide us a new path for new recitation so that the system can start detecting in the next run that a new recitation has been added and that it needs to be processed. 

- Make sure that the processing of the recitation is well illustrated as an algorithm and is easy to maintain and extend: 

- Make sure that the system automatically scrapes the folder that contains the audio files. 

- Make sure that the system starts processing one Surah at a time. 

- Make sure that the system starts processing one Ayah per current Surah at a time. 

- Make sure that the system doesn’t skip an Ayah before it saves the data of the processing of the current one properly if not, the system should not continue processing otherwise we would be waiting while the data is not being saved. 

- Make sure to create a test function to check the current index of the processed Ayahs as quantification algorithm to double check that the data of the current ayah has been saved. 

- Make sure that the algorithm moves to the next Surah once we double check that the current Surah’s processing has been done successfully by comparing the number of created timestamps (Surah’s Records count in Ayahs_Timestamp database entity which is the total number of Ayah’s of the currently under-processing Surah in the Original Quran Script. This will double check in quantified approach that the processing is being done successfully. 

- Make sure that there is a logging algorithm that logges all processing footprint in a file that is being divided by Surah per each recitation. 

- Make sure that the logging files are being organized by directories in the following structure: Reciration_Name-UUID - Directory. Surah_Number-Surah_Name – Directory. Logging File.csv 

- Make sure that the logging file internal structure (Schema) follows a specific small data structure not just a list of operations in raw text. 

- Make sure that the system can use word-level timestamps from the ASR output for precise boundary detection. 
- Make sure that Ayah based audio files (if the Quran Audio Library is divided by Ayah's each Ayah in a separate audio clip) that it gets mapped directly to Ayah's because it is already chunked before. (Future Version) 

- Make sure that the alignment is performed by sequentially matching transcribed words to the reference words of the Ayah. 

- Make sure that normalization of both ASR and canonical text is performed before alignment to ensure correct word matching despite typographical differences (misspellings). 

- Make sure that the alignment loop handles overlap detection and removal simultaneously by strictly comparing the current transcribed word against the expected word and skipping any redundant transcribed words (overlap/noise) if a sequence mismatch is detected. 