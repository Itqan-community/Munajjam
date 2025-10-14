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