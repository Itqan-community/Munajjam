import os
import subprocess
import json
import time
import uuid
import pandas as pd

from src.transcribe import transcribe, load_model  # Import custom transcription functions
from src import config  # Import configuration module
from src.align_segments import clean_and_merge_segments

# Path to folder containing Quran audio files
AUDIO_FOLDER = r"Quran/audio"

# CSV file containing list of Quran ayahs
QURAN_AYAS_CSV = r"data/Quran Ayas List.csv"

# Maximum number of retries if processing fails
MAX_RETRIES = 3  

# Function to get total number of ayahs in a given sura
def get_total_ayas(sura_id):
    df = pd.read_csv(QURAN_AYAS_CSV)  # Load CSV into a DataFrame
    return len(df[df["sura_id"] == int(sura_id)])  # Count ayahs for the given sura

def main():
    # Load transcription model, processor, and device (CPU/GPU)
    # Handle both faster-whisper (returns model, device) and transformers (returns processor, model, device)
    result = load_model()
    if len(result) == 2:
        model, device = result
        processor = None  # faster-whisper doesn't use processor
    else:
        processor, model, device = result

    # Get list of all .wav files in AUDIO_FOLDER and sort them
    all_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")])

    # Specify target sura files to process (currently only 67.wav available)
    target_files = ["067.wav"]
#
    # Name of the reciter
    reciter_name = "badr al-turki"

    # Process each target audio file
    for wav_file in target_files:
        audio_path = os.path.join(AUDIO_FOLDER, wav_file)  # Full path to the audio file
        sura_id = os.path.splitext(wav_file)[0]  # Extract sura ID from filename
        print(f"\nProcessing Sura {sura_id} ({wav_file})...")

        # Generate a unique UUID for this recitation
        surah_uuid = str(uuid.uuid4())
        config.SURAH_UUID = surah_uuid
        print(f"Generated UUID: {surah_uuid}")

        # Save current recitation configuration
        config_data = {
            "RECITER_NAME": reciter_name,
            "SURAH_UUID": surah_uuid
        }

        with open(os.path.join("data", "current_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        # Get the total number of ayahs in this sura
        total_ayas = get_total_ayas(sura_id)
        attempt = 1
        success = False
        print("TOTAL AYAHS :", total_ayas)

        # Retry loop in case of failures
        while attempt <= MAX_RETRIES and not success:
            print(f"--- Attempt {attempt} ---")

            # Run transcription
            transcribe(audio_path, processor=processor, model=model, device=device)

            # -------------------
            # Run segment aligning script for the current sura
            # -------------------
            try: 
                clean_and_merge_segments(sura_id)
            except Exception as e:
                # If aligning fails, retry
                print(f"❌ align_segments.py failed for Sura {sura_id} with error: {e}")
                attempt += 1
                time.sleep(1)
                continue

            # -------------------
            # Check if the corrected segments file exists and is complete
            # -------------------
            corrected_file = os.path.join("data", "corrected_segments", f"corrected_segments_{sura_id}.json")
            if os.path.exists(corrected_file):
                with open(corrected_file, encoding="utf-8") as f:
                    current_segments = json.load(f)

                # Count only ayahs (exclude Isti'aza/Basmala segments with id 0 or ayah_index -1)
                ayah_segments = [seg for seg in current_segments 
                                if seg.get('id') != 0 and seg.get('ayah_index', -1) >= 0]

                # If all ayahs are aligned, mark success
                if len(ayah_segments) == total_ayas:
                    success = True
                    print(f"✅ All {total_ayas} ayat aligned successfully for Sura {sura_id} (uuid={surah_uuid})")
                    # time.sleep(30)  # Wait before next sura
                else:
                    # If segments are incomplete, retry
                    print(f"⚠️ Only {len(ayah_segments)} ayah segments found, expected {total_ayas}. Retrying...")
                    attempt += 1
                    time.sleep(1)
            else:
                # If corrected file does not exist, retry
                print(f"⚠️ {corrected_file} not found. Retrying...")
                attempt += 1
                time.sleep(1)

        # If all retries fail, skip to next sura
        if not success:
            print(f"❌ Failed to fully process Sura {sura_id} after {MAX_RETRIES} attempts. Skipping to next.")

        # -------------------
        # Save transcription results to database
        # -------------------
        try:
            subprocess.run(["python", "src/save_to_db.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ save_to_db.py failed for Sura {sura_id}")

# Entry point
if __name__ == "__main__":
    main()