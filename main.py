import os
import subprocess
import json
import time
import uuid

from transcribe import transcribe, load_model
import config  

AUDIO_FOLDER = r"E:\Quran\quran_wav"
QURAN_AYAS_CSV = r"E:\Quran\Quran Ayas List.csv"
MAX_RETRIES = 3  

def get_total_ayas(sura_id):
    import pandas as pd
    df = pd.read_csv(QURAN_AYAS_CSV)
    return len(df[df["sura_id"] == int(sura_id)])

def main():
    # -----------------------------
    # Load model once
    # -----------------------------
    processor, model, device = load_model()

    all_files = sorted([f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")])
    target_files = ["74.wav"]  

    reciter_name = "عمر النبراوي"

    for wav_file in target_files:
        audio_path = os.path.join(AUDIO_FOLDER, wav_file)
        sura_id = os.path.splitext(wav_file)[0]
        print(f"\nProcessing Sura {sura_id} ({wav_file})...")

        # -------------------
        # new uuid for every surah
        recitation_uuid = str(uuid.uuid4())
        config.RECITATION_UUID = recitation_uuid
        print(recitation_uuid)
        config_data = {
        "RECITER_NAME": reciter_name,
        "RECITATION_UUID": recitation_uuid
}

        with open("current_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        total_ayas = get_total_ayas(sura_id)
        attempt = 1
        success = False

        while attempt <= MAX_RETRIES and not success:
            print(f"--- Attempt {attempt} ---")

            # Step 1: Transcribe
            transcribe(audio_path, processor=processor, model=model, device=device)

            # Step 2: Align
            try:
                subprocess.run(
                    ["python", "align_segments.py", "--sura_id", str(sura_id)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"❌ align_segments.py failed for Sura {sura_id}")
                print(f"Return code: {e.returncode}")
                attempt += 1
                time.sleep(1)
                continue

            
            if os.path.exists("aligned.json"):
                with open("aligned.json", encoding="utf-8") as f:
                    aligned = json.load(f)
                if len(aligned) == total_ayas:
                    success = True
                    print(f"✅ All {total_ayas} ayat aligned successfully for Sura {sura_id}")
                else:
                    print(f"⚠️ Only {len(aligned)} of {total_ayas} ayat aligned. Retrying...")
                    attempt += 1
                    time.sleep(1)
            else:
                print("⚠️ aligned.json not found. Retrying...")
                attempt += 1
                time.sleep(1)

        if not success:
            print(f"❌ Failed to fully process Sura {sura_id} after {MAX_RETRIES} attempts. Skipping to next.")

        # Step 4: Save to DB
        try:
            subprocess.run(["python", "save_to_db.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ save_to_db.py failed for Sura {sura_id}")
            print(f"Return code: {e.returncode}")

if __name__ == "__main__":
    main()
