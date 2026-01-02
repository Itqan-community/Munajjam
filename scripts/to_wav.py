import os
from pydub import AudioSegment

# Input folder containing recitations in different formats (e.g., .mp3, .m4a, .ogg, .flac)
input_folder = os.path.join("data", "quran_mp3")

# Output folder where converted .wav files will be saved
output_folder = os.path.join("data", "quran_wav")
os.makedirs(output_folder, exist_ok=True)

# Supported input file extensions
input_extensions = [".mp3", ".m4a", ".ogg", ".flac"]

# -----------------------------
# Conversion Loop
# -----------------------------
for filename in os.listdir(input_folder):
    name, ext = os.path.splitext(filename)

    # Only process supported audio formats
    if ext.lower() in input_extensions:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{name}.wav")

        # Load audio file and export as .wav
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")

        print(f"Converted {filename} → {name}.wav")
