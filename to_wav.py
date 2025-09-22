import os
from pydub import AudioSegment


input_folder = "quran_mp3"

output_folder = "quran_wav"
os.makedirs(output_folder, exist_ok=True)


input_extensions = [".mp3", ".m4a", ".ogg", ".flac"]

for filename in os.listdir(input_folder):
    name, ext = os.path.splitext(filename)
    if ext.lower() in input_extensions:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{name}.wav")

        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"✅ Converted {filename} → {name}.wav")
