import requests
import os

# Define the folder where downloaded MP3 files will be saved
save_folder = os.path.join("data", "quran_mp3")
os.makedirs(save_folder, exist_ok=True)  # Create the folder if it does not exist

# Base URL template for downloading Quran audio files
# {:03d} ensures that the number is zero-padded to 3 digits (e.g., 018, 076, 114)
base_url = "https://download.quran.islamway.net/quran3/5278/19806/64/{:03d}.mp3"

# Loop through Surah/recitation numbers from 018 to 114 (inclusive)
for num in range(18, 115): 
    url = base_url.format(num)  # Insert the number into the URL
    file_path = os.path.join(save_folder, f"{num:03d}.mp3")  # Save file with zero-padded name
    
    try:
        # Send a GET request to download the audio file
        response = requests.get(url)
        response.raise_for_status()  # Raise error if status code is not 200 (OK)
        
        # Save the downloaded file to disk
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        print(f"Downloaded {num:03d}.mp3")
    except requests.HTTPError:
        # Handle cases where the file does not exist or the server returns an error
        print(f"File {num:03d}.mp3 not found or error")
