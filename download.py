import requests
import os

save_folder = "quran_mp3/"
os.makedirs(save_folder, exist_ok=True)


base_url = "https://download.quran.islamway.net/quran3/5278/19806/64/{:03d}.mp3"

for num in range(18, 115): 
    url = base_url.format(num)
    file_path = os.path.join(save_folder, f"{num:03d}.mp3")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        
        print(f"Downloaded {num:03d}.mp3")
    except requests.HTTPError:
        print(f"File {num:03d}.mp3 not found or error")
