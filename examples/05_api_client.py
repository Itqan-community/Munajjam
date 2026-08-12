"""
API Client Example for Munajjam

This example demonstrates how to interact with the Munajjam API Server
running via Docker or uvicorn. It uploads an audio file, starts an alignment
job, and polls for the result.
"""

import time
import requests
from pathlib import Path

API_URL = "http://localhost:8000"


def main():
    surah_number = 114
    # Placeholder for the audio path, update to your actual path
    audio_path = Path("Quran/badr_alturki_audio/114.wav")

    if not audio_path.exists():
        print(
            f"Please place an audio file at {audio_path} or update the path in this script."
        )
        return

    print(f"Submitting Surah {surah_number} to Munajjam API...")

    # Step 1: Submit the job
    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{API_URL}/align/{surah_number}",
                files={"file": f},
                data={"riwaya": "hafs"},
            )
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data["job_id"]
        print(f"Job started successfully! Job ID: {job_id}")
    except requests.exceptions.ConnectionError:
        print(
            f"Error: Could not connect to API at {API_URL}. Make sure the server or Docker container is running."
        )
        return
    except Exception as e:
        print(f"Error submitting job: {e}")
        return

    # Step 2: Poll for completion
    print("Polling for results (this may take a few seconds)...")
    while True:
        try:
            status_response = requests.get(f"{API_URL}/align/status/{job_id}")
            status_response.raise_for_status()
            status_data = status_response.json()

            status = status_data.get("status")
            if status == "success":
                print("\nJob Completed Successfully!")
                results = status_data["data"]

                print("\nAligned Ayahs:")
                print("=" * 40)
                for ayah in results:
                    print(
                        f"Ayah {ayah['ayah_number']}: {ayah['start_time']:.2f}s - {ayah['end_time']:.2f}s"
                    )
                break
            elif status == "error":
                print(f"\nJob Failed: {status_data.get('message')}")
                break
            else:
                print(f"Status: {status} - {status_data.get('message', 'Waiting...')}")
                time.sleep(2)  # Wait before checking again

        except Exception as e:
            print(f"\nError checking status: {e}")
            break


if __name__ == "__main__":
    main()
