import sqlite3
import json
from logging_utils import init_logging_file

# -----------------------
# Load config from JSON
# -----------------------
with open("current_config.json", encoding="utf-8") as f:
    config_data = json.load(f)

RECITER_NAME = config_data["RECITER_NAME"]
RECITATION_UUID = config_data["RECITATION_UUID"]

# -----------------------
# Load aligned results
# -----------------------
with open("aligned.json", "r", encoding="utf-8") as f:
    aligned = json.load(f)

if not aligned:
    raise ValueError("aligned.json is empty!")

current_sura_id = aligned[0]["sura_id"]

# -----------------------
# DB setup
# -----------------------
conn = sqlite3.connect("quran.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS ayah_timestamps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recitation_uuid TEXT,
    surah_num INTEGER,
    ayah_num INTEGER,
    start_time REAL,
    end_time REAL,
    reciter_name TEXT
)
""")

# Insert data
for rec in aligned:
    cursor.execute("""
        INSERT INTO ayah_timestamps 
        (recitation_uuid, surah_num, ayah_num, start_time, end_time, reciter_name)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        RECITATION_UUID,
        rec["sura_id"],
        rec["index"],
        rec["start"],
        rec["end"],
        RECITER_NAME
    ))

conn.commit()
conn.close()

# -----------------------
# Create logging
# -----------------------
log_file = init_logging_file(current_sura_id)
print(f"✅ Data saved and logging initialized at: {log_file}")
