import sqlite3
import os


def create_connection(db_file=os.path.join("data", "quran.db")):
    """
    Create and return a connection to the SQLite database.

    Args:
        db_file (str): Path to the SQLite database file. Defaults to "data/quran.db".

    Returns:
        sqlite3.Connection: SQLite database connection object.
    """
    return sqlite3.connect(db_file)


def create_table(conn):
    """
    Create the `ayah_timestamps` table if it does not already exist.

    The table schema:
        - id (INTEGER PRIMARY KEY): Auto-increment ID.
        - recitation_uuid (TEXT): Unique identifier for the recitation session.
        - surah_num (INTEGER): Surah number (1–114).
        - ayah_num (INTEGER): Ayah number within the Surah.
        - start_time (REAL): Start timestamp of the ayah in the audio.
        - end_time (REAL): End timestamp of the ayah in the audio.
        - reciter_name (TEXT): Name of the reciter.

    Args:
        conn (sqlite3.Connection): Active database connection.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ayah_timestamps (
            id INTEGER PRIMARY KEY,
            recitation_uuid TEXT NOT NULL,
            surah_num INTEGER NOT NULL,
            ayah_num INTEGER NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            reciter_name TEXT
        )
    """)
    conn.commit()


def insert_ayah_timestamp(conn, ayah_data):
    """
    Insert a single ayah timestamp record into the database.

    Args:
        conn (sqlite3.Connection): Active database connection.
        ayah_data (tuple): Record containing:
            (recitation_uuid, surah_num, ayah_num, start_time, end_time, reciter_name)

    Example:
        ayah_data = (
            "123e4567-e89b-12d3-a456-426614174000",  # recitation_uuid
            1,                                      # surah_num
            7,                                      # ayah_num
            0.0,                                    # start_time
            5.32,                                   # end_time
            "Omar Al-Nabrawy"                       # reciter_name
        )
    """
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ayah_timestamps
        (recitation_uuid, surah_num, ayah_num, start_time, end_time, reciter_name)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ayah_data)
    conn.commit()
