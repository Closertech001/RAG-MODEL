# db.py

import sqlite3
from sqlite3 import Connection, Row
import os

# Optional: use env var or fallback to default file name
DB_PATH = os.getenv("SQLITE_DB_PATH", "crescent_db.sqlite3")

# ✅ Function that returns a connection object
def get_connection() -> Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = Row  # Enables row access like dictionaries
    return conn

# ✅ Function to initialize tables
def init_tables():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                department TEXT,
                faculty TEXT,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)
        conn.commit()
