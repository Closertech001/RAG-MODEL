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
jdef init_tables():
    with get_connection() as conn:
        cur = conn.cursor()

        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                department TEXT,
                faculty TEXT,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create chats table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        conn.commit()

# ✅ Get user by name
def get_user(name):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE name = ?", (name,))
        return cur.fetchone()

# ✅ Create new user
def create_user(name, faculty, department):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (name, faculty, department)
            VALUES (?, ?, ?)
        """, (name, faculty, department))
        conn.commit()
        return cur.lastrowid

# ✅ Save chat history
def save_chat(user_id, message, response):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chats (user_id, message, response)
            VALUES (?, ?, ?)
        """, (user_id, message, response))
        conn.commit()
