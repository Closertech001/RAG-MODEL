import sqlite3
from sqlite3 import Connection, Row
import os

# === DATABASE CONFIGURATION ===
DB_FILENAME = "crescent_db.sqlite3"

# Use a fixed path that works both locally and on Streamlit Cloud
DB_PATH = os.path.join(os.getcwd(), DB_FILENAME)


# === CONNECT TO DB ===
def get_connection() -> Connection:
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = Row  # Allows row["column_name"] access
    return conn


# === INIT TABLES ===
def init_tables():
    """Create necessary tables if they don't already exist."""
    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    department TEXT,
                    faculty TEXT,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Chats table
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
    except Exception as e:
        import streamlit as st
        st.error(f"❌ Database initialization failed: {e}")
        raise


# === GET USER BY NAME ===
def get_user(name: str):
    """Fetch user info by name."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE name = ?", (name,))
        return cur.fetchone()


# === CREATE NEW USER ===
def create_user(name: str, faculty: str, department: str):
    """Insert a new user and return the new user ID."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (name, faculty, department)
            VALUES (?, ?, ?)
        """, (name, faculty, department))
        conn.commit()
        return cur.lastrowid


# === SAVE CHAT HISTORY ===
def save_chat(user_id: int, message: str, response: str):
    """Save a chat exchange into the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chats (user_id, message, response)
            VALUES (?, ?, ?)
        """, (user_id, message, response))
        conn.commit()
