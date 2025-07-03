# db.py
import psycopg2
from psycopg2.extras import RealDictCursor
import os

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/crescent_db")

def get_connection():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def init_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name TEXT,
            department TEXT,
            faculty TEXT,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id SERIAL PRIMARY KEY,
            user_id INT REFERENCES users(id),
            message TEXT,
            response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def get_user(name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name=%s", (name,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user

def create_user(name, faculty=None, department=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, faculty, department) VALUES (%s, %s, %s) RETURNING id",
        (name, faculty, department)
    )
    user_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return user_id

def save_chat(user_id, message, response):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO chats (user_id, message, response) VALUES (%s, %s, %s)",
                (user_id, message, response))
    conn.commit()
    cur.close()
    conn.close()
