import psycopg2
from psycopg2.extras import RealDictCursor
import os

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/crescent_db")

def get_connection():
    return psycopg2.connect(DB_URL)

def init_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
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

def get_user(name):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE name=%s", (name,))
            return cur.fetchone()

def create_user(name, faculty=None, department=None):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO users (name, faculty, department) VALUES (%s, %s, %s) RETURNING id",
                (name, faculty, department)
            )
            user_id = cur.fetchone()["id"]
        conn.commit()
    return user_id

def save_chat(user_id, message, response):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (user_id, message, response) VALUES (%s, %s, %s)",
                (user_id, message, response)
            )
        conn.commit()
