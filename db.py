# --- db.py ---
import psycopg2
import os

DB_NAME = os.getenv("DB_NAME", "crescent_chatbot")
DB_USER = os.getenv("DB_USER", "chatbot_user")
DB_PASS = os.getenv("DB_PASS", "yourpassword")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )

def init_db():
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        name TEXT,
        department TEXT,
        faculty TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS chat_history (
        id SERIAL PRIMARY KEY,
        user_id INT REFERENCES users(id),
        role TEXT,  -- 'user' or 'assistant'
        message TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()

def add_user(name, department, faculty):
    conn = connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (name, department, faculty) VALUES (%s, %s, %s) RETURNING id",
                (name, department, faculty))
    user_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return user_id

def get_user(name):
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name = %s", (name,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None

def save_chat(user_id, role, message):
    conn = connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO chat_history (user_id, role, message) VALUES (%s, %s, %s)",
                (user_id, role, message))
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=10):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT role, message FROM chat_history 
        WHERE user_id = %s ORDER BY timestamp DESC LIMIT %s
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"role": r, "content": m} for r, m in reversed(rows)]
