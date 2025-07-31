# db.py
import sqlite3
from pathlib import Path

# This will live next to your code
DB_PATH = Path(__file__).parent / "app_data.db"

def get_conn():
    """Return a sqlite3 connection with row access by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the budgets table if it doesn't exist, and seed defaults."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                area TEXT PRIMARY KEY,
                allocated INTEGER NOT NULL
            )
        """)
        # If table is empty, insert your four default rows:
        cur = conn.execute("SELECT COUNT(*) as cnt FROM budgets")
        if cur.fetchone()["cnt"] == 0:
            defaults = [
                ("Sky Retail",   70000),
                ("Sky Business", 70000),
                ("Sky VIP",      70000),
                ("Tier 2",       70000),
            ]
            conn.executemany(
                "INSERT INTO budgets(area, allocated) VALUES(?, ?)",
                defaults
            )
