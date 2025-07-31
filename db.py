# db.py
import sqlite3
from pathlib import Path

# SQLite file next to your code
DB_PATH = Path(__file__).parent / "app_data.db"

def get_conn():
    """Return a sqlite3 connection with row access by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create budgets & expenses tables if they don't exist, and seed defaults."""
    with get_conn() as conn:
        # ─── Budgets ───────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                area TEXT PRIMARY KEY,
                allocated INTEGER NOT NULL
            )
        """)
        # Seed budgets if empty
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

        # ─── Expenses ──────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT,
                date        TEXT,
                area        TEXT,
                description TEXT,
                amount      REAL
            )
        """)
        # no seeding for expenses—starts empty

def add_expense(rec: dict):
    """
    Insert one expense record into the expenses table.
    rec must have keys: Name, Date, Area, Description, Amount
    """
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO expenses(name, date, area, description, amount) VALUES(?,?,?,?,?)",
            (rec["Name"], rec["Date"], rec["Area"], rec["Description"], rec["Amount"])
        )
