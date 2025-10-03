from pathlib import Path

from src.rag.db.conn import get_connection


def init_db():
    """Initialize database by executing DDL."""
    conn = get_connection()
    cur = conn.cursor()

    # Read and execute DDL
    ddl_path = Path(__file__).parent.parent / "src" / "rag" / "db" / "ddl.sql"
    with open(ddl_path) as f:
        ddl = f.read()

    cur.execute(ddl)
    conn.commit()

    print("Database initialized successfully!")

    cur.close()
    conn.close()


if __name__ == "__main__":
    init_db()
