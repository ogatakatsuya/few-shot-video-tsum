import psycopg2
from psycopg2.extensions import connection


def get_connection() -> connection:
    """Get PostgreSQL database connection."""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="videosum",
        user="postgres",
        password="postgres",
    )
