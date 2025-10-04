import numpy as np
from psycopg2.extensions import connection


def insert_video_embedding(
    conn: connection,
    video_id: str,
    embedding: np.ndarray,
    summary_ground_truth: str,
) -> None:
    """
    Insert video embedding and metadata into database.

    Args:
        conn: PostgreSQL connection
        video_id: Unique video identifier
        embedding: Video embedding vector (768,)
        summary_ground_truth: Ground truth summary text
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_embeddings (video_id, embedding, summary_ground_truth)
            VALUES (%s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                summary_ground_truth = EXCLUDED.summary_ground_truth
            """,
            (video_id, embedding.tolist(), summary_ground_truth),
        )
    conn.commit()


def search_similar_videos(
    conn: connection, query_embedding: np.ndarray, limit: int = 5
) -> list[tuple[str, str, float]]:
    """
    Search for similar videos using cosine similarity.

    Args:
        conn: PostgreSQL connection
        query_embedding: Query embedding vector (768,)
        limit: Number of results to return

    Returns:
        List of tuples: (video_id, summary_ground_truth, distance)
    """
    with conn.cursor() as cur:
        # Convert numpy array to string format for vector casting
        embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
        cur.execute(
            """
            SELECT video_id, summary_ground_truth, embedding <=> %s::vector AS distance
            FROM video_embeddings
            ORDER BY distance
            LIMIT %s
            """,
            (embedding_str, limit),
        )
        return cur.fetchall()
