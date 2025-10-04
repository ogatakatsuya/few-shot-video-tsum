"""Load embeddings from file and insert to database (for local machine)."""

import pickle
from pathlib import Path

from src.rag.db.conn import get_connection
from src.rag.operations import insert_video_embedding


def load_embeddings_to_db(embeddings_file: Path) -> None:
    """
    Load embeddings from pickle file and insert to database.

    Args:
        embeddings_file: Pickle file containing embeddings data
    """
    # Load embeddings
    print(f"Loading embeddings from {embeddings_file}...")
    with open(embeddings_file, "rb") as f:
        embeddings_data = pickle.load(f)

    print(f"Loaded {len(embeddings_data)} embeddings")

    # Connect to database
    conn = get_connection()

    # Insert each embedding
    for idx, data in enumerate(embeddings_data, 1):
        video_id = data["video_id"]
        print(f"[{idx}/{len(embeddings_data)}] Inserting {video_id}...")

        try:
            insert_video_embedding(
                conn=conn,
                video_id=video_id,
                embedding=data["embedding"],
                summary_ground_truth=data["summary_ground_truth"],
            )
            print(f"  ✓ Inserted {video_id}")

        except Exception as e:
            print(f"  ✗ Error inserting {video_id}: {e}")
            continue

    conn.close()
    print("Done!")


def main():
    """Load embeddings from data/embeddings.pkl to database."""
    embeddings_file = Path(
        "/home/ogata-katsuya/Study/VideoSum/Code/few-shot-tsum/data/embeddings_dinov3.pkl"
    )
    load_embeddings_to_db(embeddings_file)


if __name__ == "__main__":
    main()
