import json
from pathlib import Path

from src.video_encoder.dinov3 import DINOv3VideoEncoder
from src.rag.db.conn import get_connection
from src.rag.operations import insert_video_embedding


def build_rag(
    video_dir: Path,
    annotation_file: Path,
) -> None:
    """
    Build RAG database from videos and annotations.

    Args:
        video_dir: Directory containing video files
        annotation_file: JSON file with video annotations
    """
    # Load annotations
    with open(annotation_file) as f:
        annotations = json.load(f)

    # Create video_id to summary mapping
    video_summaries = {item["video_id"]: " ".join(item["tsum"]) for item in annotations}

    # Initialize encoder
    print("Initializing DINOv3 encoder...")
    encoder = DINOv3VideoEncoder()

    # Connect to database
    conn = get_connection()

    # Process each video
    video_files = sorted(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos")

    for idx, video_path in enumerate(video_files, 1):
        video_id = video_path.stem  # e.g., "v_ehGHCYKzyZ8"

        # Skip if no annotation
        if video_id not in video_summaries:
            print(f"[{idx}/{len(video_files)}] Skipping {video_id} (no annotation)")
            continue

        print(f"[{idx}/{len(video_files)}] Processing {video_id}...")

        try:
            # Extract embedding
            embedding = encoder.get_embedding(video_path)

            # Insert into database
            insert_video_embedding(
                conn=conn,
                video_id=video_id,
                embedding=embedding,
                summary_ground_truth=video_summaries[video_id],
            )

            print(f"  ✓ Inserted {video_id} (embedding shape: {embedding.shape})")

        except Exception as e:
            print(f"  ✗ Error processing {video_id}: {e}")
            continue

    conn.close()
    print("Done!")


def main():
    """Build RAG from videos in data/videos directory."""
    video_dir = Path("data/videos")
    annotation_file = Path("data/val_videoxum.json")

    build_rag(video_dir, annotation_file)


if __name__ == "__main__":
    main()
