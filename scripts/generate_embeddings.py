"""Generate video embeddings and save to file (for GPU server)."""

import json
import pickle
from pathlib import Path

from src.video_encoder.dinov3 import DINOv3VideoEncoder


def generate_embeddings(
    video_dir: Path,
    annotation_file: Path,
    output_file: Path,
) -> None:
    """
    Generate embeddings and save to pickle file.

    Args:
        video_dir: Directory containing video files
        annotation_file: JSON file with video annotations
        output_file: Output pickle file path
    """
    # Load annotations
    with open(annotation_file) as f:
        annotations = json.load(f)

    # Create video_id to summary mapping
    video_summaries = {item["video_id"]: " ".join(item["tsum"]) for item in annotations}

    # Initialize encoder
    print("Initializing DINOv3 encoder...")
    encoder = DINOv3VideoEncoder()

    # Process each video
    video_files = sorted(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos")

    embeddings_data = []

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

            # Store data
            embeddings_data.append(
                {
                    "video_id": video_id,
                    "embedding": embedding,
                    "summary_ground_truth": video_summaries[video_id],
                }
            )

            print(f"  ✓ Generated {video_id} (embedding shape: {embedding.shape})")

        except Exception as e:
            print(f"  ✗ Error processing {video_id}: {e}")
            continue

    # Save to pickle file
    print(f"\nSaving {len(embeddings_data)} embeddings to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(embeddings_data, f)

    print("Done!")


def generate_embeddings_batch(
    video_dir: Path,
    annotation_file: Path,
    output_file: Path,
    batch_size: int = 8,
) -> None:
    """
    Generate embeddings using batch processing for better GPU utilization.

    Args:
        video_dir: Directory containing video files
        annotation_file: JSON file with video annotations
        output_file: Output pickle file path
        batch_size: Number of videos to process in each batch
    """
    # Load annotations
    with open(annotation_file) as f:
        annotations = json.load(f)

    # Create video_id to summary mapping
    video_summaries = {item["video_id"]: " ".join(item["tsum"]) for item in annotations}

    # Initialize encoder
    print("Initializing DINOv3 encoder...")
    encoder = DINOv3VideoEncoder()

    # Get valid video files (those with annotations)
    video_files = sorted(video_dir.glob("*.mp4"))
    valid_videos = []

    for video_path in video_files:
        video_id = video_path.stem
        if video_id in video_summaries:
            valid_videos.append(video_path)
        else:
            print(f"Skipping {video_id} (no annotation)")

    print(f"Found {len(valid_videos)} valid videos (out of {len(video_files)} total)")

    embeddings_data = []
    total_processed = 0

    # Process videos in batches
    for i in range(0, len(valid_videos), batch_size):
        batch_videos = valid_videos[i : i + batch_size]
        batch_start = i + 1
        batch_end = min(i + batch_size, len(valid_videos))

        print(f"\nProcessing batch {batch_start}-{batch_end}/{len(valid_videos)}...")

        try:
            # Extract embeddings for the entire batch
            batch_embeddings = encoder.get_embeddings_batch(batch_videos)

            # Store results
            for video_path in batch_videos:
                video_id = video_path.stem
                if video_id in batch_embeddings:
                    embedding = batch_embeddings[video_id]
                    embeddings_data.append(
                        {
                            "video_id": video_id,
                            "embedding": embedding,
                            "summary_ground_truth": video_summaries[video_id],
                        }
                    )
                    total_processed += 1
                else:
                    print(f"  ✗ No embedding generated for {video_id}")

        except Exception as e:
            print(f"  ✗ Error processing batch {batch_start}-{batch_end}: {e}")
            # Fallback to individual processing for this batch
            print("  Falling back to individual processing...")
            for video_path in batch_videos:
                video_id = video_path.stem
                try:
                    embedding = encoder.get_embedding(video_path)
                    embeddings_data.append(
                        {
                            "video_id": video_id,
                            "embedding": embedding,
                            "summary_ground_truth": video_summaries[video_id],
                        }
                    )
                    total_processed += 1
                    print(f"    ✓ Generated {video_id} (fallback)")
                except Exception as fallback_e:
                    print(f"    ✗ Error processing {video_id}: {fallback_e}")

    # Save to pickle file
    print(f"\nSaving {len(embeddings_data)} embeddings to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(embeddings_data, f)

    print(f"Done! Successfully processed {total_processed}/{len(valid_videos)} videos.")


def main():
    """Generate embeddings from videos in data/videos directory."""
    video_dir = Path("data/videos")
    annotation_file = Path("data/val_videoxum.json")
    output_file = Path("data/embeddings.pkl")

    generate_embeddings(video_dir, annotation_file, output_file)


def main_batch():
    """Generate embeddings using batch processing for better GPU utilization."""
    video_dir = Path("data/videos")
    annotation_file = Path("data/val_videoxum.json")
    output_file = Path("data/embeddings_batch.pkl")

    # Increased batch size since we're now using frame sampling
    # With 32 frames per video and batch_size=16, we process ~512 frames at once
    generate_embeddings_batch(video_dir, annotation_file, output_file, batch_size=16)


if __name__ == "__main__":
    # Use batch processing by default for better GPU utilization
    main_batch()
    # Uncomment below to use original single-video processing
    # main()
