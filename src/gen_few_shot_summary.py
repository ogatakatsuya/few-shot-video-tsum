"""Generate few-shot summary using retrieved examples."""

from datetime import datetime
import json
import random
import csv
from pathlib import Path

from jinja2 import Template
from psycopg2.extensions import connection

from src.retrieval.operations import search_similar_videos
from src.retrieval.db.conn import get_connection
from src.video_encoder.dinov3 import DINOv3VideoEncoder
from src.tsummarizer.base import Prompt
from src.tsummarizer.gemini import GeminiTsumGenerator
from src.env import env

NUM_VIDEOS = 10
NUM_EXAMPLES = 2

PROMPT = """
Summarize this video in 2-5 sentences as a single continuous paragraph.
Don't use bullet points, line breaks, or numbered lists.
Don't use audio information.

[EXAMPLES]
{{examples}}
"""


def retrieve_similar_videos(
    conn: connection,
    encoder: DINOv3VideoEncoder,
    video_path: Path,
    top_k: int = NUM_EXAMPLES,
) -> list[tuple[str, str, float]]:
    """
    Retrieve similar videos using RAG.

    Args:
        conn: Database connection
        encoder: Video encoder
        video_path: Path to query video
        top_k: Number of similar videos to retrieve

    Returns:
        List of (video_id, summary, distance) tuples
    """
    print("  Extracting embedding...")
    query_embedding = encoder.get_embedding(video_path)

    print(f"  Retrieving top-{top_k} similar videos...")
    retrieved = search_similar_videos(conn, query_embedding, limit=top_k)

    for video_id, _, distance in retrieved:
        print(f"    - {video_id} (distance: {distance:.4f})")

    return retrieved


def generate_summary_with_examples(
    summarizer: GeminiTsumGenerator,
    video_path: Path,
    examples: list[tuple[str, str, float]],
) -> str:
    """
    Generate summary using LLM with few-shot examples.

    Args:
        summarizer: Summary generator
        video_path: Path to video
        examples: List of (video_id, summary, distance) tuples

    Returns:
        Generated summary text
    """
    # Few-shotプロンプトの作成
    example_texts = []
    for idx, example in enumerate(examples, 1):
        example_texts.append(f"EXAMPLE {idx}:\n{example[1]}\n")

    # 最終的なプロンプトテキスト
    examples_section = "\n".join(example_texts)
    prompt_text = Template(PROMPT).render(examples=examples_section)
    print(f"  Prompt:\n{prompt_text}")

    # 要約を生成
    print("  Generating summary...")
    tsum = summarizer.generate(
        [
            Prompt(
                video_path=video_path,
                text=prompt_text,
            )
        ]
    )
    print(f"  Generated: {tsum[:100]}...")

    return tsum


def save_results_to_csv(results: list[dict[str, str]], output_csv: Path) -> None:
    """
    Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_csv: Path to output CSV file
    """
    print(f"\nSaving results to {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "tsum", "ground_truth"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Done! Results saved to {output_csv}")


def main():
    json_path = Path("data/test_videoxum.json")
    video_dir = Path("data/videos/test")
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = Path(f"results/few-shot/summaries_output_{now}.csv")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter videos that actually exist
    existing_videos = []
    for item in data:
        video_id = item["video_id"]
        video_path = video_dir / f"{video_id}.mp4"
        if video_path.exists():
            existing_videos.append(item)

    print(f"Found {len(existing_videos)} videos out of {len(data)} in JSON")

    # Randomly select NUM_VIDEOS videos
    if len(existing_videos) < NUM_VIDEOS:
        print(f"Warning: Only {len(existing_videos)} videos available")
        selected_videos = existing_videos
    else:
        selected_videos = random.sample(existing_videos, NUM_VIDEOS)

    print("\nInitializing models...")
    encoder = DINOv3VideoEncoder()
    summarizer = GeminiTsumGenerator(api_key=env.GEMINI_API_KEY)

    # Connect to database
    conn = get_connection()

    # Store results
    results = []

    # Generate summary for each video
    for i, video_data in enumerate(selected_videos, 1):
        video_id = video_data["video_id"]
        video_path = video_dir / f"{video_id}.mp4"

        print(f"\n[{i}/{len(selected_videos)}] Processing {video_id}...")

        # Get ground truth
        ground_truth = " ".join(video_data["tsum"])

        try:
            # Retrieve similar videos
            retrieved = retrieve_similar_videos(
                conn, encoder, video_path, top_k=NUM_EXAMPLES
            )

            # Generate summary
            tsum = generate_summary_with_examples(summarizer, video_path, retrieved)

        except Exception as e:
            print(f"  Error: {e}")
            tsum = f"ERROR: {e}"

        results.append(
            {"video_id": video_id, "tsum": tsum, "ground_truth": ground_truth}
        )

    # Close database connection
    conn.close()

    save_results_to_csv(results, output_csv)


if __name__ == "__main__":
    main()
