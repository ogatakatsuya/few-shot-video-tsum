from datetime import datetime
import json
import random
import csv
from pathlib import Path

from src.tsummarizer.base import Prompt
from src.tsummarizer.gemini import GeminiTsumGenerator
from src.env import env

NUM_VIDEOS = 10
PROMPT = """
Summarize this video in 2-5 sentences as a single continuous paragraph.
Don't use bullet points, line breaks, or numbered lists.
Don't use audio information.
"""


def generate_zero_shot_summary(
    summarizer: GeminiTsumGenerator,
    video_path: Path,
) -> str:
    """
    Generate summary using LLM without examples.

    Args:
        summarizer: Summary generator
        video_path: Path to video

    Returns:
        Generated summary text
    """
    print("  Generating summary...")
    tsum = summarizer.generate(
        [
            Prompt(
                video_path=video_path,
                text=PROMPT,
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
    # Setup paths
    json_path = Path("data/val_videoxum.json")
    video_dir = Path("data/videos/validation")
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = Path(f"results/zero-shot/summaries_output_{now}.csv")

    # Create output directory
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load JSON data
    print("Loading JSON data...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter videos that actually exist
    print("Checking which videos exist...")
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

    print(f"Selected {len(selected_videos)} videos for summarization")

    summarizer = GeminiTsumGenerator(api_key=env.GEMINI_API_KEY)

    # Store results
    results = []

    # Generate summary for each video
    for i, video_data in enumerate(selected_videos, 1):
        video_id = video_data["video_id"]
        video_path = video_dir / f"{video_id}.mp4"

        print(f"\n[{i}/{len(selected_videos)}] Processing {video_id}...")

        # Get ground truth (join tsum list)
        ground_truth = " ".join(video_data["tsum"])

        # Generate summary
        try:
            tsum = generate_zero_shot_summary(summarizer, video_path)
        except Exception as e:
            print(f"  Error: {e}")
            tsum = f"ERROR: {e}"

        # Append result
        results.append(
            {"video_id": video_id, "tsum": tsum, "ground_truth": ground_truth}
        )

    # Save to CSV
    save_results_to_csv(results, output_csv)


if __name__ == "__main__":
    main()
