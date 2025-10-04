from datetime import datetime
import json
import random
import csv
from pathlib import Path

from jinja2 import Template

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


def select_random_examples(
    all_videos: list[dict],
    current_video_id: str,
    video_dir: Path,
    num_examples: int = NUM_EXAMPLES,
) -> list[dict]:
    """
    Randomly select few-shot examples.

    Args:
        all_videos: All available videos
        current_video_id: Current video ID to exclude
        video_dir: Directory containing videos
        num_examples: Number of examples to select

    Returns:
        List of example video data
    """
    # Exclude current video
    available_examples = [v for v in all_videos if v["video_id"] != current_video_id]
    few_shot_examples = random.sample(
        available_examples, min(num_examples, len(available_examples))
    )

    # Filter only existing videos
    valid_examples = []
    for example in few_shot_examples:
        example_video_path = video_dir / f"{example['video_id']}.mp4"
        if example_video_path.exists():
            valid_examples.append(example)

    return valid_examples


def generate_summary_with_random_examples(
    summarizer: GeminiTsumGenerator,
    video_path: Path,
    examples: list[dict],
) -> str:
    """
    Generate summary using LLM with randomly selected few-shot examples.

    Args:
        summarizer: Summary generator
        video_path: Path to video
        examples: List of example video data

    Returns:
        Generated summary text
    """
    # Create few-shot prompt
    example_texts = []
    for idx, example in enumerate(examples, 1):
        example_summary = " ".join(example["tsum"])
        example_texts.append(f"EXAMPLE {idx}:\nSummary: {example_summary}\n")

    # Final prompt text
    examples_section = "\n".join(example_texts)
    prompt_text = Template(PROMPT).render(examples=examples_section)

    # Generate summary
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
    # Setup paths
    json_path = Path("data/val_videoxum.json")
    video_dir = Path("data/videos/validation")
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = Path(f"results/random-few-shot/summaries_output_{now}.csv")

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

    # Initialize summarizer
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

        try:
            # Select random examples
            examples = select_random_examples(
                existing_videos, video_id, video_dir, num_examples=NUM_EXAMPLES
            )
            print(f"  Selected {len(examples)} random examples")

            # Generate summary
            tsum = generate_summary_with_random_examples(
                summarizer, video_path, examples
            )

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
