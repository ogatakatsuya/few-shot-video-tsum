"""Download videos from VideoXum dataset using video IDs."""

import json
from pathlib import Path
import yt_dlp
from tqdm import tqdm


def download_video(video_id: str, output_dir: Path) -> bool:
    """
    Download a video using yt-dlp.

    Args:
        video_id: Video ID (e.g., 'v_ehGHCYKzyZ8')
        output_dir: Directory to save the video

    Returns:
        True if successful, False otherwise
    """
    # Extract the actual ID (remove 'v_' prefix if present)
    if video_id.startswith("v_"):
        actual_id = video_id[2:]
    else:
        actual_id = video_id

    # YouTube URL
    url = f"https://www.youtube.com/watch?v={actual_id}"

    # Output template: video_id.ext
    output_template = str(output_dir / f"{video_id}.%(ext)s")

    # Check if video already exists (any extension)
    existing_files = list(output_dir.glob(f"{video_id}.*"))
    if existing_files:
        print(f"Skipping {video_id}: already downloaded")
        return True

    ydl_opts = {
        "format": "best[ext=mp4]/best",  # Prefer mp4
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,  # Skip unavailable videos
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([url])
            if error_code != 0:
                return False
        return True
    except Exception:
        # Silently skip errors (video unavailable, deleted, etc.)
        return False


def main():
    """Main function to download all videos from the dataset."""
    # Load dataset
    data_file = Path("data/val_videoxum.json")
    output_dir = Path("data/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_file}...")
    with open(data_file) as f:
        data = json.load(f)

    print(f"Found {len(data)} videos to download")

    # Download videos
    success_count = 0
    fail_count = 0

    for item in tqdm(data, desc="Downloading videos"):
        video_id = item["video_id"]
        if download_video(video_id, output_dir):
            success_count += 1
        else:
            fail_count += 1

    print("\nDownload complete!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
