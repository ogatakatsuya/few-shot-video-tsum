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
PROMPT = """
Summarize this video in 2-5 sentences as a single continuous paragraph.
Don't use bullet points, line breaks, or numbered lists.
Don't use audio information.

[EXAMPLES]
{{examples}}
"""


def main():
    # パスの設定
    json_path = Path("data/val_videoxum.json")
    video_dir = Path("data/videos")
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_csv = Path(f"results/random-few-shot/summaries_output_{now}.csv")

    # 出力ディレクトリの作成
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # JSONデータを読み込む
    print("Loading JSON data...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # 実際に存在する動画のみをフィルタリング
    print("Checking which videos exist...")
    existing_videos = []
    for item in data:
        video_id = item["video_id"]
        video_path = video_dir / f"{video_id}.mp4"
        if video_path.exists():
            existing_videos.append(item)

    print(f"Found {len(existing_videos)} videos out of {len(data)} in JSON")

    # ランダムにNUM_VIDEOS個選択（テスト用）
    if len(existing_videos) < NUM_VIDEOS:
        print(f"Warning: Only {len(existing_videos)} videos available")
        selected_videos = existing_videos
    else:
        selected_videos = random.sample(existing_videos, NUM_VIDEOS)

    print(f"Selected {len(selected_videos)} videos for summarization")

    # 要約生成器の初期化
    summarizer = GeminiTsumGenerator(api_key=env.GEMINI_API_KEY)

    # 結果を保存するリスト
    results = []

    # 各動画について要約を生成
    for i, video_data in enumerate(selected_videos, 1):
        video_id = video_data["video_id"]
        video_path = video_dir / f"{video_id}.mp4"

        print(f"\n[{i}/{len(selected_videos)}] Processing {video_id}...")

        # Ground truthを取得（tsumリストを結合）
        ground_truth = " ".join(video_data["tsum"])

        # Few-shot用の例を選択（現在処理中の動画以外からランダムに2つ選択）
        available_examples = [v for v in existing_videos if v["video_id"] != video_id]
        few_shot_examples = random.sample(
            available_examples, min(2, len(available_examples))
        )

        # Few-shotプロンプトの作成
        example_texts = []
        for idx, example in enumerate(few_shot_examples, 1):
            example_video_path = video_dir / f"{example['video_id']}.mp4"
            example_summary = " ".join(example["tsum"])

            # 例が存在する動画のみを追加
            if example_video_path.exists():
                example_texts.append(
                    f"EXAMPLE {idx}:\n"
                    f"Video: {example['video_id']}\n"
                    f"Summary: {example_summary}\n"
                )

        # 最終的なプロンプトテキスト
        examples_section = "\n".join(example_texts)
        prompt_text = Template(PROMPT).render(examples=examples_section)

        # 要約を生成
        try:
            tsum = summarizer.generate(
                [
                    Prompt(
                        video_path=video_path,
                        text=prompt_text,
                    )
                ]
            )
            print(f"Generated summary: {tsum[:100]}...")
        except Exception as e:
            print(f"Error generating summary: {e}")
            tsum = f"ERROR: {e}"

        # 結果を追加
        results.append(
            {"video_id": video_id, "tsum": tsum, "ground_truth": ground_truth}
        )

    # CSVファイルに保存
    print(f"\nSaving results to {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "tsum", "ground_truth"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Results saved to {output_csv}")


if __name__ == "__main__":
    main()
