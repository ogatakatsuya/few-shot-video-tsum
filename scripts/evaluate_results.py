"""
Evaluate summary results from CSV files.

Expected CSV format:
- Columns: video_id, tsum, ground_truth
    - video_id: Video identifier
    - tsum: Generated summary text
    - ground_truth: Reference summary text

Usage:
    uv run -m scripts.evaluate_results <results_directory>
"""

import csv
import sys
from pathlib import Path

from src.metrics.metrics import (
    compute_bleu,
    compute_cider,
    compute_meteor,
    compute_rouge_l,
)


def load_csv_data(csv_path: Path) -> tuple[list[str], list[str]]:
    """
    Load predictions and ground truth from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (predictions, ground_truths)
    """
    predictions = []
    ground_truths = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row["tsum"])
            ground_truths.append(row["ground_truth"])

    return predictions, ground_truths


def evaluate_csv(csv_path: Path) -> dict[str, float]:
    """
    Evaluate a single CSV file with all metrics.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dictionary of metric names to scores
    """
    print(f"\nEvaluating: {csv_path}")

    # Load data
    predictions, ground_truths = load_csv_data(csv_path)
    print(f"  Loaded {len(predictions)} samples")

    # Convert ground truths to list of lists (required format for metrics)
    references = [[gt] for gt in ground_truths]

    # Compute metrics
    scores = {}

    try:
        bleu_score = compute_bleu(predictions, references, max_order=4)
        scores["BLEU-4"] = bleu_score
        print(f"  BLEU-4: {bleu_score:.4f}")
    except Exception as e:
        print(f"  BLEU-4 failed: {e}")
        scores["BLEU-4"] = 0.0

    try:
        meteor_score = compute_meteor(predictions, references)
        scores["METEOR"] = meteor_score
        print(f"  METEOR: {meteor_score:.4f}")
    except Exception as e:
        print(f"  METEOR failed: {e}")
        scores["METEOR"] = 0.0

    try:
        rouge_l_score = compute_rouge_l(predictions, references)
        scores["ROUGE-L"] = rouge_l_score
        print(f"  ROUGE-L: {rouge_l_score:.4f}")
    except Exception as e:
        print(f"  ROUGE-L failed: {e}")
        scores["ROUGE-L"] = 0.0

    try:
        cider_score = compute_cider(predictions, references)
        scores["CIDEr"] = cider_score
        print(f"  CIDEr: {cider_score:.4f}")
    except Exception as e:
        print(f"  CIDEr failed: {e}")
        scores["CIDEr"] = 0.0

    return scores


def main():
    """Main function to evaluate all CSV files in a directory."""
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.evaluate_results <results_directory>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Error: {results_dir} is not a valid directory")
        sys.exit(1)

    # Find all CSV files
    csv_files = list(results_dir.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s)")

    # Evaluate each CSV
    all_results = {}
    for csv_path in sorted(csv_files):
        scores = evaluate_csv(csv_path)
        all_results[csv_path.relative_to(results_dir)] = scores

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for csv_name, scores in all_results.items():
        print(f"\n{csv_name}")
        print("-" * 40)
        print(f"  BLEU-4:  {scores['BLEU-4']:.4f}")
        print(f"  METEOR:  {scores['METEOR']:.4f}")
        print(f"  ROUGE-L: {scores['ROUGE-L']:.4f}")
        print(f"  CIDEr:   {scores['CIDEr']:.4f}")

    # Compute and print average across all CSVs
    if all_results:
        print("\n" + "=" * 80)
        print("AVERAGE ACROSS ALL CSVs")
        print("=" * 80)

        avg_bleu = sum(scores["BLEU-4"] for scores in all_results.values()) / len(
            all_results
        )
        avg_meteor = sum(scores["METEOR"] for scores in all_results.values()) / len(
            all_results
        )
        avg_rouge_l = sum(scores["ROUGE-L"] for scores in all_results.values()) / len(
            all_results
        )
        avg_cider = sum(scores["CIDEr"] for scores in all_results.values()) / len(
            all_results
        )

        print(f"  BLEU-4:  {avg_bleu:.4f}")
        print(f"  METEOR:  {avg_meteor:.4f}")
        print(f"  ROUGE-L: {avg_rouge_l:.4f}")
        print(f"  CIDEr:   {avg_cider:.4f}")


if __name__ == "__main__":
    main()
