"""Unified text evaluation metrics using existing libraries."""

from evaluate import load
from pycocoevalcap.cider.cider import Cider


def compute_bleu(
    predictions: list[str], references: list[list[str]], max_order: int = 4
) -> float:
    """
    Compute BLEU-4 score using Hugging Face evaluate library.

    Args:
        predictions: list of generated texts
        references: list of reference texts (each element is a list of reference strings)
        max_order: Maximum n-gram order (default: 4)

    Returns:
        BLEU score between 0 and 100
    """
    bleu = load("bleu")

    result = bleu.compute(
        predictions=predictions, references=references, max_order=max_order
    )

    if result is None or "bleu" not in result:
        raise ValueError("BLEU computation returned None. Check inputs.")

    return result["bleu"] * 100  # Convert to percentage


def compute_meteor(predictions: list[str], references: list[list[str]]) -> float:
    """
    Compute METEOR score using Hugging Face evaluate library.

    Args:
        predictions: list of generated texts
        references: list of reference texts (each element is a list of reference strings)

    Returns:
        METEOR score between 0 and 1
    """
    meteor = load("meteor")

    result = meteor.compute(predictions=predictions, references=references)

    if result is None or "meteor" not in result:
        raise ValueError("METEOR computation returned None. Check inputs.")

    return result["meteor"]


def compute_rouge_l(predictions: list[str], references: list[list[str]]) -> float:
    """
    Compute ROUGE-L score using Hugging Face evaluate library.

    Args:
        predictions: list of generated texts
        references: list of reference texts (each element is a list of reference strings)

    Returns:
        ROUGE-L F1 score between 0 and 1
    """
    rouge = load("rouge")

    # ROUGE library expects single reference per prediction
    # Use the first reference from each list
    normalized_refs = [ref[0] for ref in references]

    result = rouge.compute(
        predictions=predictions, references=normalized_refs, rouge_types=["rougeL"]
    )

    if result is None or "rougeL" not in result:
        raise ValueError("ROUGE-L computation returned None. Check inputs.")

    return result["rougeL"]


def compute_cider(predictions: list[str], references: list[list[str]]) -> float:
    """
    Compute CIDEr score using pycocoevalcap library.

    Args:
        predictions: list of generated texts
        references: list of reference texts (each element is a list of reference strings)

    Returns:
        CIDEr score
    """
    # Format data for pycocoevalcap
    gts = {}  # ground truth (references)
    res = {}  # results (predictions)

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        res[i] = [pred]
        gts[i] = ref

    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)

    if score is None or not isinstance(score, float):
        raise ValueError("CIDEr computation returned None. Check inputs.")

    return score
