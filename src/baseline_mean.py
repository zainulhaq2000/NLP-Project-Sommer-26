#!/usr/bin/env python3
"""
baseline_mean.py

Mean-score baseline for the NLP Summer 2026 AmbiStory project.

The model ignores the input text and always predicts the same score:
the mean of the `average` values in the training set.

This is useful as the first baseline because it tells us how far we can get
without using any information from the story, homonym, judged meaning, or ending.

This file contains the model logic. The root-level predict.py file should import
this file and use it as the official entry point.

Example development usage from the project root:

    python src/baseline_mean.py \
        --train data/train.json \
        --input data/dev.json \
        --output results/dev_predictions_mean.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# Computed from the provided train.json:
# 2280 training samples, mean of the `average` field = 3.140029239766082.
# This fallback keeps predict.py runnable even if data/train.json is not present.
DEFAULT_TRAINING_MEAN = 3.140029239766082


def load_json_dict(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a project JSON file.

    The expected format is a single JSON object:
        {
            "0": {...},
            "1": {...},
            ...
        }

    The top-level keys must be preserved because the official prediction output
    must use these keys as prediction IDs.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected {path} to contain a JSON object/dictionary, "
            f"but got {type(data).__name__}."
        )

    for sample_id, sample in data.items():
        if not isinstance(sample, dict):
            raise ValueError(
                f"Sample {sample_id!r} in {path} is not a JSON object."
            )

    return data


def sorted_sample_ids(data: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Sort sample IDs numerically when possible, otherwise lexicographically.

    The official format allows predictions in any order, but deterministic order
    makes files easier to inspect and compare.
    """
    def sort_key(sample_id: str) -> Tuple[int, object]:
        sample_id = str(sample_id)
        return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)

    return sorted((str(sample_id) for sample_id in data.keys()), key=sort_key)


def extract_average_scores(data: Dict[str, Dict[str, Any]]) -> List[float]:
    """
    Extract the gold average plausibility scores from a training JSON object.
    """
    scores: List[float] = []

    for sample_id in sorted_sample_ids(data):
        sample = data[sample_id]

        if "average" not in sample:
            raise ValueError(
                f"Training sample {sample_id!r} is missing the field 'average'."
            )

        try:
            scores.append(float(sample["average"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Training sample {sample_id!r} has a non-numeric average: "
                f"{sample['average']!r}."
            ) from exc

    if not scores:
        raise ValueError("Cannot compute a mean baseline from an empty dataset.")

    return scores


def compute_mean_score(training_data: Dict[str, Dict[str, Any]]) -> float:
    """
    Compute the mean of the human average ratings in the training data.
    """
    scores = extract_average_scores(training_data)
    return sum(scores) / len(scores)


def clip_score(score: float, lower: int = 1, upper: int = 5) -> float:
    """
    Clip a score to the valid project range [1, 5].
    """
    return max(lower, min(upper, float(score)))


def round_to_valid_integer(score: float) -> int:
    """
    Convert a continuous score to an integer prediction from 1 to 5.

    We use conventional rounding via floor(score + 0.5), then clip to [1, 5].
    This avoids Python's banker's rounding for values exactly ending in .5.
    """
    rounded = math.floor(float(score) + 0.5)
    return int(clip_score(rounded, lower=1, upper=5))


class MeanBaselineModel:
    """
    Mean-score baseline model.

    After fitting, the model always predicts the rounded training-set mean.
    It deliberately ignores the input sample content.
    """

    def __init__(self, mean_score: float | None = None) -> None:
        self.mean_score = mean_score

    def fit(self, training_data: Dict[str, Dict[str, Any]]) -> "MeanBaselineModel":
        """
        Estimate the constant prediction from training data.
        """
        self.mean_score = compute_mean_score(training_data)
        return self

    def predict_one(self, sample: Dict[str, Any] | None = None) -> int:
        """
        Predict one integer plausibility score.

        The sample argument is accepted for a consistent model interface, but it
        is not used by the mean baseline.
        """
        if self.mean_score is None:
            raise ValueError("MeanBaselineModel must be fitted before prediction.")

        return round_to_valid_integer(self.mean_score)

    def predict_many(self, data: Dict[str, Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Predict one integer score for every sample in a JSON object.

        Returns:
            List of (top_level_sample_id, prediction) pairs.
        """
        prediction = self.predict_one()
        return [(sample_id, prediction) for sample_id in sorted_sample_ids(data)]


def load_or_default_model(train_path: Path | None = None) -> MeanBaselineModel:
    """
    Build a mean-baseline model.

    If train_path exists, the mean is computed from that file.
    Otherwise, we use the fallback mean computed from the provided train.json.
    """
    if train_path is not None and train_path.exists():
        training_data = load_json_dict(train_path)
        model = MeanBaselineModel()
        model.fit(training_data)
        return model

    return MeanBaselineModel(mean_score=DEFAULT_TRAINING_MEAN)


def write_predictions_jsonl(predictions: Iterable[Tuple[str, int]], output_path: Path) -> None:
    """
    Write predictions in the official JSONL format.

    Each line has exactly:
        {"id": "<top-level input key>", "prediction": <integer 1..5>}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for sample_id, prediction in predictions:
            record = {
                "id": str(sample_id),
                "prediction": int(prediction),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def predict_file(input_path: Path, output_path: Path, train_path: Path | None = None) -> None:
    """
    Load input samples, run the mean baseline, and write official JSONL predictions.
    """
    input_data = load_json_dict(input_path)
    model = load_or_default_model(train_path)
    predictions = model.predict_many(input_data)
    write_predictions_jsonl(predictions, output_path)

    print(f"[baseline_mean] Mean score used: {model.mean_score:.12f}")
    print(f"[baseline_mean] Integer prediction: {model.predict_one()}")
    print(f"[baseline_mean] Wrote {len(predictions)} predictions to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the mean-score baseline for the AmbiStory project."
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train.json"),
        help="Path to the training JSON file. Default: data/train.json",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input JSON file, for example data/dev.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the prediction JSONL file will be written.",
    )

    args = parser.parse_args()

    predict_file(
        input_path=args.input,
        output_path=args.output,
        train_path=args.train,
    )


if __name__ == "__main__":
    main()
