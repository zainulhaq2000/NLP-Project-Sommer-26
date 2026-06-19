#!/usr/bin/env python3
"""
metrics.py

Evaluation metrics for the NLP Summer 2026 AmbiStory project.

Official project metrics
------------------------
The project description defines two primary evaluation metrics:

1. Spearman correlation
   Measures whether predicted plausibility scores rank the examples similarly
   to the human average ratings.

2. Accuracy within standard deviation
   A prediction is counted as correct if it is close enough to the average
   human rating:
       abs(prediction - average) <= max(stdev, 1.0)

Additional diagnostic regression metrics
----------------------------------------
This file also computes MAE, MSE, and RMSE. These are useful for debugging
regression models, but they do not replace the official project metrics.

Prediction file formats supported
---------------------------------
This script supports two prediction formats:

1. Internal CSV format, useful during development:
       row_id,prediction
       0,3.14
       1,2.80

2. Official JSONL format, required by the coding standards:
       {"id": "0", "prediction": 3}
       {"id": "1", "prediction": 4}

Gold file formats supported
---------------------------
This script supports two gold formats:

1. Processed CSV from preprocess.py:
       row_id,average,stdev,...

2. Raw project JSON:
       {
         "0": {"average": 3.6, "stdev": 1.67, ...},
         "1": {"average": 4.0, "stdev": 1.00, ...}
       }

The implementation uses only the Python standard library.

Examples
--------
Run a built-in metric test with fake predictions:

    python src/metrics.py --demo

Evaluate internal CSV predictions against processed dev CSV:

    python src/metrics.py \
        --gold data/processed/dev_processed.csv \
        --predictions results/predictions_mean.csv

Evaluate official JSONL predictions against raw dev JSON:

    python src/metrics.py \
        --gold data/dev.json \
        --predictions results/dev_predictions.jsonl

The format is usually inferred automatically from the file extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Basic checks and score handling
# ---------------------------------------------------------------------------

def check_same_length(*sequences: Sequence[object]) -> None:
    """
    Check that all sequences have the same length.

    This prevents silent metric mistakes, for example comparing 588 gold
    labels with only 587 predictions.
    """
    if not sequences:
        return

    expected = len(sequences[0])

    for index, sequence in enumerate(sequences[1:], start=2):
        if len(sequence) != expected:
            raise ValueError(
                f"All inputs must have the same length. "
                f"Input 1 has length {expected}, but input {index} has "
                f"length {len(sequence)}."
            )


def check_not_empty(sequence: Sequence[object], name: str = "sequence") -> None:
    """
    Raise an error if a sequence is empty.
    """
    if len(sequence) == 0:
        raise ValueError(f"Cannot compute metrics on empty {name}.")


def to_float_list(values: Sequence[object], name: str) -> List[float]:
    """
    Convert a sequence to a list of floats with a helpful error message.
    """
    converted: List[float] = []

    for index, value in enumerate(values):
        try:
            converted.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not convert {name}[{index}]={value!r} to float."
            ) from exc

    return converted


def clip_score(score: float, lower: float = 1.0, upper: float = 5.0) -> float:
    """
    Clip one score to the valid project score range [1, 5].

    Regression models may output values such as 0.8 or 5.2. Since the project
    target is a score from 1 to 5, final continuous predictions should be clipped
    before evaluation.
    """
    score = float(score)
    return max(lower, min(upper, score))


def clip_scores(scores: Sequence[float], lower: float = 1.0, upper: float = 5.0) -> List[float]:
    """
    Clip a list of scores to the valid project score range [1, 5].
    """
    return [clip_score(score, lower=lower, upper=upper) for score in scores]


def round_to_official_integer(score: float) -> int:
    """
    Convert a continuous score to the official integer range 1..5.

    This is useful when creating or checking official JSONL predictions, because
    the coding standards require prediction to be an integer between 1 and 5.
    """
    rounded = int(round(float(score)))
    return int(clip_score(rounded, lower=1.0, upper=5.0))


# ---------------------------------------------------------------------------
# Official metric 1: Spearman correlation
# ---------------------------------------------------------------------------

def average_ranks(values: Sequence[float]) -> List[float]:
    """
    Convert numeric values to ranks, using average ranks for ties.

    Ranks start at 1.

    Example:
        values = [10, 20, 20, 30]

        10 receives rank 1.
        The two 20 values share ranks 2 and 3, so both receive rank 2.5.
        30 receives rank 4.

        output = [1.0, 2.5, 2.5, 4.0]
    """
    check_not_empty(values, "values")

    indexed_values = list(enumerate(values))
    indexed_values.sort(key=lambda pair: pair[1])

    ranks = [0.0] * len(values)
    position = 0

    while position < len(indexed_values):
        tie_end = position

        while (
            tie_end + 1 < len(indexed_values)
            and indexed_values[tie_end + 1][1] == indexed_values[position][1]
        ):
            tie_end += 1

        # Positions position..tie_end correspond to 1-based ranks
        # position+1..tie_end+1.
        average_rank = ((position + 1) + (tie_end + 1)) / 2.0

        for tied_position in range(position, tie_end + 1):
            original_index = indexed_values[tied_position][0]
            ranks[original_index] = average_rank

        position = tie_end + 1

    return ranks


def pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    """
    Compute Pearson correlation between two numeric sequences.

    Spearman correlation is computed by first turning values into ranks and then
    applying Pearson correlation to those rank vectors.
    """
    check_same_length(x_values, y_values)
    check_not_empty(x_values, "x_values")

    x = to_float_list(x_values, "x_values")
    y = to_float_list(y_values, "y_values")

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    denominator_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    denominator = denominator_x * denominator_y

    # If one vector is constant, correlation is mathematically undefined.
    # We return 0.0 to keep the project pipeline robust.
    if denominator == 0:
        return 0.0

    return numerator / denominator


def spearman_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Compute Spearman rank correlation.

    This metric focuses on ranking:
        Do examples with higher human scores also receive higher predicted scores?

    Ties are handled with average ranks.
    """
    check_same_length(y_true, y_pred)
    check_not_empty(y_true, "y_true")

    gold = to_float_list(y_true, "y_true")
    pred = to_float_list(y_pred, "y_pred")

    gold_ranks = average_ranks(gold)
    pred_ranks = average_ranks(pred)

    return pearson_correlation(gold_ranks, pred_ranks)


# ---------------------------------------------------------------------------
# Official metric 2: Accuracy within standard deviation
# ---------------------------------------------------------------------------

def accuracy_within_standard_deviation(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    stdevs: Sequence[float],
    min_tolerance: float = 1.0,
) -> float:
    """
    Compute accuracy within standard deviation.

    For each example, the prediction is correct if:

        abs(prediction - average_human_rating) <= max(stdev, min_tolerance)

    The project specifies "within standard deviation (at least 1)".
    Therefore, if stdev is smaller than 1, we still use tolerance 1.
    """
    check_same_length(y_true, y_pred, stdevs)
    check_not_empty(y_true, "y_true")

    gold = to_float_list(y_true, "y_true")
    pred = to_float_list(y_pred, "y_pred")
    stdev_values = to_float_list(stdevs, "stdevs")

    correct = 0

    for gold_i, pred_i, stdev_i in zip(gold, pred, stdev_values):
        tolerance = max(stdev_i, min_tolerance)
        error = abs(pred_i - gold_i)

        if error <= tolerance:
            correct += 1

    return correct / len(gold)


# ---------------------------------------------------------------------------
# Additional diagnostic regression metrics
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Compute mean absolute error.

    MAE measures the average absolute distance between predictions and gold
    scores. It is easy to interpret because it is expressed in score points.
    """
    check_same_length(y_true, y_pred)
    check_not_empty(y_true, "y_true")

    gold = to_float_list(y_true, "y_true")
    pred = to_float_list(y_pred, "y_pred")

    return sum(abs(gold_i - pred_i) for gold_i, pred_i in zip(gold, pred)) / len(gold)


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Compute mean squared error.

    MSE penalizes large errors more strongly because the errors are squared.
    It is useful for regression diagnostics and for checking models trained with
    squared-error loss.
    """
    check_same_length(y_true, y_pred)
    check_not_empty(y_true, "y_true")

    gold = to_float_list(y_true, "y_true")
    pred = to_float_list(y_pred, "y_pred")

    return sum((gold_i - pred_i) ** 2 for gold_i, pred_i in zip(gold, pred)) / len(gold)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Compute root mean squared error.

    RMSE is the square root of MSE, so it is again expressed in score points.
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    stdevs: Sequence[float],
    clip: bool = True,
) -> Dict[str, float]:
    """
    Compute official metrics and additional diagnostic metrics.

    Returned metrics:
        spearman
        accuracy_within_stdev
        mae
        mse
        rmse
        n_examples

    If clip=True, predictions are clipped to [1, 5] before evaluation.
    """
    check_same_length(y_true, y_pred, stdevs)
    check_not_empty(y_true, "y_true")

    gold = to_float_list(y_true, "y_true")
    pred = to_float_list(y_pred, "y_pred")
    stdev_values = to_float_list(stdevs, "stdevs")

    if clip:
        pred = clip_scores(pred)

    return {
        "spearman": spearman_correlation(gold, pred),
        "accuracy_within_stdev": accuracy_within_standard_deviation(gold, pred, stdev_values),
        "mae": mean_absolute_error(gold, pred),
        "mse": mean_squared_error(gold, pred),
        "rmse": root_mean_squared_error(gold, pred),
        "n_examples": float(len(gold)),
    }


# ---------------------------------------------------------------------------
# Gold-file loading
# ---------------------------------------------------------------------------

def load_gold_csv(path: Path) -> Dict[str, Tuple[float, float]]:
    """
    Load gold labels from a processed CSV file.

    The file should contain at least:
        row_id, average, stdev

    Returns:
        {
            row_id: (average, stdev),
            ...
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path}")

    gold: Dict[str, Tuple[float, float]] = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"row_id", "average", "stdev"}
        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                f"Gold CSV file {path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            row_id = str(row["row_id"])
            average = float(row["average"])
            stdev = float(row["stdev"])
            gold[row_id] = (average, stdev)

    return gold


def load_gold_json(path: Path) -> Dict[str, Tuple[float, float]]:
    """
    Load gold labels from a raw project JSON file.

    The raw train/dev/test-like files are JSON objects whose top-level keys are
    string IDs and whose values are samples containing average and stdev.

    Returns:
        {
            top_level_id: (average, stdev),
            ...
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Gold file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Gold JSON file {path} must be a dictionary keyed by sample IDs."
        )

    gold: Dict[str, Tuple[float, float]] = {}

    for sample_id, sample in data.items():
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {sample_id!r} in {path} is not an object.")

        missing = {"average", "stdev"} - set(sample.keys())
        if missing:
            raise ValueError(
                f"Sample {sample_id!r} in {path} is missing fields: {sorted(missing)}"
            )

        gold[str(sample_id)] = (float(sample["average"]), float(sample["stdev"]))

    return gold


def load_gold_file(path: Path, gold_format: str = "auto") -> Dict[str, Tuple[float, float]]:
    """
    Load gold labels from either processed CSV or raw project JSON.

    gold_format can be:
        auto
        csv
        json
    """
    if gold_format == "auto":
        suffix = path.suffix.lower()

        if suffix == ".csv":
            gold_format = "csv"
        elif suffix == ".json":
            gold_format = "json"
        else:
            raise ValueError(
                f"Could not infer gold format from extension {suffix!r}. "
                f"Use --gold-format csv or --gold-format json."
            )

    if gold_format == "csv":
        return load_gold_csv(path)

    if gold_format == "json":
        return load_gold_json(path)

    raise ValueError(
        f"Unsupported gold format: {gold_format!r}. "
        f"Use one of: auto, csv, json."
    )


# ---------------------------------------------------------------------------
# Prediction-file loading
# ---------------------------------------------------------------------------

def load_prediction_csv(
    path: Path,
    prediction_column: str = "prediction",
) -> Dict[str, float]:
    """
    Load predictions from an internal CSV file.

    The file should contain:
        row_id, prediction

    The prediction column name can be changed with prediction_column.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions: Dict[str, float] = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"row_id", prediction_column}
        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                f"Prediction CSV file {path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            row_id = str(row["row_id"])
            prediction = float(row[prediction_column])
            predictions[row_id] = prediction

    return predictions


def load_prediction_jsonl(path: Path, strict: bool = True) -> Dict[str, float]:
    """
    Load predictions from the official JSONL format.

    Expected line format:
        {"id": "42", "prediction": 3}

    If strict=True, each JSON object must have exactly two keys: id and prediction.
    The prediction must be an integer in the range 1..5.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions: Dict[str, float] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} of {path} must contain a JSON object."
                )

            required_keys = {"id", "prediction"}

            if strict and set(row.keys()) != required_keys:
                raise ValueError(
                    f"Line {line_number} of {path} must have exactly the keys "
                    f"{sorted(required_keys)}, but got {sorted(row.keys())}."
                )

            missing = required_keys - set(row.keys())
            if missing:
                raise ValueError(
                    f"Line {line_number} of {path} is missing keys: {sorted(missing)}"
                )

            sample_id = str(row["id"])
            prediction = row["prediction"]

            if strict:
                if not isinstance(prediction, int):
                    raise ValueError(
                        f"Line {line_number} of {path}: prediction must be an "
                        f"integer from 1 to 5, but got {prediction!r}."
                    )

                if prediction < 1 or prediction > 5:
                    raise ValueError(
                        f"Line {line_number} of {path}: prediction must be in "
                        f"the range 1..5, but got {prediction}."
                    )

            prediction = float(prediction)

            if sample_id in predictions:
                raise ValueError(
                    f"Duplicate prediction id {sample_id!r} on line {line_number} "
                    f"of {path}."
                )

            predictions[sample_id] = prediction

    return predictions


def load_prediction_file(
    path: Path,
    prediction_format: str = "auto",
    prediction_column: str = "prediction",
    strict_jsonl: bool = True,
) -> Dict[str, float]:
    """
    Load predictions from either internal CSV or official JSONL.

    prediction_format can be:
        auto
        csv
        jsonl
    """
    if prediction_format == "auto":
        suffix = path.suffix.lower()

        if suffix == ".csv":
            prediction_format = "csv"
        elif suffix == ".jsonl":
            prediction_format = "jsonl"
        else:
            raise ValueError(
                f"Could not infer prediction format from extension {suffix!r}. "
                f"Use --prediction-format csv or --prediction-format jsonl."
            )

    if prediction_format == "csv":
        return load_prediction_csv(path, prediction_column=prediction_column)

    if prediction_format == "jsonl":
        return load_prediction_jsonl(path, strict=strict_jsonl)

    raise ValueError(
        f"Unsupported prediction format: {prediction_format!r}. "
        f"Use one of: auto, csv, jsonl."
    )


# ---------------------------------------------------------------------------
# File-based evaluation
# ---------------------------------------------------------------------------

def sorted_ids(ids: Sequence[str]) -> List[str]:
    """
    Sort IDs numerically when possible, otherwise lexicographically.
    """
    def sort_key(sample_id: str):
        return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)

    return sorted(ids, key=sort_key)


def evaluate_prediction_file(
    gold_path: Path,
    predictions_path: Path,
    gold_format: str = "auto",
    prediction_format: str = "auto",
    prediction_column: str = "prediction",
    strict_jsonl: bool = True,
    clip: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a prediction file against a gold file.

    Gold and predictions are matched by ID:
        - processed CSV gold uses row_id
        - raw JSON gold uses the top-level JSON key
        - official JSONL predictions use id
        - internal CSV predictions use row_id
    """
    gold = load_gold_file(gold_path, gold_format=gold_format)
    predictions = load_prediction_file(
        predictions_path,
        prediction_format=prediction_format,
        prediction_column=prediction_column,
        strict_jsonl=strict_jsonl,
    )

    gold_ids = set(gold.keys())
    prediction_ids = set(predictions.keys())

    missing = sorted_ids(list(gold_ids - prediction_ids))
    extra = sorted_ids(list(prediction_ids - gold_ids))

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Prediction file is missing {len(missing)} id values. "
            f"First missing ids: {preview}"
        )

    if extra:
        preview = ", ".join(extra[:10])
        raise ValueError(
            f"Prediction file contains {len(extra)} extra id values. "
            f"First extra ids: {preview}"
        )

    ids = sorted_ids(list(gold_ids))

    y_true = [gold[sample_id][0] for sample_id in ids]
    stdevs = [gold[sample_id][1] for sample_id in ids]
    y_pred = [predictions[sample_id] for sample_id in ids]

    return evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        stdevs=stdevs,
        clip=clip,
    )


def write_metrics_json(metrics: Dict[str, float], path: Path) -> None:
    """
    Save metric results as a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Demo with fake predictions
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Test the metric functions on small fake predictions.

    This is the first sanity check before evaluating real model outputs.
    """
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    stdevs = [0.2, 0.8, 1.5, 0.6, 0.4]

    fake_systems = {
        "perfect_predictions": [1.0, 2.0, 3.0, 4.0, 5.0],
        "almost_perfect_predictions": [1.2, 2.1, 3.2, 3.9, 4.7],
        "reversed_predictions": [5.0, 4.0, 3.0, 2.0, 1.0],
        "constant_middle_predictions": [3.0, 3.0, 3.0, 3.0, 3.0],
        "out_of_range_predictions": [0.5, 2.0, 3.0, 4.0, 5.7],
    }

    print("Metric demo with fake predictions")
    print("=" * 50)

    for system_name, predictions in fake_systems.items():
        metrics = evaluate_predictions(
            y_true=y_true,
            y_pred=predictions,
            stdevs=stdevs,
            clip=True,
        )

        print(f"\n{system_name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    print("\nOfficial integer conversion demo")
    print("=" * 50)
    for score in [0.6, 1.2, 2.7, 4.5, 5.9]:
        print(f"  {score} -> {round_to_official_integer(score)}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for the NLP Summer 2026 AmbiStory project."
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a small demo with fake predictions.",
    )

    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help=(
            "Path to gold file. Can be processed CSV, e.g. "
            "data/processed/dev_processed.csv, or raw JSON, e.g. data/dev.json."
        ),
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Path to prediction file. Can be internal CSV or official JSONL."
        ),
    )

    parser.add_argument(
        "--gold-format",
        choices=["auto", "csv", "json"],
        default="auto",
        help="Gold file format. Default: auto.",
    )

    parser.add_argument(
        "--prediction-format",
        choices=["auto", "csv", "jsonl"],
        default="auto",
        help="Prediction file format. Default: auto.",
    )

    parser.add_argument(
        "--prediction-column",
        type=str,
        default="prediction",
        help="Prediction column name for CSV predictions. Default: prediction.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON.",
    )

    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip predictions to the valid project score range [1, 5].",
    )

    parser.add_argument(
        "--relaxed-jsonl",
        action="store_true",
        help=(
            "Allow JSONL prediction objects to contain extra keys and non-integer "
            "numeric predictions. By default, JSONL predictions are checked "
            "strictly against the official output format."
        ),
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.gold is None or args.predictions is None:
        parser.error("Use --demo, or provide both --gold and --predictions.")

    metrics = evaluate_prediction_file(
        gold_path=args.gold,
        predictions_path=args.predictions,
        gold_format=args.gold_format,
        prediction_format=args.prediction_format,
        prediction_column=args.prediction_column,
        strict_jsonl=not args.relaxed_jsonl,
        clip=not args.no_clip,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output is not None:
        write_metrics_json(metrics, args.output)
        print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
    main()
