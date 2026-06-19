#!/usr/bin/env python3
"""
baseline_tfidf.py

TF-IDF + linear regression baseline for the NLP Summer 2026 AmbiStory project.

The model represents each story-meaning pair with TF-IDF lexical features and
trains a linear regression model to predict the human average plausibility score.

This file contains the model logic. The root-level predict.py file should import
this file and use it as the official entry point.

Development usage from the project root:

    python src/baseline_tfidf.py \
        --train data/train.json \
        --input data/dev.json \
        --output results/dev_predictions_tfidf.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import Pipeline
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "scikit-learn is required for the TF-IDF baseline.\n"
        "Install the project dependencies with:\n\n"
        "    python -m pip install -r requirements.txt\n"
    ) from exc


def load_json_dict(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a project JSON file.

    Expected format:
        {
            "0": {...},
            "1": {...},
            ...
        }

    The top-level keys are important because the official prediction file must
    use these keys as the output IDs.
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

    The official output can be in any order, but deterministic order makes the
    output easier to inspect and compare.
    """
    def sort_key(sample_id: str) -> Tuple[int, object]:
        sample_id = str(sample_id)
        return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)

    return sorted((str(sample_id) for sample_id in data.keys()), key=sort_key)


def normalize_text(value: Any) -> str:
    """
    Convert a field value to a clean single-line string.
    """
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_model_input(sample: Dict[str, Any]) -> str:
    """
    Build the text input used by the TF-IDF model.

    We model each row as a story-meaning pair by concatenating:

    - precontext
    - ambiguous sentence
    - ending, with NO_ENDING if the ending is empty
    - homonym
    - judged meaning
    - example sentence

    Section labels are included so that the linear model can distinguish, for
    example, words appearing in the story from words appearing in the candidate
    meaning.
    """
    ending = normalize_text(sample.get("ending", ""))
    ending_text = ending if ending else "NO_ENDING"

    parts = [
        "PRECONTEXT", normalize_text(sample.get("precontext", "")),
        "SENTENCE", normalize_text(sample.get("sentence", "")),
        "ENDING", ending_text,
        "HOMONYM", normalize_text(sample.get("homonym", "")),
        "JUDGED_MEANING", normalize_text(sample.get("judged_meaning", "")),
        "EXAMPLE_SENTENCE", normalize_text(sample.get("example_sentence", "")),
    ]

    return " ".join(parts)


def extract_training_examples(
    training_data: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], List[float]]:
    """
    Convert training samples into model inputs and continuous target scores.
    """
    texts: List[str] = []
    targets: List[float] = []

    for sample_id in sorted_sample_ids(training_data):
        sample = training_data[sample_id]

        if "average" not in sample:
            raise ValueError(
                f"Training sample {sample_id!r} is missing the field 'average'."
            )

        try:
            target = float(sample["average"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Training sample {sample_id!r} has a non-numeric average: "
                f"{sample['average']!r}."
            ) from exc

        texts.append(build_model_input(sample))
        targets.append(target)

    if not texts:
        raise ValueError("Cannot train TF-IDF baseline from an empty dataset.")

    return texts, targets


def clip_score(score: float, lower: int = 1, upper: int = 5) -> float:
    """
    Clip a score to the valid project range [1, 5].
    """
    return max(lower, min(upper, float(score)))


def round_to_valid_integer(score: float) -> int:
    """
    Convert a continuous regression score to an official integer score from 1 to 5.

    We use conventional rounding via floor(score + 0.5), then clip to [1, 5].
    """
    rounded = math.floor(float(score) + 0.5)
    return int(clip_score(rounded, lower=1, upper=5))


class TfidfLinearRegressionModel:
    """
    TF-IDF + linear regression baseline.

    The TF-IDF part creates lexical features from the concatenated story-meaning
    input. The regression part predicts the continuous human average score.
    Official predictions are obtained by rounding and clipping to integers 1..5.
    """

    def __init__(
        self,
        max_features: int = 12000,
        min_df: int = 2,
        ngram_range: Tuple[int, int] = (1, 2),
        alpha: float = 1e-4,
        random_state: int = 13,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.random_state = random_state

        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_features=self.max_features,
                        sublinear_tf=True,
                        norm="l2",
                    ),
                ),
                (
                    "regressor",
                    SGDRegressor(
                        loss="squared_error",
                        penalty="l2",
                        alpha=self.alpha,
                        max_iter=1000,
                        tol=1e-4,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, training_data: Dict[str, Dict[str, Any]]) -> "TfidfLinearRegressionModel":
        """
        Fit the TF-IDF vectorizer and linear regression model.
        """
        texts, targets = extract_training_examples(training_data)
        self.pipeline.fit(texts, targets)
        return self

    def predict_continuous_one(self, sample: Dict[str, Any]) -> float:
        """
        Predict one continuous plausibility score before rounding.
        """
        text = build_model_input(sample)
        prediction = self.pipeline.predict([text])[0]
        return float(prediction)

    def predict_one(self, sample: Dict[str, Any]) -> int:
        """
        Predict one official integer plausibility score from 1 to 5.
        """
        return round_to_valid_integer(self.predict_continuous_one(sample))

    def predict_many(self, data: Dict[str, Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Predict one score for every sample.

        Returns:
            List of (top_level_sample_id, prediction) pairs.
        """
        sample_ids = sorted_sample_ids(data)
        texts = [build_model_input(data[sample_id]) for sample_id in sample_ids]

        continuous_predictions = self.pipeline.predict(texts)
        integer_predictions = [
            round_to_valid_integer(score) for score in continuous_predictions
        ]

        return list(zip(sample_ids, integer_predictions))


def write_predictions_jsonl(
    predictions: Iterable[Tuple[str, int]],
    output_path: Path,
) -> None:
    """
    Write official prediction JSONL.

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


def train_model_from_file(train_path: Path) -> TfidfLinearRegressionModel:
    """
    Load train.json and fit the TF-IDF baseline.
    """
    training_data = load_json_dict(train_path)
    model = TfidfLinearRegressionModel()
    model.fit(training_data)
    return model


def predict_file(input_path: Path, output_path: Path, train_path: Path) -> None:
    """
    Train on train_path, predict for input_path, and write official JSONL output.
    """
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "The TF-IDF baseline needs data/train.json in the repository so it "
            "can fit the vectorizer and regression model before predicting."
        )

    print(f"[baseline_tfidf] Loading training data from: {train_path}")
    training_data = load_json_dict(train_path)

    print(f"[baseline_tfidf] Fitting TF-IDF + linear regression on {len(training_data)} samples.")
    model = TfidfLinearRegressionModel()
    model.fit(training_data)

    print(f"[baseline_tfidf] Loading input data from: {input_path}")
    input_data = load_json_dict(input_path)

    predictions = model.predict_many(input_data)
    write_predictions_jsonl(predictions, output_path)

    print(f"[baseline_tfidf] Wrote {len(predictions)} predictions to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TF-IDF + linear regression baseline."
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
