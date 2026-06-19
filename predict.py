#!/usr/bin/env python3
"""
predict.py

Official prediction entry point.

Usage:
    python predict.py <input_json> <output_jsonl>

To switch models, only change ACTIVE_MODEL below.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Change only this value:
# Options: "mean", "tfidf", "cnn", "roberta"
ACTIVE_MODEL = "roberta"


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_json> <output_jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    repo_root = Path(__file__).resolve().parent
    train_path = repo_root / "data" / "train.json"

    if ACTIVE_MODEL == "mean":
        from src.baseline_mean import predict_file

        predict_file(
            input_path=input_path,
            output_path=output_path,
            train_path=train_path,
        )

    elif ACTIVE_MODEL == "tfidf":
        from src.baseline_tfidf import predict_file

        predict_file(
            input_path=input_path,
            output_path=output_path,
            train_path=train_path,
        )

    elif ACTIVE_MODEL == "cnn":
        from src.model_cnn import predict_file

        model_dir = repo_root / "models" / "cnn_regressor_gpu"

        predict_file(
            input_path=input_path,
            output_path=output_path,
            train_path=train_path,
            model_dir=model_dir,
            auto_train_if_missing=True,
        )

    elif ACTIVE_MODEL == "roberta":
        from src.model_roberta import predict_file

        model_dir = repo_root / "models" / "roberta_regressor_gpu_e3_lr1e5"

        predict_file(
            input_path=input_path,
            output_path=output_path,
            train_path=train_path,
            model_dir=model_dir,
            auto_train_if_missing=True,
        )

    else:
        raise ValueError(
            f"Unknown ACTIVE_MODEL: {ACTIVE_MODEL}. "
            "Use one of: mean, tfidf, cnn, roberta."
        )


if __name__ == "__main__":
    main()