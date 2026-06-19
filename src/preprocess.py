#!/usr/bin/env python3
"""
preprocess.py

Preprocessing script for the NLP Summer 2026 AmbiStory project.

Run from the project root, for example:

    python src/preprocess.py

Expected project structure:

    NlpSummer2026/
    ├── data/
    │   ├── train.json
    │   └── dev.json
    └── src/
        └── preprocess.py

This script:
1. reads the raw train/dev JSON files,
2. validates that the expected fields are present,
3. cleans simple whitespace problems,
4. creates a single text input for modelling,
5. saves processed CSV and JSONL files,
6. writes a small preprocessing report with dataset statistics.

The script does NOT modify the raw JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


REQUIRED_FIELDS = [
    "homonym",
    "judged_meaning",
    "precontext",
    "sentence",
    "ending",
    "choices",
    "average",
    "stdev",
    "nonsensical",
    "sample_id",
    "example_sentence",
]


OUTPUT_COLUMNS = [
    "row_id",
    "sample_id",
    "split",
    "homonym",
    "judged_meaning",
    "precontext",
    "sentence",
    "ending",
    "example_sentence",
    "choices_json",
    "nonsensical_json",
    "average",
    "stdev",
    "has_ending",
    "story_text",
    "meaning_text",
    "model_input",
]


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        try:
            keys = sorted(raw.keys(), key=lambda x: int(x))
        except ValueError:
            keys = sorted(raw.keys())

        rows = []
        for key in keys:
            row = raw[key]
            row["_row_id"] = key
            rows.append(row)
        return rows

    if isinstance(raw, list):
        rows = []
        for i, row in enumerate(raw):
            row["_row_id"] = str(i)
            rows.append(row)
        return rows

    raise ValueError(
        f"Unsupported JSON format in {path}. Expected dict or list, got {type(raw)}."
    )


def validate_row(row: Dict[str, Any], path: Path, index: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(
            f"Missing fields in {path}, row {index}: {missing}"
        )


def build_story_text(precontext: str, sentence: str, ending: str) -> str:
    if ending:
        ending_part = ending
    else:
        ending_part = "NO_ENDING"

    return (
        f"Precontext: {precontext} "
        f"Ambiguous sentence: {sentence} "
        f"Ending: {ending_part}"
    )


def build_meaning_text(homonym: str, judged_meaning: str, example_sentence: str) -> str:
    return (
        f"Homonym: {homonym}. "
        f"Candidate meaning: {judged_meaning}. "
        f"Example sentence for this meaning: {example_sentence}"
    )


def build_model_input(story_text: str, meaning_text: str) -> str:
    return f"{story_text} [SEP] {meaning_text}"


def preprocess_rows(
    rows: List[Dict[str, Any]],
    split: str,
    source_path: Path,
) -> List[Dict[str, Any]]:
    processed = []

    for i, row in enumerate(rows):
        validate_row(row, source_path, i)

        homonym = normalize_whitespace(row["homonym"])
        judged_meaning = normalize_whitespace(row["judged_meaning"])
        precontext = normalize_whitespace(row["precontext"])
        sentence = normalize_whitespace(row["sentence"])
        ending = normalize_whitespace(row["ending"])
        example_sentence = normalize_whitespace(row["example_sentence"])

        story_text = build_story_text(
            precontext=precontext,
            sentence=sentence,
            ending=ending,
        )

        meaning_text = build_meaning_text(
            homonym=homonym,
            judged_meaning=judged_meaning,
            example_sentence=example_sentence,
        )

        model_input = build_model_input(
            story_text=story_text,
            meaning_text=meaning_text,
        )

        processed.append(
            {
                "row_id": row.get("_row_id", str(i)),
                "sample_id": normalize_whitespace(row["sample_id"]),
                "split": split,
                "homonym": homonym,
                "judged_meaning": judged_meaning,
                "precontext": precontext,
                "sentence": sentence,
                "ending": ending,
                "example_sentence": example_sentence,
                "choices_json": json.dumps(row["choices"], ensure_ascii=False),
                "nonsensical_json": json.dumps(row["nonsensical"], ensure_ascii=False),
                "average": float(row["average"]),
                "stdev": float(row["stdev"]),
                "has_ending": bool(ending),
                "story_text": story_text,
                "meaning_text": meaning_text,
                "model_input": model_input,
            }
        )

    return processed


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    averages = [row["average"] for row in rows]
    stdevs = [row["stdev"] for row in rows]

    unique_homonyms = sorted(set(row["homonym"] for row in rows))
    unique_sample_ids = sorted(set(row["sample_id"] for row in rows))

    rows_with_ending = sum(1 for row in rows if row["has_ending"])
    rows_without_ending = len(rows) - rows_with_ending

    return {
        "n_rows": len(rows),
        "n_unique_homonyms": len(unique_homonyms),
        "n_unique_sample_ids": len(unique_sample_ids),
        "n_rows_with_ending": rows_with_ending,
        "n_rows_without_ending": rows_without_ending,
        "average_min": min(averages) if averages else None,
        "average_max": max(averages) if averages else None,
        "average_mean": mean(averages) if averages else None,
        "stdev_min": min(stdevs) if stdevs else None,
        "stdev_max": max(stdevs) if stdevs else None,
        "stdev_mean": mean(stdevs) if stdevs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess train/dev JSON files for the NLP Summer 2026 project."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.json and dev.json. Default: data",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed files will be saved. Default: data/processed",
    )

    args = parser.parse_args()

    train_path = args.data_dir / "train.json"
    dev_path = args.data_dir / "dev.json"

    print(f"Loading train data from: {train_path}")
    train_raw = load_json_dataset(train_path)

    print(f"Loading dev data from: {dev_path}")
    dev_raw = load_json_dataset(dev_path)

    print("Preprocessing train split...")
    train_processed = preprocess_rows(train_raw, split="train", source_path=train_path)

    print("Preprocessing dev split...")
    dev_processed = preprocess_rows(dev_raw, split="dev", source_path=dev_path)

    all_processed = train_processed + dev_processed

    print("Saving processed files...")
    write_csv(train_processed, args.output_dir / "train_processed.csv")
    write_csv(dev_processed, args.output_dir / "dev_processed.csv")
    write_csv(all_processed, args.output_dir / "all_processed.csv")

    write_jsonl(train_processed, args.output_dir / "train_processed.jsonl")
    write_jsonl(dev_processed, args.output_dir / "dev_processed.jsonl")
    write_jsonl(all_processed, args.output_dir / "all_processed.jsonl")

    report = {
        "train": dataset_statistics(train_processed),
        "dev": dataset_statistics(dev_processed),
    }

    report_path = args.output_dir / "preprocessing_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Processed files saved in: {args.output_dir}")
    print(f"Preprocessing report saved to: {report_path}")

    print("\nDataset statistics:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()