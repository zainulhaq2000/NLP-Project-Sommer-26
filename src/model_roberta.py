#!/usr/bin/env python3
"""
model_roberta.py

RoBERTa regression model for the NLP Summer 2026 AmbiStory project.

The model represents each story-meaning pair with a RoBERTa transformer encoder
and fine-tunes a regression head to predict the human average plausibility score.

Development commands from the project root:

    python src/model_roberta.py train \
        --train data/train.json \
        --model_dir models/roberta_regressor

    python src/model_roberta.py predict \
        --input data/dev.json \
        --output results/dev_predictions_roberta.jsonl \
        --model_dir models/roberta_regressor

    python src/model_roberta.py train-and-predict \
        --train data/train.json \
        --input data/dev.json \
        --output results/dev_predictions_roberta.jsonl \
        --model_dir models/roberta_regressor

The official root-level predict.py can import predict_file from this module.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The RoBERTa model requires torch and transformers.\n"
        "Install the project dependencies with:\n\n"
        "    python -m pip install -r requirements.txt\n"
    ) from exc


DEFAULT_BASE_MODEL = "roberta-base"
DEFAULT_MODEL_DIR = Path("models/roberta_regressor")
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 2
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_RANDOM_SEED = 13


def set_random_seed(seed: int) -> None:
    """
    Make training as reproducible as reasonably possible.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_dict(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a project JSON file.

    Expected input format:
        {
            "0": {...},
            "1": {...},
            ...
        }

    The top-level keys are preserved because official prediction IDs must match
    these keys.
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
            raise ValueError(f"Sample {sample_id!r} is not a JSON object.")

    return data


def sorted_sample_ids(data: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Sort sample IDs numerically when possible, otherwise lexicographically.
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


def build_roberta_text_pair(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Build the two text segments passed to RoBERTa.

    Segment A describes the narrative context.
    Segment B describes the candidate meaning being judged.

    The model can then learn whether the candidate meaning is plausible in the
    story context.
    """
    ending = normalize_text(sample.get("ending", ""))
    ending_text = ending if ending else "NO_ENDING"

    story_text = " ".join(
        [
            "Context:",
            normalize_text(sample.get("precontext", "")),
            "Ambiguous sentence:",
            normalize_text(sample.get("sentence", "")),
            "Ending:",
            ending_text,
        ]
    )

    meaning_text = " ".join(
        [
            "Homonym:",
            normalize_text(sample.get("homonym", "")),
            "Candidate meaning:",
            normalize_text(sample.get("judged_meaning", "")),
            "Example:",
            normalize_text(sample.get("example_sentence", "")),
        ]
    )

    return story_text, meaning_text


def extract_average(sample_id: str, sample: Dict[str, Any]) -> float:
    """
    Extract the continuous target score from a training sample.
    """
    if "average" not in sample:
        raise ValueError(f"Training sample {sample_id!r} is missing 'average'.")

    try:
        return float(sample["average"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Training sample {sample_id!r} has a non-numeric average: "
            f"{sample['average']!r}."
        ) from exc


def clip_score(score: float, lower: int = 1, upper: int = 5) -> float:
    """
    Clip a score to the valid project range [1, 5].
    """
    return max(lower, min(upper, float(score)))


def round_to_valid_integer(score: float) -> int:
    """
    Convert a continuous regression score to an official integer prediction.

    The official prediction must be an integer from 1 to 5.
    """
    rounded = math.floor(float(score) + 0.5)
    return int(clip_score(rounded, lower=1, upper=5))


class AmbiStoryRegressionDataset(Dataset):
    """
    PyTorch dataset for AmbiStory regression.

    For training, labels are the human average scores.
    For prediction, labels are omitted.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        include_labels: bool,
    ) -> None:
        self.data = data
        self.sample_ids = sorted_sample_ids(data)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[index]
        sample = self.data[sample_id]

        story_text, meaning_text = build_roberta_text_pair(sample)

        encoded = self.tokenizer(
            story_text,
            meaning_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, Any] = {
            "sample_id": sample_id,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        if self.include_labels:
            label = extract_average(sample_id, sample)
            item["labels"] = torch.tensor(label, dtype=torch.float)

        return item


def make_device() -> torch.device:
    """
    Use GPU when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobertaRegressionModel:
    """
    RoBERTa model with a regression head.

    It predicts a continuous plausibility score. The official output is obtained
    by rounding and clipping to an integer from 1 to 5.
    """

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            problem_type="regression",
        )

    @classmethod
    def load(cls, model_dir: Path) -> "RobertaRegressionModel":
        """
        Load a fine-tuned RoBERTa regression model from disk.
        """
        config_path = model_dir / "training_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as file:
                saved_config = json.load(file)
            max_length = int(saved_config.get("max_length", DEFAULT_MAX_LENGTH))
            base_model_name = str(saved_config.get("base_model_name", str(model_dir)))
        else:
            max_length = DEFAULT_MAX_LENGTH
            base_model_name = str(model_dir)

        instance = cls.__new__(cls)
        instance.base_model_name = base_model_name
        instance.max_length = max_length
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return instance

    def save(self, model_dir: Path, training_config: Dict[str, Any]) -> None:
        """
        Save the fine-tuned model, tokenizer, and a small training config file.
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        config = {
            "base_model_name": self.base_model_name,
            "max_length": self.max_length,
            **training_config,
        }

        with (model_dir / "training_config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

    def fit(
        self,
        training_data: Dict[str, Dict[str, Any]],
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> "RobertaRegressionModel":
        """
        Fine-tune RoBERTa on the training data.
        """
        set_random_seed(seed)
        device = make_device()
        self.model.to(device)
        self.model.train()

        dataset = AmbiStoryRegressionDataset(
            data=training_data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_labels=True,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_training_steps = max(1, len(dataloader) * epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_training_steps)),
            num_training_steps=total_training_steps,
        )

        print(f"[model_roberta] Device: {device}")
        print(f"[model_roberta] Training samples: {len(dataset)}")
        print(f"[model_roberta] Epochs: {epochs}")
        print(f"[model_roberta] Batch size: {batch_size}")

        for epoch_index in range(epochs):
            total_loss = 0.0

            for step_index, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad(set_to_none=True)

                model_inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device),
                }

                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

                outputs = self.model(**model_inputs)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                total_loss += float(loss.item())

                if step_index % 50 == 0 or step_index == len(dataloader):
                    avg_loss = total_loss / step_index
                    print(
                        f"[model_roberta] Epoch {epoch_index + 1}/{epochs}, "
                        f"step {step_index}/{len(dataloader)}, "
                        f"avg_loss={avg_loss:.4f}"
                    )

        return self

    def predict_continuous_many(
        self,
        data: Dict[str, Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Tuple[str, float]]:
        """
        Predict continuous scores for every sample.
        """
        device = make_device()
        self.model.to(device)
        self.model.eval()

        dataset = AmbiStoryRegressionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            include_labels=False,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions: List[Tuple[str, float]] = []

        with torch.no_grad():
            for batch in dataloader:
                model_inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }

                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

                outputs = self.model(**model_inputs)
                scores = outputs.logits.squeeze(-1).detach().cpu().tolist()

                if isinstance(scores, float):
                    scores = [scores]

                sample_ids = [str(sample_id) for sample_id in batch["sample_id"]]

                for sample_id, score in zip(sample_ids, scores):
                    predictions.append((sample_id, float(score)))

        return predictions

    def predict_many(
        self,
        data: Dict[str, Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Tuple[str, int]]:
        """
        Predict official integer scores for every sample.
        """
        continuous_predictions = self.predict_continuous_many(data, batch_size=batch_size)
        return [
            (sample_id, round_to_valid_integer(score))
            for sample_id, score in continuous_predictions
        ]


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


def train_and_save_model(
    train_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_RANDOM_SEED,
) -> RobertaRegressionModel:
    """
    Fine-tune RoBERTa and save the resulting checkpoint.
    """
    training_data = load_json_dict(train_path)

    model = RobertaRegressionModel(
        base_model_name=base_model_name,
        max_length=max_length,
    )

    model.fit(
        training_data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    model.save(
        model_dir=model_dir,
        training_config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "train_path": str(train_path),
        },
    )

    print(f"[model_roberta] Saved fine-tuned model to: {model_dir}")
    return model


def model_checkpoint_exists(model_dir: Path) -> bool:
    """
    Check whether a saved Hugging Face model directory exists.
    """
    return (
        model_dir.exists()
        and (model_dir / "config.json").exists()
        and (
            (model_dir / "pytorch_model.bin").exists()
            or (model_dir / "model.safetensors").exists()
        )
    )


def predict_with_saved_model(
    input_path: Path,
    output_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Load a saved fine-tuned model and predict for input_path.
    """
    if not model_checkpoint_exists(model_dir):
        raise FileNotFoundError(
            f"No saved RoBERTa checkpoint found in {model_dir}.\n"
            "Train it first with:\n\n"
            "    python src/model_roberta.py train "
            "--train data/train.json --model_dir models/roberta_regressor\n"
        )

    print(f"[model_roberta] Loading fine-tuned model from: {model_dir}")
    model = RobertaRegressionModel.load(model_dir)

    print(f"[model_roberta] Loading input data from: {input_path}")
    input_data = load_json_dict(input_path)

    predictions = model.predict_many(input_data, batch_size=batch_size)
    write_predictions_jsonl(predictions, output_path)

    print(f"[model_roberta] Wrote {len(predictions)} predictions to: {output_path}")


def train_and_predict_file(
    train_path: Path,
    input_path: Path,
    output_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """
    Train RoBERTa, save it, and predict for input_path.
    """
    model = train_and_save_model(
        train_path=train_path,
        model_dir=model_dir,
        base_model_name=base_model_name,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    print(f"[model_roberta] Loading input data from: {input_path}")
    input_data = load_json_dict(input_path)

    predictions = model.predict_many(input_data, batch_size=batch_size)
    write_predictions_jsonl(predictions, output_path)

    print(f"[model_roberta] Wrote {len(predictions)} predictions to: {output_path}")


def predict_file(
    input_path: Path,
    output_path: Path,
    train_path: Path | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    auto_train_if_missing: bool = True,
) -> None:
    """
    Official prediction helper used by root-level predict.py.

    If a saved RoBERTa checkpoint exists, it is loaded and used.

    If no checkpoint exists and auto_train_if_missing=True, the model is trained
    from data/train.json first. This keeps predict.py runnable, but the first run
    can be slow.
    """
    if model_checkpoint_exists(model_dir):
        predict_with_saved_model(
            input_path=input_path,
            output_path=output_path,
            model_dir=model_dir,
        )
        return

    if auto_train_if_missing and train_path is not None and train_path.exists():
        print(
            "[model_roberta] No saved checkpoint found. "
            "Training RoBERTa before prediction."
        )
        train_and_predict_file(
            train_path=train_path,
            input_path=input_path,
            output_path=output_path,
            model_dir=model_dir,
        )
        return

    raise FileNotFoundError(
        f"No saved checkpoint found in {model_dir}, and training data was not available.\n"
        "Either train the model first or provide data/train.json."
    )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments shared by training commands.
    """
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the fine-tuned RoBERTa checkpoint.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base Hugging Face model name. Default: roberta-base",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token sequence length. Default: 256",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of fine-tuning epochs. Default: 2",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size. Lower this if you run out of memory. Default: 8",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate. Default: 2e-5",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay. Default: 0.01",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed. Default: 13",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and run the RoBERTa regression model."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune and save RoBERTa.")
    train_parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train.json"),
        help="Path to train.json.",
    )
    add_shared_arguments(train_parser)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict using a saved fine-tuned RoBERTa checkpoint.",
    )
    predict_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file.",
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL prediction file.",
    )
    predict_parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the fine-tuned RoBERTa checkpoint.",
    )
    predict_parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Prediction batch size. Default: 8",
    )

    train_predict_parser = subparsers.add_parser(
        "train-and-predict",
        help="Fine-tune RoBERTa, save it, and predict.",
    )
    train_predict_parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train.json"),
        help="Path to train.json.",
    )
    train_predict_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file.",
    )
    train_predict_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL prediction file.",
    )
    add_shared_arguments(train_predict_parser)

    args = parser.parse_args()

    if args.command == "train":
        train_and_save_model(
            train_path=args.train,
            model_dir=args.model_dir,
            base_model_name=args.base_model,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )

    elif args.command == "predict":
        predict_with_saved_model(
            input_path=args.input,
            output_path=args.output,
            model_dir=args.model_dir,
            batch_size=args.batch_size,
        )

    elif args.command == "train-and-predict":
        train_and_predict_file(
            train_path=args.train,
            input_path=args.input,
            output_path=args.output,
            model_dir=args.model_dir,
            base_model_name=args.base_model,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
