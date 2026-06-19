#!/usr/bin/env python3
"""
model_cnn.py

CNN regression model for the NLP Summer 2026 AmbiStory project.

The model represents each story-meaning pair as a sequence of word tokens,
maps tokens to embeddings, applies 1D convolution filters, and predicts the
human average plausibility score with a regression head.

Development commands from the project root:

    python src/model_cnn.py train \
        --train data/train.json \
        --model_dir models/cnn_regressor

    python src/model_cnn.py predict \
        --input data/dev.json \
        --output results/dev_predictions_cnn.jsonl \
        --model_dir models/cnn_regressor

    python src/model_cnn.py train-and-predict \
        --train data/train.json \
        --input data/dev.json \
        --output results/dev_predictions_cnn.jsonl \
        --model_dir models/cnn_regressor

The official root-level predict.py can import predict_file from this module.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The CNN model requires torch.\n"
        "Install the project dependencies with:\n\n"
        "    python -m pip install -r requirements.txt\n"
    ) from exc


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEFAULT_MODEL_DIR = Path("models/cnn_regressor")
DEFAULT_MAX_LENGTH = 256
DEFAULT_MAX_VOCAB_SIZE = 20000
DEFAULT_MIN_FREQ = 2
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_NUM_FILTERS = 128
DEFAULT_KERNEL_SIZES = (2, 3, 4)
DEFAULT_DROPOUT = 0.3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 8
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
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


def build_cnn_input(sample: Dict[str, Any]) -> str:
    """
    Build the single text sequence used by the CNN.

    The sequence contains both the story context and the candidate meaning.
    Field labels are included to help the model distinguish where information
    comes from.
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


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase word-like tokens.

    This is intentionally simple and fully local. It is enough for a baseline
    CNN without requiring external tokenizers.
    """
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", text.lower())


def build_vocabulary(
    training_data: Dict[str, Dict[str, Any]],
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    min_freq: int = DEFAULT_MIN_FREQ,
) -> Dict[str, int]:
    """
    Build a token-to-id vocabulary from the training data.
    """
    counts: Counter[str] = Counter()

    for sample_id in sorted_sample_ids(training_data):
        sample = training_data[sample_id]
        counts.update(tokenize(build_cnn_input(sample)))

    vocab: Dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, count in counts.most_common():
        if count < min_freq:
            continue
        if token in vocab:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    """
    Convert text into a fixed-length sequence of token IDs.
    """
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]

    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]

    if len(token_ids) < max_length:
        token_ids = token_ids + [vocab[PAD_TOKEN]] * (max_length - len(token_ids))

    return token_ids


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


class AmbiStoryCnnDataset(Dataset):
    """
    PyTorch dataset for CNN regression.

    For training, labels are the human average scores.
    For prediction, labels are omitted.
    """

    def __init__(
        self,
        data: Dict[str, Dict[str, Any]],
        vocab: Dict[str, int],
        max_length: int,
        include_labels: bool,
    ) -> None:
        self.data = data
        self.sample_ids = sorted_sample_ids(data)
        self.vocab = vocab
        self.max_length = max_length
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[index]
        sample = self.data[sample_id]

        input_text = build_cnn_input(sample)
        input_ids = encode_text(input_text, self.vocab, self.max_length)

        item: Dict[str, Any] = {
            "sample_id": sample_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
        }

        if self.include_labels:
            item["labels"] = torch.tensor(
                extract_average(sample_id, sample),
                dtype=torch.float,
            )

        return item


class TextCnnRegressor(nn.Module):
    """
    Text CNN regression model.

    Architecture:
        token ids -> embedding -> parallel 1D convolutions -> max pooling
        -> dropout -> linear regression head
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: Tuple[int, ...],
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)

        pooled_outputs = []

        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            pooled = torch.max(features, dim=2).values
            pooled_outputs.append(pooled)

        combined = torch.cat(pooled_outputs, dim=1)
        combined = self.dropout(combined)
        prediction = self.regression_head(combined).squeeze(-1)

        return prediction


def make_device() -> torch.device:
    """
    Use GPU when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CnnRegressionModel:
    """
    CNN model wrapper.

    It handles the vocabulary, neural network, training, saving, loading, and
    prediction logic.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        max_length: int = DEFAULT_MAX_LENGTH,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_filters: int = DEFAULT_NUM_FILTERS,
        kernel_sizes: Tuple[int, ...] = DEFAULT_KERNEL_SIZES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        self.vocab = vocab
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout

        self.model = TextCnnRegressor(
            vocab_size=len(vocab),
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            padding_idx=vocab[PAD_TOKEN],
        )

    @classmethod
    def from_training_data(
        cls,
        training_data: Dict[str, Dict[str, Any]],
        max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
        min_freq: int = DEFAULT_MIN_FREQ,
        max_length: int = DEFAULT_MAX_LENGTH,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_filters: int = DEFAULT_NUM_FILTERS,
        kernel_sizes: Tuple[int, ...] = DEFAULT_KERNEL_SIZES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> "CnnRegressionModel":
        vocab = build_vocabulary(
            training_data=training_data,
            max_vocab_size=max_vocab_size,
            min_freq=min_freq,
        )

        return cls(
            vocab=vocab,
            max_length=max_length,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

    @classmethod
    def load(cls, model_dir: Path) -> "CnnRegressionModel":
        """
        Load a saved CNN model from disk.
        """
        checkpoint_path = model_dir / "cnn_model.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        instance = cls(
            vocab=checkpoint["vocab"],
            max_length=int(checkpoint["config"]["max_length"]),
            embedding_dim=int(checkpoint["config"]["embedding_dim"]),
            num_filters=int(checkpoint["config"]["num_filters"]),
            kernel_sizes=tuple(checkpoint["config"]["kernel_sizes"]),
            dropout=float(checkpoint["config"]["dropout"]),
        )

        instance.model.load_state_dict(checkpoint["model_state_dict"])
        return instance

    def save(self, model_dir: Path, training_config: Dict[str, Any]) -> None:
        """
        Save the CNN checkpoint.
        """
        model_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "vocab": self.vocab,
            "model_state_dict": self.model.state_dict(),
            "config": {
                "max_length": self.max_length,
                "embedding_dim": self.embedding_dim,
                "num_filters": self.num_filters,
                "kernel_sizes": list(self.kernel_sizes),
                "dropout": self.dropout,
                **training_config,
            },
        }

        torch.save(checkpoint, model_dir / "cnn_model.pt")

    def fit(
        self,
        training_data: Dict[str, Dict[str, Any]],
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> "CnnRegressionModel":
        """
        Train the CNN regression model.
        """
        set_random_seed(seed)
        device = make_device()
        self.model.to(device)
        self.model.train()

        dataset = AmbiStoryCnnDataset(
            data=training_data,
            vocab=self.vocab,
            max_length=self.max_length,
            include_labels=True,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        loss_function = nn.MSELoss()

        print(f"[model_cnn] Device: {device}")
        print(f"[model_cnn] Training samples: {len(dataset)}")
        print(f"[model_cnn] Vocabulary size: {len(self.vocab)}")
        print(f"[model_cnn] Epochs: {epochs}")
        print(f"[model_cnn] Batch size: {batch_size}")

        for epoch_index in range(epochs):
            total_loss = 0.0

            for step_index, batch in enumerate(dataloader, start=1):
                optimizer.zero_grad(set_to_none=True)

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                predictions = self.model(input_ids)
                loss = loss_function(predictions, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item())

                if step_index % 20 == 0 or step_index == len(dataloader):
                    avg_loss = total_loss / step_index
                    print(
                        f"[model_cnn] Epoch {epoch_index + 1}/{epochs}, "
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

        dataset = AmbiStoryCnnDataset(
            data=data,
            vocab=self.vocab,
            max_length=self.max_length,
            include_labels=False,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions: List[Tuple[str, float]] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                scores = self.model(input_ids).detach().cpu().tolist()

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


def model_checkpoint_exists(model_dir: Path) -> bool:
    """
    Check whether a saved CNN checkpoint exists.
    """
    return (model_dir / "cnn_model.pt").exists()


def train_and_save_model(
    train_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    min_freq: int = DEFAULT_MIN_FREQ,
    max_length: int = DEFAULT_MAX_LENGTH,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    num_filters: int = DEFAULT_NUM_FILTERS,
    kernel_sizes: Tuple[int, ...] = DEFAULT_KERNEL_SIZES,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_RANDOM_SEED,
) -> CnnRegressionModel:
    """
    Train the CNN model and save it.
    """
    training_data = load_json_dict(train_path)

    model = CnnRegressionModel.from_training_data(
        training_data=training_data,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        max_length=max_length,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
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
            "max_vocab_size": max_vocab_size,
            "min_freq": min_freq,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "train_path": str(train_path),
        },
    )

    print(f"[model_cnn] Saved trained CNN model to: {model_dir}")
    return model


def predict_with_saved_model(
    input_path: Path,
    output_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Load a saved CNN model and predict for input_path.
    """
    if not model_checkpoint_exists(model_dir):
        raise FileNotFoundError(
            f"No saved CNN checkpoint found in {model_dir}.\n"
            "Train it first with:\n\n"
            "    python src/model_cnn.py train "
            "--train data/train.json --model_dir models/cnn_regressor\n"
        )

    print(f"[model_cnn] Loading trained CNN model from: {model_dir}")
    model = CnnRegressionModel.load(model_dir)

    print(f"[model_cnn] Loading input data from: {input_path}")
    input_data = load_json_dict(input_path)

    predictions = model.predict_many(input_data, batch_size=batch_size)
    write_predictions_jsonl(predictions, output_path)

    print(f"[model_cnn] Wrote {len(predictions)} predictions to: {output_path}")


def train_and_predict_file(
    train_path: Path,
    input_path: Path,
    output_path: Path,
    model_dir: Path = DEFAULT_MODEL_DIR,
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    min_freq: int = DEFAULT_MIN_FREQ,
    max_length: int = DEFAULT_MAX_LENGTH,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    num_filters: int = DEFAULT_NUM_FILTERS,
    kernel_sizes: Tuple[int, ...] = DEFAULT_KERNEL_SIZES,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """
    Train CNN, save it, and predict for input_path.
    """
    model = train_and_save_model(
        train_path=train_path,
        model_dir=model_dir,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        max_length=max_length,
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )

    print(f"[model_cnn] Loading input data from: {input_path}")
    input_data = load_json_dict(input_path)

    predictions = model.predict_many(input_data, batch_size=batch_size)
    write_predictions_jsonl(predictions, output_path)

    print(f"[model_cnn] Wrote {len(predictions)} predictions to: {output_path}")


def predict_file(
    input_path: Path,
    output_path: Path,
    train_path: Path | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    auto_train_if_missing: bool = True,
) -> None:
    """
    Official prediction helper used by root-level predict.py.

    If a saved CNN checkpoint exists, it is loaded and used.

    If no checkpoint exists and auto_train_if_missing=True, the model is trained
    from data/train.json first. This keeps predict.py runnable, but the first run
    can take some time.
    """
    if model_checkpoint_exists(model_dir):
        predict_with_saved_model(
            input_path=input_path,
            output_path=output_path,
            model_dir=model_dir,
        )
        return

    if auto_train_if_missing and train_path is not None and train_path.exists():
        print("[model_cnn] No saved checkpoint found. Training CNN before prediction.")
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


def parse_kernel_sizes(value: str) -> Tuple[int, ...]:
    """
    Parse a comma-separated list such as '2,3,4'.
    """
    try:
        kernel_sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "kernel sizes must be comma-separated integers, for example: 2,3,4"
        ) from exc

    if not kernel_sizes:
        raise argparse.ArgumentTypeError("at least one kernel size is required")

    return kernel_sizes


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments shared by training commands.
    """
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory for the trained CNN checkpoint.",
    )
    parser.add_argument(
        "--max_vocab_size",
        type=int,
        default=DEFAULT_MAX_VOCAB_SIZE,
        help="Maximum vocabulary size. Default: 20000",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=DEFAULT_MIN_FREQ,
        help="Minimum token frequency for vocabulary. Default: 2",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token sequence length. Default: 256",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help="Embedding dimension. Default: 128",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=DEFAULT_NUM_FILTERS,
        help="Number of filters per kernel size. Default: 128",
    )
    parser.add_argument(
        "--kernel_sizes",
        type=parse_kernel_sizes,
        default=DEFAULT_KERNEL_SIZES,
        help="Comma-separated convolution kernel sizes. Default: 2,3,4",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout probability. Default: 0.3",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs. Default: 8",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size. Default: 32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate. Default: 1e-3",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay. Default: 1e-4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed. Default: 13",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and run the CNN regression model."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save CNN.")
    train_parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train.json"),
        help="Path to train.json.",
    )
    add_shared_arguments(train_parser)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict using a saved CNN checkpoint.",
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
        help="Directory for the trained CNN checkpoint.",
    )
    predict_parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Prediction batch size. Default: 32",
    )

    train_predict_parser = subparsers.add_parser(
        "train-and-predict",
        help="Train CNN, save it, and predict.",
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
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_length=args.max_length,
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            kernel_sizes=args.kernel_sizes,
            dropout=args.dropout,
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
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_length=args.max_length,
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            kernel_sizes=args.kernel_sizes,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
