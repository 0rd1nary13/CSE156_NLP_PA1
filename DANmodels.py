"""Deep Averaging Network (DAN) models and training utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sentiment_data import SentimentExample, WordEmbeddings, read_sentiment_examples, read_word_embeddings

BEST_EMBEDDINGS_PATH = "data/glove.6B.300d-relativized.txt"
BEST_HIDDEN_SIZES: Tuple[int, ...] = (100,)
BEST_DROPOUT = 0.2
BEST_LEARNING_RATE = 5e-4
BEST_FREEZE_EMBEDDINGS = False
BEST_EPOCHS = 10


@dataclass(frozen=True)
class DANHyperparams:
    """Container for DAN training hyperparameters."""

    embeddings_path: str
    hidden_sizes: Tuple[int, ...]
    dropout: float
    learning_rate: float
    freeze_embeddings: bool
    epochs: int
    batch_size: int
    log_every: int
    patience: int
    random_init: bool
    embedding_dim: int | None


class SentimentDatasetDAN(Dataset):
    """Dataset that maps tokenized sentences to word indices for DAN."""

    def __init__(self, infile: str, word_embeddings: WordEmbeddings) -> None:
        """Initialize the dataset from a sentiment file.

        Args:
            infile: Path to the sentiment data file.
            word_embeddings: Preloaded word embeddings with an indexer.
        """
        self.examples: List[SentimentExample] = read_sentiment_examples(infile)
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.unk_index = self.word_indexer.index_of("UNK")
        if self.unk_index == -1:
            raise ValueError("UNK token is missing from the word indexer.")
        self.pad_index = self.word_indexer.index_of("PAD")
        if self.pad_index == -1:
            raise ValueError("PAD token is missing from the word indexer.")

        self.sentences: List[List[int]] = [
            self._words_to_indices(ex.words) for ex in self.examples
        ]
        self.labels: List[int] = [ex.label for ex in self.examples]

    def _words_to_indices(self, words: List[str]) -> List[int]:
        """Convert a list of tokens into embedding indices.

        Args:
            words: Tokenized sentence.

        Returns:
            List of indices for the embedding lookup.
        """
        if not words:
            return [self.pad_index]
        indices: List[int] = []
        for word in words:
            word_index = self.word_indexer.index_of(word)
            if word_index == -1:
                word_index = self.unk_index
            indices.append(word_index)
        return indices

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        """Return token indices and label for a single example."""
        return self.sentences[idx], self.labels[idx]


def collate_dan_batch(
    batch: Sequence[Tuple[List[int], int]],
    pad_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sentences and build tensors for DAN training.
    
    Note: This function was generated with AI assistance.

    Args:
        batch: Sequence of (token_indices, label) pairs.
        pad_index: Index used to pad sequences.

    Returns:
        Tuple of (padded_indices, lengths, labels) tensors.
    """
    sequences, labels = zip(*batch)
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    padded = torch.full((batch_size, max_len), pad_index, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        lengths[i] = seq_len

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels_tensor


class DAN(nn.Module):
    """Deep Averaging Network for sentence classification."""

    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_sizes: Sequence[int] | None = None,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        """Initialize the DAN model.

        Args:
            embedding_layer: Embedding layer for word indices.
            hidden_sizes: Sizes of hidden layers.
            dropout: Dropout probability applied after hidden layers.
            num_classes: Number of output classes.
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = (100,)

        self.embedding = embedding_layer
        embedding_dim = embedding_layer.embedding_dim
        layers: List[nn.Module] = []
        input_dim = embedding_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.feedforward = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of sentences.

        Args:
            input_ids: Tensor of token indices (batch_size, seq_len).
            lengths: Tensor of sequence lengths (batch_size,).

        Returns:
            Logits for each class (batch_size, num_classes).
        """
        embeddings = self.embedding(input_ids)
        max_len = embeddings.size(1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).type_as(embeddings)
        summed = (embeddings * mask).sum(dim=1)
        lengths = lengths.clamp(min=1).unsqueeze(1).type_as(summed)
        averaged = summed / lengths
        hidden = self.feedforward(averaged)
        return self.output(hidden)


def get_best_dan_hyperparams() -> Dict[str, object]:
    """Return the best hyperparameters found in grid search.

    Returns:
        Dictionary of hyperparameters for the strongest dev accuracy.
    """
    return {
        "embeddings_path": BEST_EMBEDDINGS_PATH,
        "hidden_sizes": BEST_HIDDEN_SIZES,
        "dropout": BEST_DROPOUT,
        "learning_rate": BEST_LEARNING_RATE,
        "freeze_embeddings": BEST_FREEZE_EMBEDDINGS,
        "epochs": BEST_EPOCHS,
    }


def build_embedding_layer(
    embeddings: WordEmbeddings,
    embedding_dim: int | None,
    freeze_embeddings: bool,
    random_init: bool,
) -> nn.Embedding:
    """Create an embedding layer from pretrained vectors or random init.
    
    Note: This function was generated with AI assistance.

    Args:
        embeddings: Preloaded word embeddings (for vocab and vectors).
        embedding_dim: Optional embedding size for random init.
        freeze_embeddings: Whether to freeze embeddings for pretrained init.
        random_init: If True, ignore pretrained vectors and use random init.

    Returns:
        Embedding layer for the model.
    """
    if random_init:
        if embedding_dim is None:
            embedding_dim = embeddings.get_embedding_length()
        return nn.Embedding(num_embeddings=len(embeddings.word_indexer), embedding_dim=embedding_dim)
    return embeddings.get_initialized_embedding_layer(frozen=freeze_embeddings)


def build_dan_loaders(
    embeddings: WordEmbeddings,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create DAN data loaders for train and dev splits.
    
    Note: This function was generated with AI assistance.

    Args:
        embeddings: Preloaded word embeddings.
        batch_size: Batch size for data loaders.

    Returns:
        Tuple of (train_loader, dev_loader, pad_index).
    """
    train_data = SentimentDatasetDAN("data/train.txt", embeddings)
    dev_data = SentimentDatasetDAN("data/dev.txt", embeddings)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_dan_batch(batch, train_data.pad_index),
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_dan_batch(batch, dev_data.pad_index),
    )
    return train_loader, dev_loader, train_data.pad_index


def train_epoch_dan(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one epoch of training for the DAN model.

    Args:
        data_loader: Training data loader.
        model: Model to train.
        loss_fn: Loss function.
        optimizer: Optimizer for updating parameters.
        device: Torch device to use.

    Returns:
        Tuple of (accuracy, average_loss).
    """
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0.0, 0.0
    for input_ids, lengths, labels in data_loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        logits = model(input_ids, lengths)
        loss = loss_fn(logits, labels)
        train_loss += loss.item()
        correct += (logits.argmax(1) == labels).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


def eval_epoch_dan(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the DAN model for one epoch.

    Args:
        data_loader: Evaluation data loader.
        model: Model to evaluate.
        loss_fn: Loss function.
        device: Torch device to use.

    Returns:
        Tuple of (accuracy, average_loss).
    """
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss, correct = 0.0, 0.0
    for input_ids, lengths, labels in data_loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        logits = model(input_ids, lengths)
        loss = loss_fn(logits, labels)
        eval_loss += loss.item()
        correct += (logits.argmax(1) == labels).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


def train_dan(
    embeddings: WordEmbeddings,
    config: DANHyperparams,
    device: torch.device,
) -> Tuple[float, int]:
    """Train a DAN model and return the best dev accuracy.

    Args:
        embeddings: Preloaded word embeddings.
        config: Hyperparameter configuration.
        device: Torch device to use.

    Returns:
        Tuple of (best_dev_accuracy, best_epoch).
    """
    train_loader, dev_loader, _ = build_dan_loaders(embeddings, config.batch_size)
    embedding_layer = build_embedding_layer(
        embeddings=embeddings,
        embedding_dim=config.embedding_dim,
        freeze_embeddings=config.freeze_embeddings,
        random_init=config.random_init,
    )
    model = DAN(
        embedding_layer=embedding_layer,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_dev_accuracy = 0.0
    best_epoch = 0
    epochs_since_improve = 0
    for epoch in range(config.epochs):
        train_accuracy, train_loss = train_epoch_dan(
            train_loader, model, loss_fn, optimizer, device
        )
        dev_accuracy, dev_loss = eval_epoch_dan(dev_loader, model, loss_fn, device)
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_epoch = epoch + 1
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % config.log_every == config.log_every - 1:
            print(
                f"Epoch #{epoch + 1}: "
                f"train loss {train_loss:.4f}, dev loss {dev_loss:.4f}, "
                f"train acc {train_accuracy:.3f}, dev acc {dev_accuracy:.3f}"
            )
        if epochs_since_improve >= config.patience:
            break

    return best_dev_accuracy, best_epoch


def parse_hidden_sizes(raw: Iterable[str]) -> Tuple[int, ...]:
    """Parse hidden layer sizes from CLI tokens.
    
    Note: This function was generated with AI assistance.

    Args:
        raw: Iterable of string tokens.

    Returns:
        Tuple of hidden layer sizes.
    """
    return tuple(int(item) for item in raw)


def build_best_config(random_init: bool) -> DANHyperparams:
    """Create the best-known DAN config for either init style.
    
    Note: This function was generated with AI assistance.

    Args:
        random_init: If True, use random embedding init.

    Returns:
        Best-known configuration for the requested init.
    """
    return DANHyperparams(
        embeddings_path=BEST_EMBEDDINGS_PATH,
        hidden_sizes=BEST_HIDDEN_SIZES,
        dropout=BEST_DROPOUT,
        learning_rate=BEST_LEARNING_RATE,
        freeze_embeddings=BEST_FREEZE_EMBEDDINGS,
        epochs=BEST_EPOCHS,
        batch_size=32,
        log_every=5,
        patience=6,
        random_init=random_init,
        embedding_dim=None,
    )


def main() -> None:
    """Train a DAN model from command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Deep Averaging Network model.")
    parser.add_argument(
        "--embeddings",
        type=str,
        default=BEST_EMBEDDINGS_PATH,
        help="Path to pretrained embeddings file.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        nargs="+",
        default=[str(size) for size in BEST_HIDDEN_SIZES],
        help="Hidden layer sizes, e.g. --hidden-sizes 100 50",
    )
    parser.add_argument("--dropout", type=float, default=BEST_DROPOUT, help="Dropout probability.")
    parser.add_argument("--lr", type=float, default=BEST_LEARNING_RATE, help="Learning rate.")
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="Freeze embeddings (default is fine-tuning).",
    )
    parser.add_argument("--epochs", type=int, default=BEST_EPOCHS, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--log-every", type=int, default=5, help="Log interval.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use random embeddings instead of GloVe initialization.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding size when using random init (defaults to GloVe dimension).",
    )
    args = parser.parse_args()

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    embeddings = read_word_embeddings(args.embeddings)
    config = DANHyperparams(
        embeddings_path=args.embeddings,
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
        learning_rate=args.lr,
        freeze_embeddings=args.freeze_embeddings,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=args.log_every,
        patience=args.patience,
        random_init=args.random_init,
        embedding_dim=args.embedding_dim,
    )
    device = torch.device("cpu")
    best_dev, best_epoch = train_dan(embeddings, config, device)
    print(
        f"Best dev accuracy: {best_dev:.3f} at epoch {best_epoch} "
        f"(random_init={config.random_init})"
    )


if __name__ == "__main__":
    main()
