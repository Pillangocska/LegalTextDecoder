"""
HuBERT Fine-tuning Script for Hungarian Legal Text Readability Classification

This script fine-tunes the SZTAKI-HLT/hubert-base-cc model for predicting
readability scores (1-5) of Hungarian ÁSZF (Terms and Conditions) texts.

The script is organized into three main components:
    1. DataPreprocessor - Handles data loading, tokenization, and dataset creation
    2. ModelTrainer - Handles model initialization and training loop
    3. ModelEvaluator - Handles evaluation metrics and visualization

Input:
    - _data/final/train.csv
    - _data/final/test.csv

Output:
    - models/best_model.pt
    - logs/training_log_<timestamp>.txt
    - models/training_history.png
    - models/confusion_matrix_*.png
"""

from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import datetime
import warnings
import torch

from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
)

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model settings
    model_name: str = 'SZTAKI-HLT/hubert-base-cc'
    num_classes: int = 5
    max_length: int = 256

    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    val_split: float = 0.2

    # Paths (stored as strings internally, converted via properties)
    _train_path: str = '_data/final/train.csv'
    _test_path: str = '_data/final/test.csv'
    _model_dir: str = 'models'
    _log_dir: str = 'logs'

    # Random seed
    random_seed: int = 42

    @property
    def train_path(self) -> Path:
        """Get train_path as Path object."""
        return Path(self._train_path)

    @train_path.setter
    def train_path(self, value: Any) -> None:
        """Set train_path from string or Path."""
        self._train_path = str(value)

    @property
    def test_path(self) -> Path:
        """Get test_path as Path object."""
        return Path(self._test_path)

    @test_path.setter
    def test_path(self, value: Any) -> None:
        """Set test_path from string or Path."""
        self._test_path = str(value)

    @property
    def model_dir(self) -> Path:
        """Get model_dir as Path object."""
        return Path(self._model_dir)

    @model_dir.setter
    def model_dir(self, value: Any) -> None:
        """Set model_dir from string or Path."""
        self._model_dir = str(value)

    @property
    def log_dir(self) -> Path:
        """Get log_dir as Path object."""
        return Path(self._log_dir)

    @log_dir.setter
    def log_dir(self, value: Any) -> None:
        """Set log_dir from string or Path."""
        self._log_dir = str(value)


@dataclass
class TrainingHistory:
    """Container for training history metrics."""

    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_accuracy: float = 0.0


# =============================================================================
# DATASET CLASS
# =============================================================================

class ASZFDataset(Dataset):
    """PyTorch Dataset for ÁSZF readability classification."""

    def __init__(self, encodings: Dict[str, List[Any]]):
        """
        Initialize the dataset.

        Args:
            encodings: Dictionary containing input_ids, attention_mask, and labels
        """
        self.encodings: Dict[str, List[Any]] = encodings

    def __len__(self) -> int:
        return len(self.encodings['labels'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.encodings['labels'][idx] - 1, dtype=torch.long),  # Convert 1-5 to 0-4
        }


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================

class DataPreprocessor:
    """
    Handles data loading, tokenization, and dataset creation.

    This class is responsible for:
        - Loading CSV data
        - Tokenizing texts using the HuBERT tokenizer
        - Creating train/validation/test splits
        - Computing class weights for imbalanced data
        - Creating PyTorch DataLoaders
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the preprocessor.

        Args:
            config: Training configuration
        """
        self.config: TrainingConfig = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.class_weights: Optional[Dict[int, float]] = None

        # Data containers
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None

        # Encoded datasets
        self.train_encodings: Optional[Dict[str, List[Any]]] = None
        self.val_encodings: Optional[Dict[str, List[Any]]] = None
        self.test_encodings: Optional[Dict[str, List[Any]]] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from CSV files.

        Returns:
            Tuple of (train_df, test_df)
        """
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)

        self.df_train = pd.read_csv(self.config.train_path)
        self.df_test = pd.read_csv(self.config.test_path)

        print(f"\nTraining data: {self.df_train.shape}")
        print(f"Test data: {self.df_test.shape}")

        return self.df_train, self.df_test

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer for the specified model.

        Returns:
            Loaded tokenizer
        """
        print(f"\nLoading tokenizer: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        print(f"✓ Tokenizer loaded (vocab size: {self.tokenizer.vocab_size})")

        return self.tokenizer

    def tokenize_texts(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, List[Any]]:
        """
        Tokenize texts and prepare encodings.

        Args:
            texts: Array of text strings
            labels: Array of label values

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        encodings = self.tokenizer(
            texts.tolist(),
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=None,
        )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels.tolist(),
        }

    def compute_class_weights(self) -> Dict[int, float]:
        """
        Compute class weights for handling class imbalance.

        Returns:
            Dictionary mapping class labels to weights
        """
        unique_labels: np.ndarray = np.unique(self.df_train['label_numeric'])

        weights_array: np.ndarray = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=self.df_train['label_numeric'],
        )

        self.class_weights = dict(zip(unique_labels, weights_array))

        print("\nClass weights:")
        for cls, weight in self.class_weights.items():
            print(f"  Class {cls}: {weight:.4f}")

        return self.class_weights

    def create_splits(self) -> Tuple[Dict, Dict, Dict]:
        """
        Create train/validation/test splits with tokenized data.

        Returns:
            Tuple of (train_encodings, val_encodings, test_encodings)
        """
        print("\n" + "=" * 60)
        print("CREATING DATA SPLITS")
        print("=" * 60)

        # Tokenize all training data first
        print("\nTokenizing training data...")
        full_train_encodings: Dict[str, List[Any]] = self.tokenize_texts(
            self.df_train['text'].values,
            self.df_train['label_numeric'].values,
        )
        print(f"✓ Training data tokenized: {len(full_train_encodings['input_ids'])} samples")

        # Tokenize test data
        print("Tokenizing test data...")
        self.test_encodings = self.tokenize_texts(
            self.df_test['text'].values,
            self.df_test['label_numeric'].values,
        )
        print(f"✓ Test data tokenized: {len(self.test_encodings['input_ids'])} samples")

        # Create train/validation split
        train_indices, val_indices = train_test_split(
            range(len(full_train_encodings['input_ids'])),
            test_size=self.config.val_split,
            random_state=self.config.random_seed,
            stratify=full_train_encodings['labels'],
        )

        # Split encodings
        self.train_encodings = {
            'input_ids': [full_train_encodings['input_ids'][i] for i in train_indices],
            'attention_mask': [full_train_encodings['attention_mask'][i] for i in train_indices],
            'labels': [full_train_encodings['labels'][i] for i in train_indices],
        }

        self.val_encodings = {
            'input_ids': [full_train_encodings['input_ids'][i] for i in val_indices],
            'attention_mask': [full_train_encodings['attention_mask'][i] for i in val_indices],
            'labels': [full_train_encodings['labels'][i] for i in val_indices],
        }

        print(f"\nSplit sizes:")
        print(f"  Training:   {len(self.train_encodings['input_ids'])} samples")
        print(f"  Validation: {len(self.val_encodings['input_ids'])} samples")
        print(f"  Test:       {len(self.test_encodings['input_ids'])} samples")

        # Print label distribution
        self._print_label_distribution()

        return self.train_encodings, self.val_encodings, self.test_encodings

    def _print_label_distribution(self) -> None:
        """Print label distribution in training set."""
        print("\nLabel distribution in training:")

        train_labels: pd.Series = pd.Series(self.train_encodings['labels'])
        label_dist: pd.Series = train_labels.value_counts().sort_index()

        for label, count in label_dist.items():
            pct: float = (count / len(train_labels)) * 100
            print(f"  Class {label}: {count:4d} samples ({pct:5.2f}%)")

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train, validation, and test sets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        print("\n" + "=" * 60)
        print("CREATING DATALOADERS")
        print("=" * 60)

        # Create datasets
        train_dataset: ASZFDataset = ASZFDataset(self.train_encodings)
        val_dataset: ASZFDataset = ASZFDataset(self.val_encodings)
        test_dataset: ASZFDataset = ASZFDataset(self.test_encodings)

        print(f"✓ Created PyTorch datasets")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val:   {len(val_dataset)} samples")
        print(f"  Test:  {len(test_dataset)} samples")

        # Create dataloaders
        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        test_loader: DataLoader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        print(f"\n✓ Created DataLoaders")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")

        # Verify sample batch
        sample_batch: Dict[str, torch.Tensor] = next(iter(train_loader))
        print(f"\nSample batch verification:")
        print(f"  input_ids shape: {sample_batch['input_ids'].shape}")
        print(f"  attention_mask shape: {sample_batch['attention_mask'].shape}")
        print(f"  labels shape: {sample_batch['labels'].shape}")
        print(f"  labels range: {sample_batch['labels'].min().item()} to {sample_batch['labels'].max().item()}")

        return train_loader, val_loader, test_loader

    def preprocess(self) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, float]]:
        """
        Run the full preprocessing pipeline.

        Returns:
            Tuple of (train_loader, val_loader, test_loader, class_weights)
        """
        self.load_data()
        self.load_tokenizer()
        self.compute_class_weights()
        self.create_splits()
        train_loader, val_loader, test_loader = self.create_dataloaders()

        return train_loader, val_loader, test_loader, self.class_weights


# =============================================================================
# MODEL TRAINER
# =============================================================================

class ModelTrainer:
    """
    Handles model initialization and training loop.

    This class is responsible for:
        - Loading and configuring the model
        - Setting up optimizer and scheduler
        - Running the training loop
        - Saving checkpoints
        - Logging training progress
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Dict[int, float],
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            class_weights: Dictionary of class weights
        """
        self.config: TrainingConfig = config
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.class_weights: Dict[int, float] = class_weights

        # Device setup
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model components (initialized in setup)
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[Any] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None

        # Training state
        self.history: TrainingHistory = TrainingHistory()
        self.log_file: Optional[Path] = None

        # Set random seeds
        self._set_random_seeds()

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def setup(self) -> None:
        """Set up model, optimizer, scheduler, and logging."""
        print("\n" + "=" * 60)
        print("INITIALIZING TRAINING COMPONENTS")
        print("=" * 60)

        print(f"\nDevice: {self.device}")
        print(f"PyTorch version: {torch.__version__}")

        self._load_model()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()

        print(f"\n✓ All training components initialized!")

    def _load_model(self) -> None:
        """Load and configure the model."""
        print(f"\nLoading model: {self.config.model_name}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            problem_type="single_label_classification",
        )
        self.model.to(self.device)

        print(f"✓ Model loaded and moved to {self.device}")

        # Count parameters
        total_params: int = sum(p.numel() for p in self.model.parameters())
        trainable_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    def _setup_criterion(self) -> None:
        """Set up loss function with class weights."""
        class_weights_list: List[float] = [self.class_weights[i] for i in range(1, 6)]
        class_weights_tensor: torch.Tensor = torch.tensor(
            class_weights_list,
            dtype=torch.float,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        print(f"\n✓ Class weights prepared: {[f'{w:.3f}' for w in class_weights_list]}")
        print(f"✓ Loss function: CrossEntropyLoss with class weights")

    def _setup_optimizer(self) -> None:
        """Set up the optimizer."""
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        print(f"✓ Optimizer: AdamW (lr={self.config.learning_rate})")

    def _setup_scheduler(self) -> None:
        """Set up learning rate scheduler."""
        total_steps: int = len(self.train_loader) * self.config.num_epochs
        warmup_steps: int = int(self.config.warmup_ratio * total_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"✓ Scheduler: Linear with warmup")
        print(f"  Total training steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")

    def _setup_logging(self) -> None:
        """Set up logging directory and file."""
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

        timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.config.log_dir / f'training_log_{timestamp}.txt'

        print(f"\n✓ Log file: {self.log_file}")

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.config.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Batch size: {self.config.batch_size}\n")
            f.write(f"Learning rate: {self.config.learning_rate}\n")
            f.write(f"Epochs: {self.config.num_epochs}\n")
            f.write("=" * 60 + "\n\n")

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch in progress_bar:
            # Move batch to device
            input_ids: torch.Tensor = batch['input_ids'].to(self.device)
            attention_mask: torch.Tensor = batch['attention_mask'].to(self.device)
            labels: torch.Tensor = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits: torch.Tensor = outputs.logits

            # Calculate loss
            loss: torch.Tensor = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Calculate accuracy
            predictions: torch.Tensor = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Accumulate loss
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}',
            })

        avg_loss: float = total_loss / len(self.train_loader)
        accuracy: float = correct / total

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}\n")

        return avg_loss, accuracy

    def _validate(self, epoch: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, accuracy, predictions, labels)
        """
        self.model.eval()

        total_loss: float = 0.0
        correct: int = 0
        total: int = 0
        all_predictions: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Validating'):
                # Move batch to device
                input_ids: torch.Tensor = batch['input_ids'].to(self.device)
                attention_mask: torch.Tensor = batch['attention_mask'].to(self.device)
                labels: torch.Tensor = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits: torch.Tensor = outputs.logits

                # Calculate loss
                loss: torch.Tensor = self.criterion(logits, labels)

                # Calculate accuracy
                predictions: torch.Tensor = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Accumulate loss
                total_loss += loss.item()

                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss: float = total_loss / len(self.val_loader)
        accuracy: float = correct / total

        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(f"Validation (Epoch {epoch}) - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}\n")

        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)

    def _save_checkpoint(self, epoch: int, val_accuracy: float, val_loss: float) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_accuracy: Validation accuracy
            val_loss: Validation loss
        """
        checkpoint_path: Path = self.config.model_dir / 'best_model.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
        }, checkpoint_path)

        print(f"✓ New best model saved! (Val Acc: {val_accuracy:.4f})")

        with open(self.log_file, 'a') as f:
            f.write(f"*** Best model saved at epoch {epoch} (Val Acc: {val_accuracy:.4f}) ***\n\n")

    def train(self) -> TrainingHistory:
        """
        Run the full training loop.

        Returns:
            TrainingHistory with metrics from all epochs
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        print(f"\nTraining for {self.config.num_epochs} epochs...")

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch}/{self.config.num_epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss, train_acc = self._train_epoch(epoch)
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validate
            val_loss, val_acc, val_preds, val_labels = self._validate(epoch)
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            # Per-class recall
            print("  Per-class recall:", end=" ")
            for c in range(5):
                mask: np.ndarray = val_labels == c
                if mask.sum() > 0:
                    recall: float = (val_preds[mask] == c).sum() / mask.sum()
                    print(f"C{c+1}:{recall:.2f}", end=" ")
            print()

            # Save history
            self.history.train_loss.append(train_loss)
            self.history.train_acc.append(train_acc)
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)

            # Save best model
            if val_acc > self.history.best_val_accuracy:
                self.history.best_val_accuracy = val_acc
                self.history.best_epoch = epoch
                self._save_checkpoint(epoch, val_acc, val_loss)

        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE!")
        print(f"{'=' * 60}")
        print(f"\nBest validation accuracy: {self.history.best_val_accuracy:.4f} (Epoch {self.history.best_epoch})")

        # Log final results
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("TRAINING COMPLETE\n")
            f.write("=" * 60 + "\n")
            f.write(f"Best validation accuracy: {self.history.best_val_accuracy:.4f} (Epoch {self.history.best_epoch})\n")

        return self.history

    def load_best_model(self) -> None:
        """Load the best saved model checkpoint."""
        checkpoint_path: Path = self.config.model_dir / 'best_model.pt'

        print(f"\nLoading best model from {checkpoint_path}...")

        checkpoint: Dict[str, Any] = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:
    """
    Handles model evaluation and visualization.

    This class is responsible for:
        - Evaluating model on validation/test sets
        - Computing classification metrics
        - Computing ordinal evaluation metrics
        - Generating visualizations (confusion matrices, training curves)
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        criterion: nn.CrossEntropyLoss,
        device: torch.device,
        config: TrainingConfig,
    ):
        """
        Initialize the evaluator.

        Args:
            model: The trained model
            criterion: Loss function
            device: Device to run evaluation on
            config: Training configuration
        """
        self.model: AutoModelForSequenceClassification = model
        self.criterion: nn.CrossEntropyLoss = criterion
        self.device: torch.device = device
        self.config: TrainingConfig = config

    def evaluate(
        self,
        dataloader: DataLoader,
        split_name: str,
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for the dataset
            split_name: Name of the split (for logging)

        Returns:
            Tuple of (average_loss, accuracy, predictions, labels)
        """
        self.model.eval()

        total_loss: float = 0.0
        correct: int = 0
        total: int = 0
        all_predictions: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Evaluating {split_name}'):
                input_ids: torch.Tensor = batch['input_ids'].to(self.device)
                attention_mask: torch.Tensor = batch['attention_mask'].to(self.device)
                labels: torch.Tensor = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits: torch.Tensor = outputs.logits

                loss: torch.Tensor = self.criterion(logits, labels)

                predictions: torch.Tensor = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss: float = total_loss / len(dataloader)
        accuracy: float = correct / total

        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)

    @staticmethod
    def within_k_accuracy(true: np.ndarray, pred: np.ndarray, k: int = 1) -> float:
        """
        Calculate accuracy where prediction within k classes of true is correct.

        Args:
            true: True labels
            pred: Predicted labels
            k: Tolerance (default 1)

        Returns:
            Within-k accuracy
        """
        return float((np.abs(true - pred) <= k).mean())

    def print_classification_report(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        split_name: str,
    ) -> None:
        """
        Print classification report for a dataset.

        Args:
            labels: True labels
            predictions: Predicted labels
            split_name: Name of the split
        """
        print(f"\nClassification Report ({split_name}):")
        print(classification_report(
            labels,
            predictions,
            target_names=[f'Class {i}' for i in range(1, 6)],
            digits=4,
        ))

    def print_ordinal_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        split_name: str,
    ) -> None:
        """
        Print ordinal evaluation metrics.

        Args:
            labels: True labels
            predictions: Predicted labels
            split_name: Name of the split
        """
        print(f"\n{split_name}:")
        print(f"  Exact Accuracy:     {(predictions == labels).mean():.4f}")
        print(f"  Within-1 Accuracy:  {self.within_k_accuracy(labels, predictions, 1):.4f}")
        print(f"  Within-2 Accuracy:  {self.within_k_accuracy(labels, predictions, 2):.4f}")

    def print_per_class_recall(
        self,
        val_labels: np.ndarray,
        val_preds: np.ndarray,
        test_labels: np.ndarray,
        test_preds: np.ndarray,
    ) -> None:
        """
        Print per-class recall comparison between validation and test sets.

        Args:
            val_labels: Validation true labels
            val_preds: Validation predictions
            test_labels: Test true labels
            test_preds: Test predictions
        """
        print("\n" + "=" * 60)
        print("PER-CLASS RECALL ANALYSIS")
        print("=" * 60)

        print("\nRecall by class:")
        print("-" * 40)
        print(f"{'Class':<10} {'Validation':<15} {'Test':<15}")
        print("-" * 40)

        for c in range(5):
            val_mask: np.ndarray = val_labels == c
            test_mask: np.ndarray = test_labels == c

            val_recall: float = (val_preds[val_mask] == c).sum() / val_mask.sum() if val_mask.sum() > 0 else 0
            test_recall: float = (test_preds[test_mask] == c).sum() / test_mask.sum() if test_mask.sum() > 0 else 0

            print(f"Class {c+1:<4} {val_recall:<15.4f} {test_recall:<15.4f}")

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        split_name: str,
        cmap: str = 'Blues',
    ) -> None:
        """
        Plot and save confusion matrix.

        Args:
            labels: True labels
            predictions: Predicted labels
            split_name: Name of the split
            cmap: Color map for the heatmap
        """
        # Import here to avoid issues if matplotlib is not available
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")

        cm: np.ndarray = confusion_matrix(labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=cmap,
            xticklabels=[f'Pred {i}' for i in range(1, 6)],
            yticklabels=[f'True {i}' for i in range(1, 6)],
        )
        plt.title(f'Confusion Matrix - {split_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        save_path: Path = self.config.model_dir / f'confusion_matrix_{split_name.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=150)
        plt.show()

        print(f"✓ Confusion matrix saved to {save_path}")

    def plot_training_history(self, history: TrainingHistory) -> None:
        """
        Plot and save training history curves.

        Args:
            history: TrainingHistory object with metrics
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs: range = range(1, len(history.train_loss) + 1)

        # Loss plot
        axes[0].plot(epochs, history.train_loss, marker='o', label='Train Loss')
        axes[0].plot(epochs, history.val_loss, marker='s', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot(epochs, history.train_acc, marker='o', label='Train Acc')
        axes[1].plot(epochs, history.val_acc, marker='s', label='Val Acc')
        axes[1].axhline(y=0.2, color='r', linestyle='--', label='Random Baseline (20%)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path: Path = self.config.model_dir / 'training_history.png'
        plt.savefig(save_path, dpi=150)
        plt.show()

        print(f"✓ Training history saved to {save_path}")

    def full_evaluation(
        self,
        val_loader: DataLoader,
        test_loader: DataLoader,
        history: TrainingHistory,
    ) -> None:
        """
        Run full evaluation on validation and test sets.

        Args:
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            history: Training history for plotting
        """
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        # Validation evaluation
        print("\n" + "-" * 60)
        print("VALIDATION SET RESULTS")
        print("-" * 60)

        val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader, "Validation")
        self.print_classification_report(val_labels, val_preds, "Validation")
        self.plot_confusion_matrix(val_labels, val_preds, "Validation", cmap='Blues')

        # Test evaluation
        print("\n" + "-" * 60)
        print("TEST SET RESULTS")
        print("-" * 60)

        test_loss, test_acc, test_preds, test_labels = self.evaluate(test_loader, "Test")
        self.print_classification_report(test_labels, test_preds, "Test")
        self.plot_confusion_matrix(test_labels, test_preds, "Test", cmap='Greens')

        # Ordinal metrics
        print("\n" + "=" * 60)
        print("ORDINAL EVALUATION METRICS")
        print("=" * 60)

        self.print_ordinal_metrics(val_labels, val_preds, "Validation Set")
        self.print_ordinal_metrics(test_labels, test_preds, "Test Set")

        # Per-class recall
        self.print_per_class_recall(val_labels, val_preds, test_labels, test_preds)

        # Training history plot
        print("\n" + "=" * 60)
        print("TRAINING HISTORY")
        print("=" * 60)
        self.plot_training_history(history)

        # Final summary
        self._print_summary(history, val_labels, val_preds, test_labels, test_preds)

    def _print_summary(
        self,
        history: TrainingHistory,
        val_labels: np.ndarray,
        val_preds: np.ndarray,
        test_labels: np.ndarray,
        test_preds: np.ndarray,
    ) -> None:
        """Print final training summary."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        print(f"""
Model: {self.config.model_name}
Configuration:
  - Batch size: {self.config.batch_size}
  - Learning rate: {self.config.learning_rate}
  - Epochs: {self.config.num_epochs}
  - Max length: {self.config.max_length}
  - Class weights: Yes

Results:
  Best epoch: {history.best_epoch}

  Validation:
    - Accuracy: {(val_preds == val_labels).mean():.4f}
    - Within-1: {self.within_k_accuracy(val_labels, val_preds, 1):.4f}
    - Within-2: {self.within_k_accuracy(val_labels, val_preds, 2):.4f}

  Test:
    - Accuracy: {(test_preds == test_labels).mean():.4f}
    - Within-1: {self.within_k_accuracy(test_labels, test_preds, 1):.4f}
    - Within-2: {self.within_k_accuracy(test_labels, test_preds, 2):.4f}

Files saved:
  - Model: {self.config.model_dir / 'best_model.pt'}
  - Plots: {self.config.model_dir}
""")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """Main entry point for the training script."""
    print("=" * 60)
    print("HUNGARIAN LEGAL TEXT READABILITY CLASSIFIER")
    print("HuBERT Fine-tuning Pipeline")
    print("=" * 60)

    # Initialize configuration
    config: TrainingConfig = TrainingConfig()

    config.train_path = '/content/sample_data/train.csv'
    config.test_path = '/content/sample_data/test.csv'
    config.model_dir = '/content/sample_data/model'
    config.log_dir = '/content/sample_data/log'

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max length: {config.max_length}")
    print(f"  Train path: {config.train_path}")
    print(f"  Test path: {config.test_path}")

    # ==========================================================================
    # PREPROCESSING
    # ==========================================================================
    preprocessor: DataPreprocessor = DataPreprocessor(config)
    train_loader, val_loader, test_loader, class_weights = preprocessor.preprocess()

    # ==========================================================================
    # TRAINING
    # ==========================================================================
    trainer: ModelTrainer = ModelTrainer(config, train_loader, val_loader, class_weights)
    trainer.setup()
    history: TrainingHistory = trainer.train()

    # ==========================================================================
    # EVALUATION (Training History Only)
    # ==========================================================================
    trainer.load_best_model()

    evaluator: ModelEvaluator = ModelEvaluator(
        model=trainer.model,
        criterion=trainer.criterion,
        device=trainer.device,
        config=config,
    )

    # Plot training history
    print("\n" + "=" * 60)
    print("TRAINING HISTORY")
    print("=" * 60)
    evaluator.plot_training_history(history)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model saved to: {config.model_dir / 'best_model.pt'}")
    print(f"To run full evaluation, use: python 02b_evaluate.py")


if __name__ == '__main__':
    main()
