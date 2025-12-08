"""
HuBERT Model Evaluation Script

This script evaluates a trained HuBERT model on validation and test sets,
computing classification metrics, ordinal metrics, and generating visualizations.

This script should be run after 02_train.py has completed training.

Input:
    - _data/final/train.csv (for validation split)
    - _data/final/test.csv
    - output/best_model.pt

Output:
    - output/confusion_matrix_validation.png
    - output/confusion_matrix_test.png
    - Console output with classification reports and metrics
"""

from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.util.config_manager import config
from src.util.logger import Logger

warnings.filterwarnings('ignore')

logger = Logger("evaluation")

@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    # Model settings
    model_name: str = config.get("train.model_name")
    num_classes: int = config.get("train.num_classes")
    max_length: int = config.get("train.max_token_length")

    # Data settings
    batch_size: int = config.get("train.batch_size")
    val_split: float = config.get("train.val_split")
    random_seed: int = config.get("train.random_seed")

    # Paths (stored as strings internally, converted via properties)
    _train_path: str = config.get("train.train_path")
    _test_path: str = config.get("train.test_path")
    _model_dir: str = config.get("train.model_dir")
    _model_path: str = config.get("train.model_path")

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
    def model_path(self) -> Path:
        """Get model_path as Path object."""
        return Path(self._model_path)

    @model_path.setter
    def model_path(self, value: Any) -> None:
        """Set model_path from string or Path."""
        self._model_path = str(value)

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

class EvalDataLoader:
    """Handles data loading for evaluation."""

    def __init__(self, config: EvalConfig):
        """
        Initialize the data loader.

        Args:
            config: Evaluation configuration
        """
        self.config: EvalConfig = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.class_weights: Optional[Dict[int, float]] = None

    def load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        logger.info("✓ Tokenizer loaded")
        return self.tokenizer

    def tokenize_texts(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, List[Any]]:
        """Tokenize texts and prepare encodings."""
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

    def prepare_dataloaders(self) -> Tuple[DataLoader, DataLoader, Dict[int, float]]:
        """
        Prepare validation and test dataloaders.

        Returns:
            Tuple of (val_loader, test_loader, class_weights)
        """
        logger.info("PREPARING DATA FOR EVALUATION")

        # Load data
        df_train: pd.DataFrame = pd.read_csv(self.config.train_path)
        df_test: pd.DataFrame = pd.read_csv(self.config.test_path)

        logger.info(f"\nTraining data: {df_train.shape}")
        logger.info(f"Test data: {df_test.shape}")

        # Load tokenizer
        self.load_tokenizer()

        # Compute class weights
        unique_labels: np.ndarray = np.unique(df_train['label_numeric'])
        weights_array: np.ndarray = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=df_train['label_numeric'],
        )
        self.class_weights = dict(zip(unique_labels, weights_array))

        # Tokenize training data for validation split
        logger.info("\nTokenizing data...")
        full_train_encodings: Dict[str, List[Any]] = self.tokenize_texts(
            df_train['text'].values,
            df_train['label_numeric'].values,
        )

        # Tokenize test data
        test_encodings: Dict[str, List[Any]] = self.tokenize_texts(
            df_test['text'].values,
            df_test['label_numeric'].values,
        )

        # Create validation split (same as training)
        _, val_indices = train_test_split(
            range(len(full_train_encodings['input_ids'])),
            test_size=self.config.val_split,
            random_state=self.config.random_seed,
            stratify=full_train_encodings['labels'],
        )

        val_encodings: Dict[str, List[Any]] = {
            'input_ids': [full_train_encodings['input_ids'][i] for i in val_indices],
            'attention_mask': [full_train_encodings['attention_mask'][i] for i in val_indices],
            'labels': [full_train_encodings['labels'][i] for i in val_indices],
        }

        logger.info("\nDataset sizes:")
        logger.info(f"  Validation: {len(val_encodings['input_ids'])} samples")
        logger.info(f"  Test:       {len(test_encodings['input_ids'])} samples")

        # Create dataloaders
        val_dataset: ASZFDataset = ASZFDataset(val_encodings)
        test_dataset: ASZFDataset = ASZFDataset(test_encodings)

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

        logger.info("\n✓ DataLoaders created")
        logger.info(f"  Val batches:  {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")

        return val_loader, test_loader, self.class_weights

class ModelEvaluator:
    """
    Handles model evaluation and visualization.

    This class is responsible for:
        - Loading trained model
        - Evaluating on validation/test sets
        - Computing classification metrics
        - Computing ordinal evaluation metrics
        - Generating visualizations
    """

    def __init__(self, config: EvalConfig):
        """
        Initialize the evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config: EvalConfig = config
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None

    def load_model(self, class_weights: Dict[int, float]) -> None:
        """
        Load the trained model and set up criterion.

        Args:
            class_weights: Dictionary of class weights for loss function
        """

        logger.info(f"\nDevice: {self.device}")
        logger.info(f"Loading model from: {self.config.model_path}")

        # Initialize model architecture
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            problem_type="single_label_classification",
        )

        # Load trained weights
        checkpoint: Dict[str, Any] = torch.load(self.config.model_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        logger.info(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        logger.info(f"  Validation accuracy at save: {checkpoint['val_accuracy']:.4f}")

        # Set up criterion
        class_weights_list: List[float] = [class_weights[i] for i in range(1, 6)]
        class_weights_tensor: torch.Tensor = torch.tensor(
            class_weights_list,
            dtype=torch.float,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

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

    def print_classification_report(self, labels: np.ndarray, predictions: np.ndarray, split_name: str):
        """Print classification report for a dataset."""
        logger.info(f"\nClassification Report ({split_name}):")
        logger.info(classification_report(
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
        """Print ordinal evaluation metrics."""
        logger.info(f"\n{split_name}:")
        logger.info(f"  Exact Accuracy:     {(predictions == labels).mean():.4f}")
        logger.info(f"  Within-1 Accuracy:  {self.within_k_accuracy(labels, predictions, 1):.4f}")
        logger.info(f"  Within-2 Accuracy:  {self.within_k_accuracy(labels, predictions, 2):.4f}")

    def print_per_class_recall(
        self,
        val_labels: np.ndarray,
        val_preds: np.ndarray,
        test_labels: np.ndarray,
        test_preds: np.ndarray,
    ) -> None:
        """Print per-class recall comparison."""

        logger.info("\nRecall by class:")
        logger.info("-" * 40)
        logger.info(f"{'Class':<10} {'Validation':<15} {'Test':<15}")
        logger.info("-" * 40)

        for c in range(5):
            val_mask: np.ndarray = val_labels == c
            test_mask: np.ndarray = test_labels == c

            val_recall: float = (val_preds[val_mask] == c).sum() / val_mask.sum() if val_mask.sum() > 0 else 0
            test_recall: float = (test_preds[test_mask] == c).sum() / test_mask.sum() if test_mask.sum() > 0 else 0

            logger.info(f"Class {c+1:<4} {val_recall:<15.4f} {test_recall:<15.4f}")

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        split_name: str,
        cmap: str = 'Blues',
    ) -> None:
        """Plot and save confusion matrix."""
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
        #plt.show()

        logger.info(f"✓ Confusion matrix saved to {save_path}")

    def full_evaluation(
        self,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on validation and test sets.

        Args:
            val_loader: Validation DataLoader
            test_loader: Test DataLoader

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("VALIDATION SET RESULTS")

        val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader, "Validation")
        self.print_classification_report(val_labels, val_preds, "Validation")
        self.plot_confusion_matrix(val_labels, val_preds, "Validation", cmap='Blues')

        logger.info("TEST SET RESULTS")

        test_loss, test_acc, test_preds, test_labels = self.evaluate(test_loader, "Test")
        self.print_classification_report(test_labels, test_preds, "Test")
        self.plot_confusion_matrix(test_labels, test_preds, "Test", cmap='Greens')

        logger.info("ORDINAL EVALUATION METRICS")

        self.print_ordinal_metrics(val_labels, val_preds, "Validation Set")
        self.print_ordinal_metrics(test_labels, test_preds, "Test Set")

        # Per-class recall
        self.print_per_class_recall(val_labels, val_preds, test_labels, test_preds)

        # Print summary
        self._print_summary(val_labels, val_preds, test_labels, test_preds)

        # Return results dictionary
        return {
            'validation': {
                'loss': val_loss,
                'accuracy': val_acc,
                'predictions': val_preds,
                'labels': val_labels,
                'within_1': self.within_k_accuracy(val_labels, val_preds, 1),
                'within_2': self.within_k_accuracy(val_labels, val_preds, 2),
            },
            'test': {
                'loss': test_loss,
                'accuracy': test_acc,
                'predictions': test_preds,
                'labels': test_labels,
                'within_1': self.within_k_accuracy(test_labels, test_preds, 1),
                'within_2': self.within_k_accuracy(test_labels, test_preds, 2),
            },
        }

    def _print_summary(
        self,
        val_labels: np.ndarray,
        val_preds: np.ndarray,
        test_labels: np.ndarray,
        test_preds: np.ndarray,
    ) -> None:
        """Print final evaluation summary."""
        logger.info("EVALUATION SUMMARY")

        logger.info(f"""
Model: {self.config.model_name}
Model path: {self.config.model_path}

Results:
  Validation:
    - Accuracy: {(val_preds == val_labels).mean():.4f}
    - Within-1: {self.within_k_accuracy(val_labels, val_preds, 1):.4f}
    - Within-2: {self.within_k_accuracy(val_labels, val_preds, 2):.4f}

  Test:
    - Accuracy: {(test_preds == test_labels).mean():.4f}
    - Within-1: {self.within_k_accuracy(test_labels, test_preds, 1):.4f}
    - Within-2: {self.within_k_accuracy(test_labels, test_preds, 2):.4f}

Figures saved to: {self.config.model_dir}
""")

def main() -> None:
    """Main entry point for the evaluation script."""
    logger.info("Model Evaluation")

    # Initialize configuration
    config: EvalConfig = EvalConfig()

    #config.train_path = '/content/sample_data/train.csv'
    #config.test_path = '/content/sample_data/test.csv'
    #config.model_dir = '/content/sample_data/model'
    #config.model_path = '/content/sample_data/model/best_model.pt'

    logger.info("\nConfiguration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Train path: {config.train_path}")
    logger.info(f"  Test path: {config.test_path}")
    logger.info(f"  Model path: {config.model_path}")

    # ==========================================================================
    # PREPARE DATA
    # ==========================================================================
    data_loader: EvalDataLoader = EvalDataLoader(config)
    val_loader, test_loader, class_weights = data_loader.prepare_dataloaders()

    # ==========================================================================
    # EVALUATE
    # ==========================================================================
    evaluator: ModelEvaluator = ModelEvaluator(config)
    evaluator.load_model(class_weights)
    results: Dict[str, Any] = evaluator.full_evaluation(val_loader, test_loader)

    logger.info("EVALUATION COMPLETE!")

    return results

if __name__ == '__main__':
    main()
