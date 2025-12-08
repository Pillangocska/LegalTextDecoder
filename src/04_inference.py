"""
HuBERT Model Inference Script

This script runs inference on new, unseen Hungarian legal texts (ÁSZF)
using a trained HuBERT readability classification model.

Input:
    - output/best_model.pt (trained model)
    - Text data (embedded in script or provided via function)

Output:
    - Predicted readability scores (1-5) with confidence scores

Usage:
    python 04_inference.py

    Or import and use programmatically:
        from 04_inference import ReadabilityPredictor
        predictor = ReadabilityPredictor(model_path='path/to/model.pt')
        results = predictor.predict(["Your text here..."])
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import torch

from src.util.config_manager import config
from src.util.logger import Logger
logger: Logger = Logger("inference")

@dataclass
class InferenceConfig:
    """Configuration for inference."""

    model_name: str = config.get("train.model_name")
    num_classes: int = config.get("train.num_classes")
    max_length: int = config.get("train.max_token_length")
    _model_path: str = config.get("train.model_path")

    # Label descriptions (Hungarian)
    label_descriptions: Dict[int, str] = None

    def __post_init__(self) -> None:
        """Initialize label descriptions."""
        self.label_descriptions = {
            1: "Nagyon nehezen érthető (Very difficult)",
            2: "Nehezen érthető (Difficult)",
            3: "Többé-kevésbé érthető (Somewhat understandable)",
            4: "Érthető (Understandable)",
            5: "Könnyen érthető (Easily understandable)",
        }

    @property
    def model_path(self) -> Path:
        """Get model_path as Path object."""
        return Path(self._model_path)

    @model_path.setter
    def model_path(self, value: Any) -> None:
        """Set model_path from string or Path."""
        self._model_path = str(value)

# Real-life Hungarian ÁSZF (Terms and Conditions) sample texts
# These represent various difficulty levels typical in legal documents

SAMPLE_TEXTS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "text": "A szolgáltatás díja havi 1000 Ft.",
        "description": "Simple pricing statement",
        "expected_difficulty": "easy",
    },
    {
        "id": 2,
        "text": "A szerződés határozatlan időre jön létre és bármelyik fél 30 napos felmondási idővel megszüntetheti.",
        "description": "Basic contract term",
        "expected_difficulty": "easy",
    },
    {
        "id": 3,
        "text": "Az Ügyfél a szerződéstől a termék átvételétől számított 14 napon belül indokolás nélkül elállhat. Az elállási jog gyakorlása esetén az Ügyfél köteles a terméket haladéktalanul, de legkésőbb 14 napon belül visszaküldeni.",
        "description": "Consumer withdrawal rights",
        "expected_difficulty": "medium",
    },
    {
        "id": 4,
        "text": "A Szolgáltató fenntartja a jogot az Általános Szerződési Feltételek egyoldalú módosítására. A módosításról a Szolgáltató a hatálybalépést megelőzően legalább 15 nappal korábban értesíti az Előfizetőt elektronikus úton.",
        "description": "Terms modification clause",
        "expected_difficulty": "medium",
    },
    {
        "id": 5,
        "text": "A Ptk. 6:78. § (1) bekezdése alapján a fogyasztó és a vállalkozás közötti szerződésben semmis az a kikötés, amely a fogyasztóval szerződő vállalkozás javára egyoldalúan és indokolatlanul hátrányos a fogyasztóra nézve.",
        "description": "Legal reference to Civil Code",
        "expected_difficulty": "hard",
    },
    {
        "id": 6,
        "text": "Amennyiben a Felhasználó a jelen ÁSZF-ben foglalt kötelezettségeit megszegi, különös tekintettel a szellemi tulajdonjogok megsértésére, a Szolgáltató jogosult a Felhasználó hozzáférését azonnali hatállyal felfüggeszteni vagy megszüntetni, és a Felhasználóval szemben kártérítési igényt érvényesíteni a Ptk. vonatkozó rendelkezései szerint.",
        "description": "IP violation consequences",
        "expected_difficulty": "hard",
    },
    {
        "id": 7,
        "text": "A Szolgáltató a szerződésszegéssel okozott károkért való felelősségét – a szándékosan okozott, illetve emberi életet, testi épséget vagy egészséget megkárosító szerződésszegésért való felelősség kivételével – az adott szolgáltatás egyéves díjának összegében korlátozza.",
        "description": "Liability limitation clause",
        "expected_difficulty": "hard",
    },
    {
        "id": 8,
        "text": "Adatkezelő: XY Kft. (székhely: 1234 Budapest, Példa utca 1.) Az adatkezelés célja: szerződés teljesítése. Az adatkezelés jogalapja: GDPR 6. cikk (1) bekezdés b) pont.",
        "description": "GDPR data controller info",
        "expected_difficulty": "medium",
    },
    {
        "id": 9,
        "text": "A felek a jelen szerződésből eredő vitáikat elsődlegesen békés úton, egyeztetéssel kísérlik meg rendezni. Ennek eredménytelensége esetén a felek kikötik a Budapesti II. és III. Kerületi Bíróság kizárólagos illetékességét.",
        "description": "Dispute resolution clause",
        "expected_difficulty": "medium",
    },
    {
        "id": 10,
        "text": "Az Előfizető tudomásul veszi, hogy az Eszr. 11. § (1) bekezdésében meghatározott, a szerződéskötést megelőző tájékoztatási kötelezettség teljesítése a Szolgáltató részéről az ÁSZF Előfizető részére történő hozzáférhetővé tételével megtörténik, amennyiben az ÁSZF tartalmazza az Eszr. 11. § (1) bekezdésében előírt információkat.",
        "description": "Complex regulatory reference",
        "expected_difficulty": "very_hard",
    },
]

class ReadabilityPredictor:
    """
    Predicts readability scores for Hungarian legal texts.

    This class loads a trained HuBERT model and provides methods
    for predicting readability scores on new texts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model (overrides config)
            config: Inference configuration
            logger: Logger instance
        """
        self.config: InferenceConfig = config or InferenceConfig()

        if model_path:
            self.config.model_path = model_path

        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model from: {self.config.model_path}")
        logger.info(f"Device: {self.device}")

        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Initialize model architecture
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_classes,
            problem_type="single_label_classification",
        )

        # Load trained weights
        checkpoint: Dict[str, Any] = torch.load(
            self.config.model_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully (trained epoch: {checkpoint['epoch']})")

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict readability for a single text.

        Args:
            text: Input text

        Returns:
            Dictionary with prediction results
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Move to device
        input_ids: torch.Tensor = encoding['input_ids'].to(self.device)
        attention_mask: torch.Tensor = encoding['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits: torch.Tensor = outputs.logits
            probabilities: torch.Tensor = torch.softmax(logits, dim=1)

        # Get prediction (convert back from 0-4 to 1-5)
        predicted_class: int = int(torch.argmax(probabilities, dim=1).item()) + 1
        confidence: float = float(probabilities.max().item())
        all_probs: List[float] = probabilities.squeeze().cpu().tolist()

        return {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'text_length': len(text),
            'predicted_label': predicted_class,
            'label_description': self.config.label_descriptions[predicted_class],
            'confidence': confidence,
            'probabilities': {i + 1: float(p) for i, p in enumerate(all_probs)},
        }

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict readability for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Running inference on {len(texts)} texts...")

        results: List[Dict[str, Any]] = []

        for i, text in enumerate(texts):
            result: Dict[str, Any] = self.predict_single(text)
            result['index'] = i
            results.append(result)

        logger.info("Inference complete")

        return results

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Predict readability for multiple texts using batched inference.

        Args:
            texts: List of input texts
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Running batched inference on {len(texts)} texts (batch_size={batch_size})...")

        results: List[Dict[str, Any]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts: List[str] = texts[batch_start:batch_start + batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

            input_ids: torch.Tensor = encodings['input_ids'].to(self.device)
            attention_mask: torch.Tensor = encodings['attention_mask'].to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits: torch.Tensor = outputs.logits
                probabilities: torch.Tensor = torch.softmax(logits, dim=1)

            # Process results
            predicted_classes: torch.Tensor = torch.argmax(probabilities, dim=1) + 1
            confidences: torch.Tensor = probabilities.max(dim=1).values

            for i, (text, pred, conf, probs) in enumerate(zip(
                batch_texts,
                predicted_classes.cpu().tolist(),
                confidences.cpu().tolist(),
                probabilities.cpu().tolist(),
            )):
                results.append({
                    'index': batch_start + i,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'text_length': len(text),
                    'predicted_label': pred,
                    'label_description': self.config.label_descriptions[pred],
                    'confidence': conf,
                    'probabilities': {j + 1: float(p) for j, p in enumerate(probs)},
                })

        logger.info("Batched inference complete")

        return results

def print_results(
    results: List[Dict[str, Any]],
    logger: Logger,
    samples: Optional[List[Dict[str, Any]]] = None,
    show_probabilities: bool = False,
) -> None:
    """
    Print prediction results in a formatted way.

    Args:
        results: List of prediction dictionaries
        logger: Logger instance
        samples: Optional sample data with expected difficulty
        show_probabilities: Whether to show full probability distribution
    """
    difficulty_mapping: Dict[str, str] = {
        'easy': '4-5 (Érthető/Könnyen érthető)',
        'medium': '3 (Többé-kevésbé érthető)',
        'hard': '2 (Nehezen érthető)',
        'very_hard': '1 (Nagyon nehezen érthető)',
    }

    logger.info("PREDICTION RESULTS")

    for i, result in enumerate(results):
        sample = samples[i] if samples else None

        logger.info("")
        if sample:
            logger.info(f"[Text {result['index'] + 1}] {sample['description']}")
            logger.info(f"  Expected: {sample['expected_difficulty']} → {difficulty_mapping[sample['expected_difficulty']]}")
        else:
            logger.info(f"[Text {result['index'] + 1}] {result['text']}")
        logger.info(f"  Predicted: {result['predicted_label']} - {result['label_description']}")
        logger.info(f"  Confidence: {result['confidence']:.2%}")
        logger.info(f"  Length: {result['text_length']} chars")

        if show_probabilities:
            prob_str: str = " | ".join([
                f"L{k}: {v:.1%}" for k, v in result['probabilities'].items()
            ])
            logger.info(f"  Probabilities: {prob_str}")

    logger.info("SUMMARY")

    predictions: List[int] = [r['predicted_label'] for r in results]
    confidences: List[float] = [r['confidence'] for r in results]

    logger.info(f"Total texts: {len(results)}")
    logger.info(f"Average confidence: {sum(confidences) / len(confidences):.2%}")
    logger.info("")
    logger.info("Label distribution:")

    for label in range(1, 6):
        count: int = predictions.count(label)
        pct: float = count / len(predictions) * 100
        logger.info(f"  Label {label}: {count} ({pct:.1f}%)")

def predict() -> List[Dict[str, Any]]:
    """
    Run inference on sample data.

    Returns:
        List of prediction results
    """

    logger.info("Inference Script")

    config: InferenceConfig = InferenceConfig()

    #config.model_path = '/content/sample_data/model/best_model.pt'

    logger.info(f"Model path: {config.model_path}")

    # Initialize predictor
    predictor: ReadabilityPredictor = ReadabilityPredictor(config=config, logger=logger)

    # Extract texts from sample data
    texts: List[str] = [sample['text'] for sample in SAMPLE_TEXTS]

    logger.info("")
    logger.info(f"Sample texts to classify: {len(texts)}")

    # Show sample data info
    logger.info("")
    logger.info("Sample data overview:")
    for sample in SAMPLE_TEXTS:
        logger.info(f"  [{sample['id']}] {sample['description']} (expected: {sample['expected_difficulty']})")

    # Run inference
    logger.info("")
    results: List[Dict[str, Any]] = predictor.predict_batch(texts)

    # Print results
    print_results(results, logger, samples=SAMPLE_TEXTS, show_probabilities=True)

    logger.info("INFERENCE COMPLETE")

    return results


if __name__ == "__main__":
    predict()
