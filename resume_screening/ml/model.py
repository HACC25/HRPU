"""
BERT-based resume screening classifier
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from typing import Dict, List, Tuple


class ResumeScreeningClassifier(nn.Module):
    """
    BERT-based classifier for resume screening with 3 output classes:
    - LIKELY_QUALIFIED
    - NEEDS_REVIEW
    - LIKELY_NOT_QUALIFIED
    """

    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 3, dropout_rate: float = 0.3):
        """
        Initialize the classifier

        Args:
            model_name: Name of the pre-trained BERT model
            num_labels: Number of classification labels (default: 3)
            dropout_rate: Dropout rate for regularization
        """
        super(ResumeScreeningClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # Additional layer for better representation
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional)

        Returns:
            Tuple of (logits, pooled_output)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output

        # Additional processing
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)

        return logits, pooled_output

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str, device: str = 'cpu'):
        """Load model weights"""
        self.load_state_dict(torch.load(path, map_location=device))


class ResumeScreeningModel:
    """
    Wrapper class for the resume screening model with tokenizer and inference utilities
    """

    LABEL_MAP = {
        0: 'LIKELY_QUALIFIED',
        1: 'NEEDS_REVIEW',
        2: 'LIKELY_NOT_QUALIFIED'
    }

    LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        device: str = None
    ):
        """
        Initialize the model wrapper

        Args:
            model_name: Name of the pre-trained BERT model
            max_length: Maximum sequence length
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Initialize model
        self.model = ResumeScreeningClassifier(model_name=model_name)
        self.model.to(self.device)

    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts

        Args:
            texts: List of text strings

        Returns:
            Dictionary containing tokenized inputs
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        return encoded

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict classifications for input texts

        Args:
            texts: List of resume texts

        Returns:
            List of dictionaries containing predictions
        """
        self.model.eval()

        with torch.no_grad():
            encoded = self.tokenize(texts)
            logits, _ = self.model(**encoded)

            # Get probabilities
            probs = torch.softmax(logits, dim=1)

            # Get predictions
            predictions = torch.argmax(probs, dim=1)

            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probs)):
                pred_label = self.LABEL_MAP[pred.item()]
                confidence = prob[pred].item()

                results.append({
                    'classification': pred_label,
                    'confidence': confidence,
                    'probabilities': {
                        self.LABEL_MAP[j]: prob[j].item()
                        for j in range(len(self.LABEL_MAP))
                    }
                })

            return results

    def save(self, model_path: str, tokenizer_path: str = None):
        """
        Save model and tokenizer

        Args:
            model_path: Path to save model weights
            tokenizer_path: Path to save tokenizer (optional)
        """
        self.model.save_model(model_path)

        if tokenizer_path:
            self.tokenizer.save_pretrained(tokenizer_path)

    def load(self, model_path: str, tokenizer_path: str = None):
        """
        Load model and tokenizer

        Args:
            model_path: Path to model weights
            tokenizer_path: Path to tokenizer (optional)
        """
        self.model.load_model(model_path, device=self.device)

        if tokenizer_path:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
