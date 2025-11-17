"""
Multi-task BERT-based resume screening model
Supports:
- Overall qualification classification (3-class)
- Question quality scoring (regression per question)
- Requirement matching (binary classification per requirement)
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np


class MultiTaskResumeClassifier(nn.Module):
    """
    Multi-task BERT model for resume screening with multiple prediction heads
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        num_overall_classes: int = 3,
        num_questions: int = 13,
        num_requirements: int = 5,
        dropout_rate: float = 0.3,
        hidden_size: int = 768
    ):
        """
        Initialize multi-task classifier

        Args:
            model_name: Pre-trained transformer model name
            num_overall_classes: Number of overall classification classes (default: 3)
            num_questions: Number of job-specific questions (default: 13 for AGLO)
            num_requirements: Number of job requirements to check (default: 5)
            dropout_rate: Dropout rate for regularization
            hidden_size: Hidden layer size
        """
        super(MultiTaskResumeClassifier, self).__init__()

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        # Shared intermediate layer
        self.shared_layer = nn.Linear(bert_hidden_size, hidden_size)
        self.relu = nn.ReLU()

        # Task-specific heads

        # 1. Overall Qualification Classification Head (3-class)
        self.overall_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_overall_classes)
        )

        # 2. Question Quality Scoring Head (regression for each question)
        self.question_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_questions),
            nn.Sigmoid()  # Output scores between 0 and 1
        )

        # 3. Requirement Matching Head (binary for each requirement)
        self.requirement_matcher = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_requirements),
            nn.Sigmoid()  # Output probabilities for each requirement
        )

        # 4. Continuous Overall Score Head (regression, 0-1)
        self.overall_score_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional) [batch_size, seq_len]

        Returns:
            Dictionary with outputs for each task:
            - 'overall_logits': [batch_size, num_overall_classes]
            - 'question_scores': [batch_size, num_questions]
            - 'requirement_probs': [batch_size, num_requirements]
            - 'overall_score': [batch_size, 1]
            - 'pooled_output': [batch_size, hidden_size]
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, bert_hidden_size]

        # Shared processing
        shared = self.shared_layer(pooled_output)
        shared = self.relu(shared)
        shared = self.dropout(shared)

        # Task-specific outputs
        overall_logits = self.overall_classifier(shared)
        question_scores = self.question_scorer(shared)
        requirement_probs = self.requirement_matcher(shared)
        overall_score = self.overall_score_regressor(shared)

        return {
            'overall_logits': overall_logits,
            'question_scores': question_scores,
            'requirement_probs': requirement_probs,
            'overall_score': overall_score,
            'pooled_output': shared
        }

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str, device: str = 'cpu'):
        """Load model weights"""
        self.load_state_dict(torch.load(path, map_location=device))


class MultiTaskResumeModel:
    """
    Wrapper class for multi-task resume screening model with tokenizer and inference utilities
    """

    LABEL_MAP = {
        0: 'LIKELY_QUALIFIED',
        1: 'NEEDS_REVIEW',
        2: 'LIKELY_NOT_QUALIFIED'
    }

    LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-mpnet-base-v2',
        max_length: int = 512,
        num_questions: int = 13,
        num_requirements: int = 5,
        device: str = None
    ):
        """
        Initialize the multi-task model wrapper

        Args:
            model_name: Name of the pre-trained transformer model
            max_length: Maximum sequence length
            num_questions: Number of job-specific questions
            num_requirements: Number of requirements to check
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_questions = num_questions
        self.num_requirements = num_requirements
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        self.model = MultiTaskResumeClassifier(
            model_name=model_name,
            num_questions=num_questions,
            num_requirements=num_requirements
        )
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
        Predict all tasks for input texts

        Args:
            texts: List of resume texts

        Returns:
            List of dictionaries containing predictions for all tasks
        """
        self.model.eval()

        with torch.no_grad():
            encoded = self.tokenize(texts)
            outputs = self.model(**encoded)

            # Get overall classification
            overall_probs = torch.softmax(outputs['overall_logits'], dim=1)
            overall_preds = torch.argmax(overall_probs, dim=1)

            # Get question scores
            question_scores = outputs['question_scores']

            # Get requirement probabilities
            requirement_probs = outputs['requirement_probs']

            # Get overall score
            overall_scores = outputs['overall_score']

            results = []
            for i in range(len(texts)):
                pred_label = self.LABEL_MAP[overall_preds[i].item()]
                confidence = overall_probs[i][overall_preds[i]].item()

                results.append({
                    'overall_classification': pred_label,
                    'overall_confidence': confidence,
                    'overall_probabilities': {
                        self.LABEL_MAP[j]: overall_probs[i][j].item()
                        for j in range(len(self.LABEL_MAP))
                    },
                    'overall_score': overall_scores[i].item(),
                    'question_scores': question_scores[i].cpu().numpy().tolist(),
                    'requirement_matches': requirement_probs[i].cpu().numpy().tolist(),
                    'avg_question_score': question_scores[i].mean().item(),
                    'met_requirements_count': (requirement_probs[i] > 0.5).sum().item()
                })

            return results

    def predict_single_task(self, texts: List[str], task: str) -> torch.Tensor:
        """
        Predict a single task

        Args:
            texts: List of resume texts
            task: Task name ('overall', 'questions', 'requirements', 'score')

        Returns:
            Tensor with predictions for the specified task
        """
        self.model.eval()

        with torch.no_grad():
            encoded = self.tokenize(texts)
            outputs = self.model(**encoded)

            if task == 'overall':
                return torch.argmax(torch.softmax(outputs['overall_logits'], dim=1), dim=1)
            elif task == 'questions':
                return outputs['question_scores']
            elif task == 'requirements':
                return outputs['requirement_probs']
            elif task == 'score':
                return outputs['overall_score']
            else:
                raise ValueError(f"Unknown task: {task}")

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
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def get_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Get BERT embeddings for texts

        Args:
            texts: List of texts

        Returns:
            Numpy array of embeddings [len(texts), hidden_size]
        """
        self.model.eval()

        with torch.no_grad():
            encoded = self.tokenize(texts)
            outputs = self.model(**encoded)
            embeddings = outputs['pooled_output']

        return embeddings.cpu().numpy()

    def predict_requirements(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary requirements for texts

        Args:
            texts: List of resume texts
            threshold: Probability threshold for binary classification (default: 0.5)

        Returns:
            Numpy array of binary predictions [len(texts), num_requirements]
            where 1 = requirement met, 0 = requirement not met
        """
        self.model.eval()

        with torch.no_grad():
            encoded = self.tokenize(texts)
            outputs = self.model(**encoded)
            requirement_probs = outputs['requirement_probs']

            # Convert probabilities to binary predictions
            binary_predictions = (requirement_probs > threshold).int()

        return binary_predictions.cpu().numpy()
