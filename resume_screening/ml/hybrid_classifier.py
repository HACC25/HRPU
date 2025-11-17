"""
Hybrid classifier that combines rule-based requirement checking with ML text analysis

This approach leverages the correlation between requirements met and overall classification.

The hybrid system:
1. Uses rules for clear-cut cases (all requirements met or very few met)
2. Uses ML text analysis only for borderline "gray zone" cases
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HybridResumeClassifier:
    """
    Two-stage classifier combining rule-based and ML approaches

    Supports variable number of requirements per job position.
    """

    def __init__(
        self,
        num_requirements: int = 5,
        qualified_threshold: Optional[int] = None,
        not_qualified_threshold: Optional[int] = None
    ):
        """
        Initialize hybrid classifier

        Args:
            num_requirements: Number of requirements for this job (e.g., 5 for AGLO, 4 for OAIV)
            qualified_threshold: Min requirements for LIKELY_QUALIFIED (defaults to num_requirements)
            not_qualified_threshold: Max requirements for LIKELY_NOT_QUALIFIED (defaults to ~40% of num)

        NOTE: Threshold selection is critical for model performance:
            - Gray zone (between thresholds) should contain genuinely ambiguous cases
            - If gray zone has too few samples or is heavily imbalanced, ML classifier will struggle
            - Consider your ground truth labels when setting thresholds
            - Example: For AGLO (5 reqs), many 3-4 req applicants are labeled NOT_QUALIFIED,
              suggesting not_qualified_threshold could be raised to 3 or even 4
        """
        self.ml_classifier = None
        self.scaler = None
        self.bert_model = None
        self.num_requirements = num_requirements

        # Set thresholds with sensible defaults
        if qualified_threshold is None:
            qualified_threshold = num_requirements  # Must meet all requirements

        if not_qualified_threshold is None:
            # Default: 40% or less of requirements → definitely not qualified
            # NOTE: This is conservative. Consider raising based on your data.
            not_qualified_threshold = max(0, int(num_requirements * 0.4))

        self.req_thresholds = {
            'LIKELY_QUALIFIED': qualified_threshold,
            'LIKELY_NOT_QUALIFIED': not_qualified_threshold,
        }

    def classify_by_requirements(self, requirement_vector: List[int]) -> Tuple[Optional[str], float]:
        """
        Stage 1: Rule-based classification using requirements

        Args:
            requirement_vector: List of binary values (1 = met, 0 = not met)

        Returns:
            (classification, confidence) where classification is None if needs ML analysis
        """
        req_sum = sum(requirement_vector)

        if req_sum >= self.req_thresholds['LIKELY_QUALIFIED']:
            # Meets all requirements → definitely qualified
            return 'LIKELY_QUALIFIED', 1.0

        elif req_sum <= self.req_thresholds['LIKELY_NOT_QUALIFIED']:
            # Meets very few requirements → definitely not qualified
            return 'LIKELY_NOT_QUALIFIED', 1.0

        else:
            # Gray zone (3-4 requirements met) → needs text analysis
            return None, 0.0

    def train_ml_classifier(
        self,
        texts: List[str],
        requirement_vectors: List[List[int]],
        labels: List[str],
        bert_embeddings: np.ndarray
    ):
        """
        Stage 2: Train ML classifier for gray zone cases

        Args:
            texts: Resume texts (not used if embeddings provided)
            requirement_vectors: List of requirement vectors
            labels: Overall classification labels
            bert_embeddings: Pre-computed BERT embeddings [n_samples, hidden_size]
        """
        # Calculate gray zone bounds (between not_qualified and qualified thresholds)
        gray_zone_lower = self.req_thresholds['LIKELY_NOT_QUALIFIED'] + 1
        gray_zone_upper = self.req_thresholds['LIKELY_QUALIFIED'] - 1

        # Filter for gray zone cases only
        gray_zone_mask = [
            gray_zone_lower <= sum(req_vec) <= gray_zone_upper
            for req_vec in requirement_vectors
        ]

        if sum(gray_zone_mask) == 0:
            print("Warning: No gray zone samples found for training ML classifier")
            return

        # Extract gray zone data
        gray_embeddings = bert_embeddings[gray_zone_mask]
        gray_req_vecs = np.array([req for req, mask in zip(requirement_vectors, gray_zone_mask) if mask])
        gray_labels = [label for label, mask in zip(labels, gray_zone_mask) if mask]

        # Check if we have at least 2 classes in gray zone
        unique_labels = set(gray_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Gray zone has only one class ({unique_labels}). Skipping ML classifier training.")
            print(f"  ML classifier will fall back to NEEDS_REVIEW for gray zone cases.")
            return

        # Check class balance
        from collections import Counter
        class_counts = Counter(gray_labels)
        print(f"Gray zone class distribution: {dict(class_counts)}")

        # Warn if severely imbalanced
        min_class_size = min(class_counts.values())
        max_class_size = max(class_counts.values())
        if min_class_size < 3:
            print(f"Warning: Very few samples in minority class ({min_class_size}). ML predictions may be unreliable.")
        if max_class_size / min_class_size > 5:
            print(f"Warning: Severe class imbalance (ratio {max_class_size}/{min_class_size}). Using balanced class weights.")

        # Combine BERT embeddings with requirement features
        features = np.concatenate([gray_embeddings, gray_req_vecs], axis=1)

        # Normalize features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        # Train logistic regression with class balancing and regularization
        self.ml_classifier = LogisticRegression(
            class_weight='balanced',  # Automatically adjust weights inversely proportional to class frequencies
            max_iter=2000,  # Increased from 1000 for better convergence
            random_state=42,
            C=0.1,  # Strong regularization to prevent overfitting on small datasets
            solver='lbfgs'
        )
        self.ml_classifier.fit(features_scaled, gray_labels)

        print(f"Trained ML classifier on {len(gray_labels)} gray zone samples")
        print(f"  Class weights: {dict(zip(self.ml_classifier.classes_, self.ml_classifier.coef_[0][:3] if len(self.ml_classifier.classes_) <= 3 else ['...']))}")

    def predict(
        self,
        texts: List[str],
        requirement_vectors: List[List[int]],
        bert_embeddings: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Predict using hybrid approach

        Args:
            texts: Resume texts
            requirement_vectors: List of requirement vectors
            bert_embeddings: Optional pre-computed BERT embeddings

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i, (text, req_vec) in enumerate(zip(texts, requirement_vectors)):
            # Stage 1: Try rule-based classification
            rule_pred, rule_conf = self.classify_by_requirements(req_vec)

            if rule_pred is not None:
                # Rule gave clear answer
                results.append({
                    'classification': rule_pred,
                    'confidence': rule_conf,
                    'method': 'rule-based',
                    'requirements_met': sum(req_vec),
                    'requirements_vector': req_vec
                })
            else:
                # Gray zone → use ML classifier
                if self.ml_classifier is None or bert_embeddings is None:
                    # Fallback: predict NEEDS_REVIEW for gray zone
                    results.append({
                        'classification': 'NEEDS_REVIEW',
                        'confidence': 0.5,
                        'method': 'fallback',
                        'requirements_met': sum(req_vec),
                        'requirements_vector': req_vec
                    })
                else:
                    # Use trained ML classifier
                    embedding = bert_embeddings[i:i+1]
                    req_features = np.array([req_vec])
                    features = np.concatenate([embedding, req_features], axis=1)
                    features_scaled = self.scaler.transform(features)

                    ml_pred = self.ml_classifier.predict(features_scaled)[0]
                    ml_probs = self.ml_classifier.predict_proba(features_scaled)[0]
                    ml_conf = ml_probs.max()

                    results.append({
                        'classification': ml_pred,
                        'confidence': ml_conf,
                        'method': 'ml',
                        'requirements_met': sum(req_vec),
                        'requirements_vector': req_vec,
                        'ml_probabilities': dict(zip(self.ml_classifier.classes_, ml_probs))
                    })

        return results

    def evaluate(
        self,
        texts: List[str],
        requirement_vectors: List[List[int]],
        true_labels: List[str],
        bert_embeddings: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate hybrid classifier

        Args:
            texts: Resume texts
            requirement_vectors: Requirement vectors
            true_labels: Ground truth labels
            bert_embeddings: Optional pre-computed embeddings

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(texts, requirement_vectors, bert_embeddings)

        pred_labels = [p['classification'] for p in predictions]

        # Calculate accuracy
        correct = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)
        accuracy = correct / len(true_labels)

        # Calculate per-class metrics
        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True,
            zero_division=0
        )

        conf_matrix = confusion_matrix(
            true_labels,
            pred_labels,
            labels=['LIKELY_QUALIFIED', 'NEEDS_REVIEW', 'LIKELY_NOT_QUALIFIED']
        )

        # Count how many were rule-based vs ML
        method_counts = {}
        for pred in predictions:
            method = pred['method']
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'method_counts': method_counts,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }

    def save(self, output_dir: str):
        """
        Save hybrid classifier to disk

        Args:
            output_dir: Directory to save model components
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save ML classifier and scaler
        with open(f"{output_dir}/ml_classifier.pkl", 'wb') as f:
            pickle.dump(self.ml_classifier, f)

        with open(f"{output_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save thresholds and num_requirements
        config = {
            'req_thresholds': self.req_thresholds,
            'num_requirements': self.num_requirements
        }
        with open(f"{output_dir}/config.pkl", 'wb') as f:
            pickle.dump(config, f)

        # Save BERT model reference (path only, not the model itself)
        # Assumption: BERT model saved separately and loaded independently
        print(f"Saved hybrid classifier to {output_dir}/")
        print("  Note: BERT model must be loaded separately")

    @classmethod
    def load(cls, model_dir: str, bert_model=None):
        """
        Load hybrid classifier from disk

        Args:
            model_dir: Directory containing saved model
            bert_model: Pre-loaded BERT model (must be provided)

        Returns:
            Loaded HybridResumeClassifier
        """
        # Load config first
        with open(f"{model_dir}/config.pkl", 'rb') as f:
            config = pickle.load(f)

        # Create hybrid instance with loaded config
        hybrid = cls(
            num_requirements=config.get('num_requirements', 5),
            qualified_threshold=config['req_thresholds'].get('LIKELY_QUALIFIED'),
            not_qualified_threshold=config['req_thresholds'].get('LIKELY_NOT_QUALIFIED')
        )

        # Load ML classifier and scaler
        with open(f"{model_dir}/ml_classifier.pkl", 'rb') as f:
            hybrid.ml_classifier = pickle.load(f)

        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            hybrid.scaler = pickle.load(f)

        # Set BERT model
        hybrid.bert_model = bert_model

        print(f"Loaded hybrid classifier from {model_dir}/")
        return hybrid


def create_hybrid_classifier_from_data(
    texts: List[str],
    requirement_vectors: List[List[int]],
    labels: List[str],
    bert_model
) -> HybridResumeClassifier:
    """
    Create and train a hybrid classifier from training data

    Args:
        texts: Training texts
        requirement_vectors: Training requirement vectors
        labels: Training labels
        bert_model: Trained BERT model for embeddings

    Returns:
        Trained HybridResumeClassifier
    """
    # Get BERT embeddings
    bert_embeddings = bert_model.get_embedding(texts)

    # Create and train hybrid classifier
    hybrid = HybridResumeClassifier()
    hybrid.bert_model = bert_model
    hybrid.train_ml_classifier(
        texts=texts,
        requirement_vectors=requirement_vectors,
        labels=labels,
        bert_embeddings=bert_embeddings
    )

    return hybrid
