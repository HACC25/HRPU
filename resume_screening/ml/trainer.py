"""
Training and evaluation utilities for the resume screening model
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd

from .model import ResumeScreeningModel


class ResumeDataset(Dataset):
    """PyTorch Dataset for resume data"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize dataset

        Args:
            texts: List of resume texts
            labels: List of labels (0, 1, 2)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ResumeScreeningTrainer:
    """Trainer for the resume screening model"""

    def __init__(
        self,
        model: ResumeScreeningModel,
        train_dataset: ResumeDataset,
        val_dataset: ResumeDataset,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        warmup_steps: int = 0,
        output_dir: str = './models'
    ):
        """
        Initialize trainer

        Args:
            model: ResumeScreeningModel instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            output_dir: Directory to save models and checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            eps=1e-8
        )

        # Learning rate scheduler
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc='Training'):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            labels = batch['labels'].to(self.model.device)

            # Forward pass
            logits, _ = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate loss
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def evaluate(self) -> Dict:
        """Evaluate on validation set"""
        self.model.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)

                # Forward pass
                logits, _ = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted'
        )

        # Classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=list(self.model.LABEL_MAP.values()),
            output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

    def train(self) -> Dict:
        """Train the model"""
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Device: {self.model.device}")

        best_f1 = 0
        best_epoch = 0

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")

            # Evaluate
            metrics = self.evaluate()
            print(f"Validation Loss: {metrics['loss']:.4f}")
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            print(f"Validation F1: {metrics['f1']:.4f}")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(metrics['loss'])
            self.history['val_accuracy'].append(metrics['accuracy'])
            self.history['val_f1'].append(metrics['f1'])

            # Save best model
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_epoch = epoch + 1
                model_path = os.path.join(self.output_dir, 'best_model.pt')
                self.model.save(model_path)
                print(f"Saved best model with F1: {best_f1:.4f}")

            # Save checkpoint
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            self.model.save(checkpoint_path)

        print(f"\nTraining completed!")
        print(f"Best model at epoch {best_epoch} with F1: {best_f1:.4f}")

        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def test(self, test_dataset: ResumeDataset) -> Dict:
        """
        Test the model on a test set

        Args:
            test_dataset: Test dataset

        Returns:
            Dictionary containing test metrics
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)

                # Forward pass
                logits, _ = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted'
        )

        # Classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=list(self.model.LABEL_MAP.values())
        )

        print("\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(report)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs
        }
