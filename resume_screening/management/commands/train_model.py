"""
Django management command to train the resume screening model
"""
import os
import json
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import pandas as pd

from resume_screening.utils.data_preparation import DataPreparation
from resume_screening.ml.model import ResumeScreeningModel
from resume_screening.ml.trainer import ResumeDataset, ResumeScreeningTrainer


class Command(BaseCommand):
    help = 'Train the resume screening model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--job-code',
            type=str,
            required=True,
            help='Job code (e.g., AGLO, OAIV)'
        )
        parser.add_argument(
            '--job-data-dir',
            type=str,
            required=True,
            help='Directory containing job application DOCX files'
        )
        parser.add_argument(
            '--labels-file',
            type=str,
            help='Path to JSON file with labels for training data'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='./trained_models',
            help='Directory to save trained models'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help='Batch size for training'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=10,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=2e-5,
            help='Learning rate'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Proportion of data for validation'
        )

    def handle(self, *args, **options):
        job_code = options['job_code']
        job_data_dir = options['job_data_dir']
        labels_file = options.get('labels_file')
        output_dir = options['output_dir']
        batch_size = options['batch_size']
        epochs = options['epochs']
        learning_rate = options['learning_rate']
        test_size = options['test_size']

        self.stdout.write(self.style.SUCCESS(f"Starting training for job: {job_code}"))

        # Check if job data directory exists
        if not os.path.exists(job_data_dir):
            raise CommandError(f"Job data directory not found: {job_data_dir}")

        # Check if labels file exists
        if labels_file and not os.path.exists(labels_file):
            raise CommandError(f"Labels file not found: {labels_file}")

        # Prepare data
        self.stdout.write("Preparing training data...")
        data_prep = DataPreparation()

        # Load and prepare dataset
        df = data_prep.prepare_training_dataset(
            job_data_dir=job_data_dir,
            job_code=job_code,
            labels_file=labels_file
        )

        self.stdout.write(f"Loaded {len(df)} applicants")

        # Check label distribution
        label_dist = data_prep.get_label_distribution(df)
        self.stdout.write(f"Label distribution: {label_dist}")

        if len(label_dist) == 0:
            raise CommandError(
                "No labeled data found. Please provide a labels file with training labels."
            )

        # Split data
        self.stdout.write("Splitting data into train/validation sets...")
        train_df, val_df = data_prep.create_train_val_split(
            df,
            test_size=test_size
        )

        self.stdout.write(f"Training samples: {len(train_df)}")
        self.stdout.write(f"Validation samples: {len(val_df)}")

        # Initialize model
        self.stdout.write("Initializing model...")
        model = ResumeScreeningModel()

        # Create datasets
        train_dataset = ResumeDataset(
            texts=train_df['text'].tolist(),
            labels=[model.LABEL_TO_ID[label] for label in train_df['label']],
            tokenizer=model.tokenizer
        )

        val_dataset = ResumeDataset(
            texts=val_df['text'].tolist(),
            labels=[model.LABEL_TO_ID[label] for label in val_df['label']],
            tokenizer=model.tokenizer
        )

        # Create output directory
        model_output_dir = os.path.join(output_dir, job_code)
        os.makedirs(model_output_dir, exist_ok=True)

        # Initialize trainer
        self.stdout.write("Initializing trainer...")
        trainer = ResumeScreeningTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
            output_dir=model_output_dir
        )

        # Train
        self.stdout.write(self.style.SUCCESS("Starting training..."))
        history = trainer.train()

        # Save configuration
        config = {
            'job_code': job_code,
            'training_samples': len(train_df),
            'validation_samples': len(val_df),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'label_distribution': label_dist,
            'final_metrics': {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
                'val_accuracy': history['val_accuracy'][-1],
                'val_f1': history['val_f1'][-1]
            }
        }

        config_path = os.path.join(model_output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f"Training completed! Model saved to {model_output_dir}"
        ))
