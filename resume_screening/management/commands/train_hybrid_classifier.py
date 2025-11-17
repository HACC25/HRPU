"""
Django management command to train the hybrid classifier for resume screening.

Usage:
    python manage.py train_hybrid_classifier --job-code AGLO --job-data-dir AGLO_JobData \
        --classifications-csv AGLO_JobData/__ALGO_classifications.csv
"""
from django.core.management.base import BaseCommand
import pandas as pd
from io import StringIO
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from resume_screening.ml.hybrid_classifier import HybridResumeClassifier
from resume_screening.ml.multitask_model import MultiTaskResumeModel
from resume_screening.utils.document_processor import DocumentProcessor
from resume_screening.utils.job_config import get_job_config


class Command(BaseCommand):
    help = 'Train hybrid classifier for resume screening using requirements + ML'

    def add_arguments(self, parser):
        parser.add_argument('--job-code', type=str, required=True, help='Job code (e.g., AGLO, OAIV)')
        parser.add_argument('--job-data-dir', type=str, required=True, help='Directory with resume files')
        parser.add_argument('--classifications-csv', type=str, required=True, help='CSV with labels and requirements')
        parser.add_argument('--output-dir', type=str, default='trained_models', help='Output directory for model')
        parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds (0 = no CV, ignored if test-size used)')
        parser.add_argument('--test-size', type=float, help='Fraction of data to use as test set (e.g., 0.2 for 20%)')
        parser.add_argument('--bert-model', type=str, help='Path to trained BERT model for embeddings')
        parser.add_argument('--file-prefix', type=str, help='File prefix for resume files (defaults to config value)')

    def handle(self, *args, **options):
        job_code = options['job_code']
        job_data_dir = options['job_data_dir']
        classifications_csv = options['classifications_csv']
        output_dir = options['output_dir']
        cv_folds = options['cv_folds']
        test_size = options.get('test_size')
        bert_model_path = options.get('bert_model')

        # Load job configuration
        job_config = get_job_config(job_code)

        self.stdout.write("="*80)
        self.stdout.write(self.style.SUCCESS(f"TRAINING HYBRID CLASSIFIER FOR {job_code}"))
        self.stdout.write(f"Job: {job_config.job_name}")
        self.stdout.write(f"Requirements: {job_config.num_requirements}, Questions: {job_config.num_questions}")
        self.stdout.write("="*80)

        # 1. Load classification data
        self.stdout.write("\n1. Loading classification data...")
        labels, requirement_vectors, applicant_nums = self._load_classifications(classifications_csv, job_config)
        self.stdout.write(f"   Loaded {len(labels)} samples")

        # 2. Load resume texts
        self.stdout.write("\n2. Loading resume texts...")
        processor = DocumentProcessor()
        file_prefix = options.get('file_prefix') or job_config.file_prefix
        texts = self._load_resumes(file_prefix, job_data_dir, applicant_nums, processor)
        self.stdout.write(f"   Loaded {len(texts)} resume texts")

        # 3. Load BERT model for embeddings
        self.stdout.write("\n3. Loading BERT model for embeddings...")
        bert_model = MultiTaskResumeModel(
            model_name='sentence-transformers/all-mpnet-base-v2',
            num_questions=job_config.num_questions,
            num_requirements=job_config.num_requirements
        )

        if bert_model_path and os.path.exists(bert_model_path):
            self.stdout.write(f"   Loading weights from {bert_model_path}")
            bert_model.model.load_model(bert_model_path, device=bert_model.device)
        else:
            self.stdout.write("   Using pretrained BERT weights (no fine-tuning)")

        # 4. Compute embeddings
        self.stdout.write("\n4. Computing BERT embeddings...")
        all_embeddings = bert_model.get_embedding(texts)
        self.stdout.write(f"   Embeddings shape: {all_embeddings.shape}")

        # 5. Train hybrid classifier
        if test_size:
            train_data, test_data = self._train_test_split(
                texts, requirement_vectors, labels, all_embeddings, bert_model, test_size, applicant_nums, job_config
            )
            # Use ONLY training data for final model (no data leak!)
            final_texts, final_req_vecs, final_labels, final_embeddings = train_data
        elif cv_folds > 0:
            self._train_with_cv(texts, requirement_vectors, labels, all_embeddings, bert_model, cv_folds, job_config)
            # WARNING: CV mode trains on full dataset for final model
            self.stdout.write(self.style.WARNING("\nWARNING: CV mode will train final model on FULL dataset"))
            self.stdout.write(self.style.WARNING("         This is appropriate for model deployment but creates data leak for evaluation"))
            final_texts, final_req_vecs, final_labels, final_embeddings = texts, requirement_vectors, labels, all_embeddings
        else:
            self._train_single(texts, requirement_vectors, labels, all_embeddings, bert_model, job_config)
            # Single mode uses full dataset
            final_texts, final_req_vecs, final_labels, final_embeddings = texts, requirement_vectors, labels, all_embeddings

        # 6. Save final model
        self.stdout.write(f"\n6. Saving model to {output_dir}/{job_code}_hybrid/...")
        os.makedirs(f"{output_dir}/{job_code}_hybrid", exist_ok=True)

        hybrid = HybridResumeClassifier(
            num_requirements=job_config.num_requirements,
            qualified_threshold=job_config.get_qualified_threshold(),
            not_qualified_threshold=job_config.get_not_qualified_threshold()
        )
        hybrid.bert_model = bert_model
        hybrid.train_ml_classifier(final_texts, final_req_vecs, final_labels, final_embeddings)
        hybrid.save(f"{output_dir}/{job_code}_hybrid")

        self.stdout.write(self.style.SUCCESS("\nTraining complete!"))
        self.stdout.write("="*80)

    def _load_classifications(self, csv_path, job_config):
        """Load classifications from CSV - supports both AGLO and OAIV formats"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = [line.rstrip(',\n\r ') for line in f]
        csv_content = '\n'.join(lines)
        df = pd.read_csv(StringIO(csv_content))

        label_map = {'L': 'LIKELY_QUALIFIED', 'NR': 'NEEDS_REVIEW', 'NL': 'LIKELY_NOT_QUALIFIED'}
        yn_map = {'Y': 1, 'N': 0, '?': 0}  # ? counts as not met for safety

        # Job-specific column mappings
        if 'App #' in df.columns:
            # AGLO format
            app_col = 'App #'
            label_col = 'L/NL/NR'
            req_cols = ['Basic', '1y credit', '2y farm business', '2y ag loan', 'Sup Ap']
        elif 'App' in df.columns:
            # OAIV format
            app_col = 'App'
            label_col = 'Likely (L) or Not (NL)'
            req_cols = ['Typing (>= 40)', '6m Basic', '2y Clerical', 'Super Ap']
        else:
            raise ValueError(f"Unknown CSV format. Expected 'App #' or 'App' column, found: {df.columns.tolist()}")

        labels = []
        requirement_vectors = []
        applicant_nums = []

        for _, row in df.iterrows():
            try:
                app_num = int(row[app_col])
            except (ValueError, TypeError):
                continue

            # Parse label
            label_val = str(row[label_col]).strip().upper()
            if label_val in label_map:
                labels.append(label_map[label_val])
            else:
                # Default for unclear labels
                labels.append('NEEDS_REVIEW')

            # Parse requirements
            req_vector = []
            for col in req_cols:
                val = str(row[col]).strip().upper()
                req_vector.append(yn_map.get(val, 0))

            requirement_vectors.append(req_vector)
            applicant_nums.append(app_num)

        return labels, requirement_vectors, applicant_nums

    def _load_resumes(self, file_prefix, job_data_dir, applicant_nums, processor):
        """Load resume texts"""
        texts = []
        for app_num in applicant_nums:
            filename = f"{job_data_dir}/{file_prefix}*Applicant {app_num:02d}_OCR.docx"

            # Try to find file with glob pattern
            import glob
            matches = glob.glob(filename)

            if matches:
                text = processor.extract_text(matches[0])
                text = processor.clean_text(text)
                texts.append(text)
            else:
                self.stdout.write(self.style.WARNING(f"   Warning: File not found for applicant {app_num}"))
                texts.append("")

        return texts

    def _train_with_cv(self, texts, requirement_vectors, labels, embeddings, bert_model, cv_folds, job_config):
        """Train with cross-validation"""
        self.stdout.write(f"\n5. Running {cv_folds}-fold stratified cross-validation...")
        self.stdout.write("="*80)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels), 1):
            self.stdout.write(f"\nFold {fold_idx}:")
            self.stdout.write(f"   Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

            # Split data
            train_texts = [texts[i] for i in train_idx]
            train_req_vecs = [requirement_vectors[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            train_embeddings = embeddings[train_idx]

            test_texts = [texts[i] for i in test_idx]
            test_req_vecs = [requirement_vectors[i] for i in test_idx]
            test_labels = [labels[i] for i in test_idx]
            test_embeddings = embeddings[test_idx]

            # Train hybrid classifier
            hybrid = HybridResumeClassifier(
                num_requirements=job_config.num_requirements,
                qualified_threshold=job_config.get_qualified_threshold(),
                not_qualified_threshold=job_config.get_not_qualified_threshold()
            )
            hybrid.bert_model = bert_model
            hybrid.train_ml_classifier(train_texts, train_req_vecs, train_labels, train_embeddings)

            # Evaluate
            results = hybrid.evaluate(test_texts, test_req_vecs, test_labels, test_embeddings)
            fold_results.append(results)

            self.stdout.write(f"   Accuracy: {results['accuracy']:.1%}")
            self.stdout.write(f"   Macro F1: {results['macro_f1']:.4f}")
            self.stdout.write(f"   Method Distribution: {results['method_counts']}")

        # Aggregate results
        self.stdout.write("\n" + "="*80)
        self.stdout.write("CROSS-VALIDATION SUMMARY")
        self.stdout.write("="*80)

        accuracies = [r['accuracy'] for r in fold_results]
        macro_f1s = [r['macro_f1'] for r in fold_results]

        self.stdout.write(f"\nAccuracy:    {np.mean(accuracies):.1%} +/- {np.std(accuracies):.1%}")
        self.stdout.write(f"Macro F1:    {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")

    def _train_single(self, texts, requirement_vectors, labels, embeddings, bert_model, job_config):
        """Train on full dataset"""
        self.stdout.write("\n5. Training on full dataset...")

        hybrid = HybridResumeClassifier(
            num_requirements=job_config.num_requirements,
            qualified_threshold=job_config.get_qualified_threshold(),
            not_qualified_threshold=job_config.get_not_qualified_threshold()
        )
        hybrid.bert_model = bert_model
        hybrid.train_ml_classifier(texts, requirement_vectors, labels, embeddings)

        # Evaluate on training set
        results = hybrid.evaluate(texts, requirement_vectors, labels, embeddings)
        self.stdout.write(f"   Training Accuracy: {results['accuracy']:.1%}")
        self.stdout.write(f"   Training Macro F1: {results['macro_f1']:.4f}")

    def _train_test_split(self, texts, requirement_vectors, labels, embeddings, bert_model, test_size, applicant_nums, job_config):
        """Train with a fixed train/test split"""
        self.stdout.write(f"\n5. Training with {test_size:.0%} test split...")
        self.stdout.write("="*80)

        # Stratified split
        indices = np.arange(len(texts))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=42
        )

        # Split data
        train_texts = [texts[i] for i in train_idx]
        train_req_vecs = [requirement_vectors[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_embeddings = embeddings[train_idx]
        train_app_nums = [applicant_nums[i] for i in train_idx]

        test_texts = [texts[i] for i in test_idx]
        test_req_vecs = [requirement_vectors[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        test_embeddings = embeddings[test_idx]
        test_app_nums = [applicant_nums[i] for i in test_idx]

        self.stdout.write(f"\nTrain set: {len(train_idx)} samples (applicants: {sorted(train_app_nums)})")
        self.stdout.write(f"Test set:  {len(test_idx)} samples (applicants: {sorted(test_app_nums)})")

        # Train hybrid classifier
        hybrid = HybridResumeClassifier(
            num_requirements=job_config.num_requirements,
            qualified_threshold=job_config.get_qualified_threshold(),
            not_qualified_threshold=job_config.get_not_qualified_threshold()
        )
        hybrid.bert_model = bert_model
        hybrid.train_ml_classifier(train_texts, train_req_vecs, train_labels, train_embeddings)

        # Evaluate on test set
        self.stdout.write("\n" + "="*80)
        self.stdout.write("TEST SET EVALUATION")
        self.stdout.write("="*80)

        results = hybrid.evaluate(test_texts, test_req_vecs, test_labels, test_embeddings)

        self.stdout.write(f"\nTest Accuracy: {results['accuracy']:.1%}")
        self.stdout.write(f"Test Macro F1: {results['macro_f1']:.4f}")
        self.stdout.write(f"Method Distribution: {results['method_counts']}")

        # Show per-sample results
        predictions = hybrid.predict(test_texts, test_req_vecs, test_embeddings)
        self.stdout.write("\nDetailed Test Results:")
        for i, (app_num, pred, true_label) in enumerate(zip(test_app_nums, predictions, test_labels)):
            correct = "✓" if pred['classification'] == true_label else "✗"
            method = "R" if pred['method'] == 'rule-based' else "ML"
            self.stdout.write(
                f"  App {app_num:2d}: {pred['classification']:25s} (true: {true_label:25s}) "
                f"{correct} [{method}, conf={pred['confidence']:.2f}]"
            )

        return (train_texts, train_req_vecs, train_labels, train_embeddings), \
               (test_texts, test_req_vecs, test_labels, test_embeddings)
