"""
Django management command to screen resumes using the hybrid classifier.

Usage:
    python manage.py screen_resumes_hybrid --job-code AGLO --job-data-dir AGLO_JobData \
        --model-dir trained_models/AGLO_hybrid --requirement-labels AGLO_requirement_labels.json
"""
from django.core.management.base import BaseCommand
import json
import os

from resume_screening.ml.hybrid_classifier import HybridResumeClassifier
from resume_screening.ml.multitask_model import MultiTaskResumeModel
from resume_screening.utils.document_processor import DocumentProcessor
from resume_screening.utils.job_config import get_job_config


class Command(BaseCommand):
    help = 'Screen resumes using trained hybrid classifier'

    def add_arguments(self, parser):
        parser.add_argument('--job-code', type=str, required=True, help='Job code (e.g., AGLO, OAIV)')
        parser.add_argument('--job-data-dir', type=str, required=True, help='Directory with resume files')
        parser.add_argument('--model-dir', type=str, required=True, help='Directory with trained hybrid model')
        parser.add_argument('--requirement-labels', type=str, required=True, help='JSON with requirement vectors')
        parser.add_argument('--bert-model', type=str, help='Path to trained BERT model for embeddings')
        parser.add_argument('--output-json', type=str, help='Output JSON file for predictions')
        parser.add_argument('--file-prefix', type=str, help='File prefix for resume files (defaults to config)')
        parser.add_argument('--ground-truth-csv', type=str, help='CSV with ground truth labels for evaluation')

    def handle(self, *args, **options):
        job_code = options['job_code']
        job_data_dir = options['job_data_dir']
        model_dir = options['model_dir']
        requirement_labels_path = options['requirement_labels']
        bert_model_path = options.get('bert_model')
        output_json = options.get('output_json')

        # Load job configuration
        job_config = get_job_config(job_code)

        self.stdout.write("="*80)
        self.stdout.write(self.style.SUCCESS(f"SCREENING RESUMES FOR {job_code} (HYBRID CLASSIFIER)"))
        self.stdout.write(f"Job: {job_config.job_name}")
        self.stdout.write(f"Requirements: {job_config.num_requirements}, Questions: {job_config.num_questions}")
        self.stdout.write("="*80)

        # 1. Load requirement labels
        self.stdout.write("\n1. Loading requirement labels...")
        with open(requirement_labels_path, 'r', encoding='utf-8') as f:
            requirement_labels = json.load(f)
        self.stdout.write(f"   Loaded requirements for {len(requirement_labels)} applicants")

        # 2. Load BERT model
        self.stdout.write("\n2. Loading BERT model...")
        bert_model = MultiTaskResumeModel(
            model_name='sentence-transformers/all-mpnet-base-v2',
            num_questions=job_config.num_questions,
            num_requirements=job_config.num_requirements
        )

        if bert_model_path and os.path.exists(bert_model_path):
            self.stdout.write(f"   Loading weights from {bert_model_path}")
            bert_model.model.load_model(bert_model_path, device=bert_model.device)
        else:
            self.stdout.write("   Using pretrained BERT weights")

        # 3. Load hybrid classifier
        self.stdout.write(f"\n3. Loading hybrid classifier from {model_dir}...")
        hybrid = HybridResumeClassifier.load(model_dir, bert_model=bert_model)

        # 4. Process resumes
        self.stdout.write("\n4. Screening resumes...")
        processor = DocumentProcessor()
        all_predictions = {}
        file_prefix = options.get('file_prefix') or job_config.file_prefix

        import glob
        import re

        for filename_or_num, req_vec in requirement_labels.items():
            # Extract applicant number from filename or use directly if it's a number
            if isinstance(filename_or_num, int):
                app_num = filename_or_num
            elif filename_or_num.isdigit():
                app_num = int(filename_or_num)
            else:
                # Extract number from filename like "AG LOAN OFF_Applicant 01_OCR.docx"
                match = re.search(r'Applicant[_ ](\d+)', filename_or_num)
                if match:
                    app_num = int(match.group(1))
                else:
                    self.stdout.write(self.style.WARNING(f"   Skipping {filename_or_num} (cannot parse applicant number)"))
                    continue

            # Find resume file
            pattern = f"{job_data_dir}/{file_prefix}*Applicant {app_num:02d}_OCR.docx"
            matches = glob.glob(pattern)

            if not matches:
                self.stdout.write(self.style.WARNING(f"   Skipping applicant {app_num} (file not found)"))
                continue

            # Extract and process text
            text = processor.extract_text(matches[0])
            text = processor.clean_text(text)

            # Get BERT embedding
            embedding = bert_model.get_embedding([text])

            # Predict using hybrid classifier
            predictions = hybrid.predict(
                texts=[text],
                requirement_vectors=[req_vec],
                bert_embeddings=embedding
            )

            result = predictions[0]
            all_predictions[app_num] = result

            # Display result
            method_symbol = "R" if result['method'] == 'rule-based' else "ML"
            self.stdout.write(
                f"   Applicant {app_num:2d}: {result['classification']:25s} "
                f"(conf={result['confidence']:.2f}, {method_symbol}, reqs={result['requirements_met']}/{job_config.num_requirements})"
            )

        # 5. Summary statistics
        self.stdout.write("\n" + "="*80)
        self.stdout.write("SCREENING SUMMARY")
        self.stdout.write("="*80)

        classifications = {}
        methods = {}

        for pred in all_predictions.values():
            cls = pred['classification']
            method = pred['method']
            classifications[cls] = classifications.get(cls, 0) + 1
            methods[method] = methods.get(method, 0) + 1

        self.stdout.write("\nClassification Distribution:")
        for cls, count in sorted(classifications.items()):
            pct = count / len(all_predictions) * 100
            self.stdout.write(f"  {cls:25s}: {count:3d} ({pct:5.1f}%)")

        self.stdout.write("\nMethod Distribution:")
        for method, count in sorted(methods.items()):
            pct = count / len(all_predictions) * 100
            self.stdout.write(f"  {method:15s}: {count:3d} ({pct:5.1f}%)")

        # 6. Evaluate against ground truth if provided
        ground_truth_csv = options.get('ground_truth_csv')
        if ground_truth_csv:
            self.stdout.write("\n" + "="*80)
            self.stdout.write("EVALUATION AGAINST GROUND TRUTH")
            self.stdout.write("="*80)
            self._evaluate_predictions(all_predictions, ground_truth_csv, job_config)

        # 7. Save to JSON if requested
        if output_json:
            self.stdout.write(f"\n{'7' if ground_truth_csv else '6'}. Saving predictions to {output_json}...")
            # Convert numpy types to native Python for JSON serialization
            serializable_predictions = {}
            for app_num, pred in all_predictions.items():
                serializable_predictions[str(app_num)] = {
                    'classification': pred['classification'],
                    'confidence': float(pred['confidence']),
                    'method': pred['method'],
                    'requirements_met': int(pred['requirements_met']),
                    'requirements_vector': [int(x) for x in pred['requirements_vector']]
                }
                if 'ml_probabilities' in pred:
                    serializable_predictions[str(app_num)]['ml_probabilities'] = {
                        k: float(v) for k, v in pred['ml_probabilities'].items()
                    }

            with open(output_json, 'w') as f:
                json.dump(serializable_predictions, f, indent=2)

        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.SUCCESS("Screening complete!"))

    def _evaluate_predictions(self, predictions, ground_truth_csv, job_config):
        """Evaluate predictions against ground truth labels"""
        import pandas as pd
        from io import StringIO
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

        # Load ground truth
        with open(ground_truth_csv, 'r', encoding='utf-8') as f:
            lines = [line.rstrip(',\n\r ') for line in f]
        csv_content = '\n'.join(lines)
        df = pd.read_csv(StringIO(csv_content))

        # Parse ground truth labels
        label_map = {'L': 'LIKELY_QUALIFIED', 'NR': 'NEEDS_REVIEW', 'NL': 'LIKELY_NOT_QUALIFIED'}

        if 'App #' in df.columns:
            app_col = 'App #'
            label_col = 'L/NL/NR'
        elif 'App' in df.columns:
            app_col = 'App'
            label_col = 'Likely (L) or Not (NL)'
        else:
            self.stdout.write(self.style.ERROR("Unknown CSV format"))
            return

        ground_truth = {}
        for _, row in df.iterrows():
            try:
                app_num = int(row[app_col])
                label_val = str(row[label_col]).strip().upper()
                if label_val in label_map:
                    ground_truth[app_num] = label_map[label_val]
                else:
                    ground_truth[app_num] = 'NEEDS_REVIEW'  # Default
            except (ValueError, TypeError):
                continue

        # Compare predictions to ground truth
        y_true = []
        y_pred = []
        mismatches = []

        for app_num in sorted(predictions.keys()):
            if app_num not in ground_truth:
                continue

            true_label = ground_truth[app_num]
            pred_label = predictions[app_num]['classification']

            y_true.append(true_label)
            y_pred.append(pred_label)

            if true_label != pred_label:
                mismatches.append({
                    'app_num': app_num,
                    'true': true_label,
                    'pred': pred_label,
                    'reqs_met': predictions[app_num]['requirements_met'],
                    'method': predictions[app_num]['method'],
                    'confidence': predictions[app_num]['confidence']
                })

        if not y_true:
            self.stdout.write(self.style.WARNING("No matching applicants found for evaluation"))
            return

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)

        self.stdout.write(f"\nOverall Accuracy: {accuracy:.1%} ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})")

        # Confusion matrix
        labels = ['LIKELY_QUALIFIED', 'NEEDS_REVIEW', 'LIKELY_NOT_QUALIFIED']
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        self.stdout.write("\nConfusion Matrix:")
        self.stdout.write("                    Predicted →")
        self.stdout.write(f"                    {'LQ':>12} {'NR':>12} {'NL':>12}")
        for i, true_label in enumerate(['LQ', 'NR', 'NL']):
            row_str = f"  True {true_label:>12} | "
            for j in range(3):
                row_str += f"{cm[i][j]:>12} "
            self.stdout.write(row_str)

        # Per-class metrics
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        self.stdout.write("\nPer-Class Metrics:")
        self.stdout.write(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        for label in labels:
            short_label = {'LIKELY_QUALIFIED': 'LQ', 'NEEDS_REVIEW': 'NR', 'LIKELY_NOT_QUALIFIED': 'NL'}[label]
            metrics = report[label]
            self.stdout.write(
                f"  {short_label:<25} {metrics['precision']:>10.2%} {metrics['recall']:>10.2%} "
                f"{metrics['f1-score']:>10.2%} {int(metrics['support']):>10}"
            )

        # Show misclassifications
        if mismatches:
            self.stdout.write(f"\nMisclassifications ({len(mismatches)}):")
            for m in sorted(mismatches, key=lambda x: x['app_num']):
                true_short = {'LIKELY_QUALIFIED': 'LQ', 'NEEDS_REVIEW': 'NR', 'LIKELY_NOT_QUALIFIED': 'NL'}[m['true']]
                pred_short = {'LIKELY_QUALIFIED': 'LQ', 'NEEDS_REVIEW': 'NR', 'LIKELY_NOT_QUALIFIED': 'NL'}[m['pred']]
                self.stdout.write(
                    f"  App {m['app_num']:2d}: True={true_short:>3} Pred={pred_short:>3} "
                    f"(reqs={m['reqs_met']}/{job_config.num_requirements}, {m['method']}, conf={m['confidence']:.2f})"
                )
