"""
Django management command to screen resumes using trained model
"""
import os
import json
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from resume_screening.utils.document_processor import DocumentProcessor
from resume_screening.ml.inference import ResumeScreener


class Command(BaseCommand):
    help = 'Screen resumes using trained model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--job-code',
            type=str,
            required=True,
            help='Job code (e.g., AGLO, OAIV)'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            required=True,
            help='Path to trained model weights'
        )
        parser.add_argument(
            '--resume-file',
            type=str,
            help='Path to a single resume file to screen'
        )
        parser.add_argument(
            '--resume-dir',
            type=str,
            help='Directory containing multiple resume files to screen'
        )
        parser.add_argument(
            '--output-file',
            type=str,
            help='Path to save screening results (JSON format)'
        )

    def handle(self, *args, **options):
        job_code = options['job_code']
        model_path = options['model_path']
        resume_file = options.get('resume_file')
        resume_dir = options.get('resume_dir')
        output_file = options.get('output_file')

        # Check that either resume_file or resume_dir is provided
        if not resume_file and not resume_dir:
            raise CommandError("Either --resume-file or --resume-dir must be provided")

        # Check model exists
        if not os.path.exists(model_path):
            raise CommandError(f"Model not found: {model_path}")

        self.stdout.write(self.style.SUCCESS(f"Screening resumes for job: {job_code}"))

        # Initialize screener
        self.stdout.write("Loading model...")
        screener = ResumeScreener(
            model_path=model_path,
            job_code=job_code
        )

        # Initialize document processor
        processor = DocumentProcessor()

        results = []

        # Screen single file
        if resume_file:
            if not os.path.exists(resume_file):
                raise CommandError(f"Resume file not found: {resume_file}")

            self.stdout.write(f"Screening: {resume_file}")

            # Extract text
            text = processor.extract_text(resume_file)
            text = processor.clean_text(text)

            # Screen
            result = screener.screen_resume(text)
            result['filename'] = os.path.basename(resume_file)

            results.append(result)

            # Print result
            self._print_result(result)

        # Screen directory
        elif resume_dir:
            if not os.path.exists(resume_dir):
                raise CommandError(f"Resume directory not found: {resume_dir}")

            # Get all DOCX files
            resume_files = [
                f for f in os.listdir(resume_dir)
                if f.endswith('.docx') or f.endswith('.pdf')
            ]

            self.stdout.write(f"Found {len(resume_files)} resumes to screen")

            for filename in resume_files:
                file_path = os.path.join(resume_dir, filename)

                try:
                    self.stdout.write(f"\nScreening: {filename}")

                    # Extract text
                    text = processor.extract_text(file_path)
                    text = processor.clean_text(text)

                    # Screen
                    result = screener.screen_resume(text)
                    result['filename'] = filename

                    results.append(result)

                    # Print result
                    self._print_result(result)

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"Error processing {filename}: {str(e)}")
                    )

        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.stdout.write(self.style.SUCCESS(f"\nResults saved to: {output_file}"))

        # Print summary
        self._print_summary(results)

    def _print_result(self, result):
        """Print screening result"""
        classification = result['classification']
        confidence = result['confidence']

        if classification == 'LIKELY_QUALIFIED':
            style = self.style.SUCCESS
        elif classification == 'NEEDS_REVIEW':
            style = self.style.WARNING
        else:
            style = self.style.ERROR

        self.stdout.write(style(f"  Classification: {classification}"))
        self.stdout.write(f"  Confidence: {confidence:.1%}")
        self.stdout.write(f"  Explanation: {result['explanation']}")

    def _print_summary(self, results):
        """Print summary of screening results"""
        self.stdout.write("\n" + "="*80)
        self.stdout.write(self.style.SUCCESS("SCREENING SUMMARY"))
        self.stdout.write("="*80)

        total = len(results)
        qualified = sum(1 for r in results if r['classification'] == 'LIKELY_QUALIFIED')
        needs_review = sum(1 for r in results if r['classification'] == 'NEEDS_REVIEW')
        not_qualified = sum(1 for r in results if r['classification'] == 'LIKELY_NOT_QUALIFIED')

        self.stdout.write(f"Total resumes screened: {total}")
        self.stdout.write(self.style.SUCCESS(f"Likely Qualified: {qualified} ({qualified/total*100:.1f}%)"))
        self.stdout.write(self.style.WARNING(f"Needs Review: {needs_review} ({needs_review/total*100:.1f}%)"))
        self.stdout.write(self.style.ERROR(f"Likely Not Qualified: {not_qualified} ({not_qualified/total*100:.1f}%)"))
