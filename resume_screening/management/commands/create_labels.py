"""
Django management command to create a labels template file for training data
"""
import os
import json
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Create a labels template file for training data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--job-data-dir',
            type=str,
            required=True,
            help='Directory containing job application DOCX files'
        )
        parser.add_argument(
            '--output-file',
            type=str,
            default='labels.json',
            help='Path to save labels template file'
        )

    def handle(self, *args, **options):
        job_data_dir = options['job_data_dir']
        output_file = options['output_file']

        # Check if directory exists
        if not os.path.exists(job_data_dir):
            raise CommandError(f"Directory not found: {job_data_dir}")

        # Get all DOCX files
        docx_files = [
            f for f in os.listdir(job_data_dir)
            if f.endswith('_OCR.docx')
        ]

        if not docx_files:
            raise CommandError(f"No *_OCR.docx files found in {job_data_dir}")

        # Create labels template
        labels = {}
        for filename in sorted(docx_files):
            labels[filename] = None  # User needs to fill this in

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(labels, f, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f"Created labels template with {len(docx_files)} files: {output_file}"
        ))
        self.stdout.write("\nPlease edit this file and assign one of these labels to each resume:")
        self.stdout.write("  - LIKELY_QUALIFIED")
        self.stdout.write("  - NEEDS_REVIEW")
        self.stdout.write("  - LIKELY_NOT_QUALIFIED")
        self.stdout.write("\nExample:")
        self.stdout.write(json.dumps({
            "AG LOAN OFF_Applicant 01_OCR.docx": "LIKELY_QUALIFIED",
            "AG LOAN OFF_Applicant 02_OCR.docx": "LIKELY_NOT_QUALIFIED",
            "AG LOAN OFF_Applicant 03_OCR.docx": "NEEDS_REVIEW"
        }, indent=2))
