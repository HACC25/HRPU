"""
Helper script to create sample labels for testing the training process
This creates a labels.json file with some sample labels so you can test training immediately
"""
import json
import os

def create_sample_labels(job_data_dir, output_file):
    """Create sample labels for quick testing"""

    # Get all OCR docx files
    if not os.path.exists(job_data_dir):
        print(f"Error: Directory not found: {job_data_dir}")
        return

    docx_files = sorted([
        f for f in os.listdir(job_data_dir)
        if f.endswith('_OCR.docx')
    ])

    if not docx_files:
        print(f"Error: No *_OCR.docx files found in {job_data_dir}")
        return

    print(f"Found {len(docx_files)} applicant files")
    print("\nCreating sample labels for quick testing...")
    print("NOTE: These are DUMMY labels for testing only!")
    print("For real training, you MUST manually review and label each application.\n")

    # Create sample labels - distribute evenly across classes
    labels = {}

    for i, filename in enumerate(docx_files):
        # Distribute labels evenly for balanced dataset
        if i % 3 == 0:
            label = "LIKELY_QUALIFIED"
        elif i % 3 == 1:
            label = "NEEDS_REVIEW"
        else:
            label = "LIKELY_NOT_QUALIFIED"

        labels[filename] = label
        print(f"  {filename}: {label}")

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"\n✓ Saved sample labels to: {output_file}")
    print(f"  Total: {len(labels)} files")
    print(f"  LIKELY_QUALIFIED: {sum(1 for v in labels.values() if v == 'LIKELY_QUALIFIED')}")
    print(f"  NEEDS_REVIEW: {sum(1 for v in labels.values() if v == 'NEEDS_REVIEW')}")
    print(f"  LIKELY_NOT_QUALIFIED: {sum(1 for v in labels.values() if v == 'LIKELY_NOT_QUALIFIED')}")

    print("\n⚠️  IMPORTANT:")
    print("  These are SAMPLE labels for testing the training pipeline.")
    print("  For production use:")
    print("    1. Review each application carefully")
    print("    2. Assign accurate labels based on job requirements")
    print("    3. Ensure consistency in labeling criteria")
    print(f"\n  Edit {output_file} to change label values")


if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("Resume Screening AI - Sample Labels Generator")
    print("=" * 80)
    print()

    if len(sys.argv) > 1:
        job_code = sys.argv[1]
        if job_code.upper() == 'AGLO':
            create_sample_labels('AGLO_JobData', 'AGLO_labels.json')
        elif job_code.upper() == 'OAIV':
            create_sample_labels('OAIV_JobData', 'OAIV_labels.json')
        else:
            print(f"Unknown job code: {job_code}")
            print("Usage: python create_sample_labels.py [AGLO|OAIV]")
    else:
        print("Creating sample labels for both job positions...\n")

        print("1. AG Loan Officer (AGLO)")
        print("-" * 40)
        create_sample_labels('AGLO_JobData', 'AGLO_labels.json')

        print("\n2. Office Assistant IV (OAIV)")
        print("-" * 40)
        create_sample_labels('OAIV_JobData', 'OAIV_labels.json')

        print("\n" + "=" * 80)
        print("Done! You can now train models using these sample labels.")
        print("=" * 80)
