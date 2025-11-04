# Machine Learning - Resume Screening System


### Database Models
The ML infrastructure includes 5 Django models for managing the resume screening pipeline:

1. **JobPosition** - Stores job posting details and evaluation criteria
2. **Applicant** - Manages applicant information and uploaded resumes
3. **ScreeningResult** - Stores AI screening classifications (Likely Qualified, Needs Review, Likely Not Qualified)
4. **TrainingData** - Manages labeled training data for model training
5. **ModelVersion** - Tracks different versions of trained models with metrics

### ML Components
- **Model**: PyTorch-based resume screening classifier
- **Trainer**: Training pipeline with validation and metrics
- **Inference**: Resume screening with confidence scores and explanations
- **Data Preparation**: Text extraction and preprocessing utilities

## Quick Start

### 1. Install Dependencies
*uncomment extra index in the txt for pytorch with cuda support*
```bash
pip install -r requirements.txt
```

### 2. Run Database Migrations
```bash
python manage.py migrate
```

## Training Workflow

### Step 1: Create Sample Labels (for testing)
Generate dummy labels for quick testing:

```bash
# Create labels for both job positions
python create_sample_labels.py

# Or for a specific job
python create_sample_labels.py AGLO
python create_sample_labels.py OAIV
```

This creates `AGLO_labels.json` and `OAIV_labels.json` with balanced sample labels.

**⚠️ For production**: Manually review each application and edit the JSON files with accurate labels.

### Step 2: Train the Model
Train a model for a specific job position:

```bash
python manage.py train_model \
  --job-code AGLO \
  --job-data-dir AGLO_JobData \
  --labels-file AGLO_labels.json \
  --output-dir ./trained_models \
  --epochs 10 \
  --batch-size 8
```

**Options:**
- `--job-code`: Job position code (e.g., AGLO, OAIV)
- `--job-data-dir`: Directory with applicant DOCX files
- `--labels-file`: JSON file with training labels
- `--output-dir`: Where to save trained model (default: `./trained_models`)
- `--epochs`: Training epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--test-size`: Validation split (default: 0.2)

The trained model will be saved to `./trained_models/AGLO/`

## Screening Resumes

### Screen a Single Resume
```bash
python manage.py screen_resumes \
  --job-code AGLO \
  --model-path ./trained_models/AGLO \
  --resume-file path/to/resume.docx
```

### Screen Multiple Resumes
```bash
python manage.py screen_resumes \
  --job-code AGLO \
  --model-path ./trained_models/AGLO \
  --resume-dir path/to/resumes/ \
  --output-file screening_results.json
```

**Options:**
- `--job-code`: Job position code
- `--model-path`: Path to trained model directory
- `--resume-file`: Single resume to screen
- `--resume-dir`: Directory with multiple resumes (PDF or DOCX)
- `--output-file`: Save results to JSON file (optional)

### Output Format
Results include:
- Classification: `LIKELY_QUALIFIED`, `NEEDS_REVIEW`, or `LIKELY_NOT_QUALIFIED`
- Confidence score (0-100%)
- Explanation of the decision

## Example Workflow

```bash
# 1. Create sample labels
python create_sample_labels.py AGLO

# 2. Train model
python manage.py train_model \
  --job-code AGLO \
  --job-data-dir AGLO_JobData \
  --labels-file AGLO_labels.json \
  --epochs 10

# 3. Screen new applications
python manage.py screen_resumes \
  --job-code AGLO \
  --model-path ./trained_models/AGLO \
  --resume-dir new_applications/ \
  --output-file results.json
```
