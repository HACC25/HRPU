# HRPU - AI Resume Screening System

An intelligent Django-based web application that automates resume screening for government job positions using hybrid machine learning classification.

## Overview

HRPU streamlines the hiring process by automatically analyzing job applications against specific position requirements. The system combines rule-based logic with deep learning (BERT embeddings) to classify applicants as "Likely Qualified," "Needs Review," or "Likely Not Qualified."

## Features

- **Automated Resume Screening**: Upload resumes (PDF/DOCX) and get instant AI-powered evaluations
- **Multi-Position Support**: Configure and manage multiple job positions with unique requirements
- **Hybrid ML Classification**: Combines requirement matching with sentence-transformer embeddings
- **Web Dashboard**: Filter, sort, and review applicants with detailed screening results
- **Human Review System**: Override AI decisions and add reviewer notes
- **Training Pipeline**: Train custom models on labeled data for each position

## Tech Stack

- **Backend**: Django 5.1+
- **ML/AI**: PyTorch, Sentence Transformers (BERT), scikit-learn
- **Document Processing**: python-docx, PyPDF2
- **Database**: SQLite (default)

## Project Structure

```
HRPU/
├── HRPU/                          # Django project settings
├── resume_screening/              # Main application
│   ├── models.py                  # Database models (Applicant, JobPosition, ScreeningResult)
│   ├── views.py                   # Web views (dashboard, upload, review)
│   ├── services.py                # Resume screening service
│   ├── ml/                        # Machine learning modules
│   │   ├── hybrid_classifier.py   # Hybrid classification system
│   │   └── multitask_model.py     # BERT-based multi-task model
│   ├── utils/                     # Utilities
│   │   ├── document_processor.py  # PDF/DOCX text extraction
│   │   └── job_config.py          # Job configuration loader
│   └── management/commands/       # CLI commands
│       ├── train_hybrid_classifier.py
│       └── screen_resumes_hybrid.py
├── templates/                     # HTML templates
├── trained_models/                # Stored ML models
├── media/                         # Uploaded files
├── job_configs.json               # Job position configurations
└── *_requirement_labels.json      # Labeled requirement data
```

## Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd HRPU
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run migrations**
```bash
python manage.py migrate
```

5. **Create superuser** (optional)
```bash
python manage.py createsuperuser
```

## Usage

### Start Development Server
```bash
python manage.py runserver
```

Navigate to `http://localhost:8000/` to access the dashboard.

### Train a Model
```bash
python manage.py train_hybrid_classifier <JOB_CODE>
```
Example: `python manage.py train_hybrid_classifier AGLO`

### Screen Resumes (Batch)
```bash
python manage.py screen_resumes_hybrid <JOB_CODE> <RESUMES_DIR>
```

### Web Interface
- **Dashboard** (`/`): View and filter applicants by position
- **Upload** (`/upload/`): Submit new applications
- **Review** (`/review/<id>/`): View detailed screening results and make decisions

## Job Configuration

Jobs are defined in `job_configs.json`:
```json
{
  "AGLO": {
    "job_name": "Agricultural Loan Officer",
    "num_requirements": 5,
    "num_questions": 13,
    "requirements": ["basic", "credit_analysis_1y", "farm_business_2y", ...],
    "hybrid_thresholds": {
      "qualified": 5,
      "not_qualified": 2
    }
  }
}
```

## How It Works

1. **Document Upload**: User uploads resume (PDF/DOCX)
2. **Text Extraction**: System extracts and cleans text from document
3. **BERT Embedding**: Generate semantic embeddings using sentence-transformers
4. **Requirement Matching**: Predict which job requirements the applicant meets
5. **Hybrid Classification**: Combine requirement counts with ML confidence scores
6. **Result Display**: Show classification, confidence, and requirement breakdown

## Models

### Hybrid Classifier
Combines:
- **Rule-based**: Counts met requirements against thresholds
- **ML-based**: Uses BERT embeddings + logistic regression
- **Weighted decision**: Prioritizes rules for clear cases, ML for borderline

### Multi-Task BERT Model
- Fine-tuned sentence transformer for resume analysis
- Predicts: classification label + individual requirements
- Base model: `sentence-transformers/all-mpnet-base-v2`

## Database Models

- **JobPosition**: Job details, requirements, evaluation criteria
- **Applicant**: Candidate info, uploaded resume, extracted text
- **ScreeningResult**: AI classification, confidence, criteria scores, human review
- **TrainingData**: Labeled examples for model training
- **ModelVersion**: Track trained model versions and performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Current Jobs

- **AGLO**: Agricultural Loan Officer 
- **OAIV**: Office Assistant IV 
