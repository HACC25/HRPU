"""
Service functions for resume screening operations
"""
import os
import json
from typing import Dict, Optional
from django.conf import settings

from .models import Applicant, ScreeningResult, JobPosition
from .utils.document_processor import DocumentProcessor
from .utils.job_config import get_job_config
from .ml.hybrid_classifier import HybridResumeClassifier
from .ml.multitask_model import MultiTaskResumeModel


class ResumeScreeningService:
    """Service for screening uploaded resumes"""

    @staticmethod
    def screen_uploaded_resume(
        applicant: Applicant,
        requirement_labels_path: Optional[str] = None
    ) -> ScreeningResult:
        """
        Screen an uploaded resume and create a ScreeningResult

        Args:
            applicant: Applicant instance with uploaded file
            requirement_labels_path: Optional path to requirement labels JSON

        Returns:
            ScreeningResult instance
        """
        job_code = applicant.job_position.code
        job_config = get_job_config(job_code)

        # Auto-detect requirement labels path if not provided
        if requirement_labels_path is None:
            from pathlib import Path
            project_root = Path(settings.BASE_DIR)
            default_path = project_root / f"{job_code}_requirement_labels.json"
            if default_path.exists():
                requirement_labels_path = str(default_path)

        # 1. Extract text from document if not already done
        if not applicant.extracted_text:
            file_path = applicant.application_file.path
            applicant.extracted_text = DocumentProcessor.extract_text(file_path)
            applicant.extracted_text = DocumentProcessor.clean_text(applicant.extracted_text)
            applicant.save()

        # 2. Load requirement labels if provided
        requirement_vector = None
        if requirement_labels_path and os.path.exists(requirement_labels_path):
            with open(requirement_labels_path, 'r', encoding='utf-8') as f:
                requirement_labels = json.load(f)
                # Try to match by applicant ID
                requirement_vector = requirement_labels.get(str(applicant.pk))

                # If not found by PK, try to construct filename and match
                if requirement_vector is None:
                    # Try matching by filename pattern: "JobPrefix_Applicant ##_OCR.docx"
                    import re
                    for filename, req_vec in requirement_labels.items():
                        # Extract applicant number from filename
                        match = re.search(r'Applicant[_ ](\d+)', filename)
                        if match and int(match.group(1)) == applicant.pk:
                            requirement_vector = req_vec
                            break

        # 3. Load BERT model
        bert_model = MultiTaskResumeModel(
            model_name='sentence-transformers/all-mpnet-base-v2',
            num_questions=job_config.num_questions,
            num_requirements=job_config.num_requirements
        )

        # Try to load trained weights
        model_dir = settings.ML_MODELS_DIR / f"{job_code}_hybrid"
        bert_model_path = model_dir / "bert_model"

        if os.path.exists(bert_model_path):
            bert_model.model.load_model(str(bert_model_path), device=bert_model.device)

        # 4. Load hybrid classifier
        if os.path.exists(model_dir):
            hybrid = HybridResumeClassifier.load(str(model_dir), bert_model=bert_model)
        else:
            # Fallback: create default classifier
            hybrid = HybridResumeClassifier(
                num_requirements=job_config.num_requirements,
                qualified_threshold=job_config.hybrid_thresholds['qualified'],
                not_qualified_threshold=job_config.hybrid_thresholds['not_qualified']
            )
            hybrid.bert_model = bert_model

        # 5. Get BERT embeddings
        embeddings = bert_model.get_embedding([applicant.extracted_text])

        # 6. Predict requirements if not provided
        if requirement_vector is None:
            # Try to use BERT model to predict requirements
            try:
                req_preds = bert_model.predict_requirements([applicant.extracted_text])
                requirement_vector = req_preds[0].tolist()
            except (AttributeError, Exception) as e:
                # Fallback: assume all requirements not met (conservative approach)
                print(f"Warning: Could not predict requirements: {e}")
                print("Falling back to conservative estimate (all requirements not met)")
                requirement_vector = [0] * job_config.num_requirements

        # 7. Run hybrid classifier
        predictions = hybrid.predict(
            texts=[applicant.extracted_text],
            requirement_vectors=[requirement_vector],
            bert_embeddings=embeddings
        )

        pred = predictions[0]

        # 8. Generate explanation
        explanation = ResumeScreeningService._generate_explanation(
            classification=pred['classification'],
            requirements_met=pred['requirements_met'],
            requirement_vector=requirement_vector,
            job_config=job_config
        )

        # 9. Create ScreeningResult
        screening_result = ScreeningResult.objects.create(
            applicant=applicant,
            classification=pred['classification'],
            confidence_score=pred['confidence'],
            explanation=explanation,
            criteria_scores={
                'requirements_met': pred['requirements_met'],
                'requirements_vector': requirement_vector,
                'method': pred['method'],
                'ml_probabilities': pred.get('ml_probabilities', {})
            },
            model_version=f"{job_code}_hybrid_v1.0"
        )

        return screening_result

    @staticmethod
    def _generate_explanation(
        classification: str,
        requirements_met: int,
        requirement_vector: list,
        job_config
    ) -> str:
        """
        Generate human-readable explanation for classification

        Args:
            classification: Classification result
            requirements_met: Number of requirements met
            requirement_vector: Binary vector of requirements
            job_config: Job configuration object

        Returns:
            Explanation string
        """
        total_reqs = job_config.num_requirements
        req_names = job_config.requirements

        # List requirements that were met (with human-readable descriptions)
        met_requirements = [
            job_config.get_requirement_description(req_names[i])
            for i, val in enumerate(requirement_vector) if val == 1
        ]

        # List requirements that were not met (with human-readable descriptions)
        unmet_requirements = [
            job_config.get_requirement_description(req_names[i])
            for i, val in enumerate(requirement_vector) if val == 0
        ]

        if classification == 'LIKELY_QUALIFIED':
            explanation = f"Applicant meets {requirements_met} out of {total_reqs} requirements. "
            explanation += "All key qualifications appear to be satisfied. "
            if met_requirements:
                explanation += f"Met requirements: {', '.join(met_requirements)}."

        elif classification == 'LIKELY_NOT_QUALIFIED':
            explanation = f"Applicant meets only {requirements_met} out of {total_reqs} requirements. "
            explanation += "Key qualifications are not satisfied. "
            if unmet_requirements:
                explanation += f"Missing requirements: {', '.join(unmet_requirements)}."

        else:  # NEEDS_REVIEW
            explanation = f"Applicant meets {requirements_met} out of {total_reqs} requirements. "
            explanation += "This is a borderline case requiring human judgment. "
            if met_requirements:
                explanation += f"Met: {', '.join(met_requirements)}. "
            if unmet_requirements:
                explanation += f"Not met: {', '.join(unmet_requirements)}."

        return explanation
