"""
Inference module for resume screening
"""
import os
import torch
from typing import Dict, List, Optional
from .model import ResumeScreeningModel


class ResumeScreener:
    """
    High-level interface for screening resumes
    """

    def __init__(
        self,
        model_path: str,
        job_code: str,
        model_name: str = 'bert-base-uncased',
        device: str = None
    ):
        """
        Initialize the screener

        Args:
            model_path: Path to the trained model weights
            job_code: Job position code (e.g., 'AGLO', 'OAIV')
            model_name: Name of the BERT model
            device: Device to use ('cuda' or 'cpu')
        """
        self.job_code = job_code
        self.model = ResumeScreeningModel(
            model_name=model_name,
            device=device
        )

        # Load trained weights
        if os.path.exists(model_path):
            self.model.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    def screen_resume(self, resume_text: str) -> Dict:
        """
        Screen a single resume

        Args:
            resume_text: Text extracted from the resume

        Returns:
            Dictionary containing screening results
        """
        results = self.model.predict([resume_text])[0]

        # Generate explanation based on classification
        explanation = self._generate_explanation(results)

        return {
            'classification': results['classification'],
            'confidence': results['confidence'],
            'probabilities': results['probabilities'],
            'explanation': explanation
        }

    def screen_batch(self, resume_texts: List[str]) -> List[Dict]:
        """
        Screen multiple resumes in batch

        Args:
            resume_texts: List of resume texts

        Returns:
            List of screening results
        """
        results = self.model.predict(resume_texts)

        # Add explanations
        for result in results:
            result['explanation'] = self._generate_explanation(result)

        return results

    def _generate_explanation(self, prediction: Dict) -> str:
        """
        Generate human-readable explanation for the classification

        Args:
            prediction: Prediction dictionary

        Returns:
            Explanation string
        """
        classification = prediction['classification']
        confidence = prediction['confidence']
        probs = prediction['probabilities']

        if classification == 'LIKELY_QUALIFIED':
            explanation = (
                f"The applicant appears to meet the qualifications for the {self.job_code} position "
                f"based on the information provided in their application. "
                f"Confidence: {confidence:.1%}. "
            )
        elif classification == 'NEEDS_REVIEW':
            explanation = (
                f"The applicant's qualifications for the {self.job_code} position are unclear "
                f"and require human review. The model was not confident enough to make a definitive classification. "
                f"Confidence: {confidence:.1%}. "
            )
        else:  # LIKELY_NOT_QUALIFIED
            explanation = (
                f"The applicant may not meet the minimum qualifications for the {self.job_code} position "
                f"based on the information provided in their application. "
                f"Confidence: {confidence:.1%}. "
            )

        # Add probability breakdown
        prob_breakdown = ", ".join([
            f"{label}: {prob:.1%}"
            for label, prob in probs.items()
        ])
        explanation += f"Probability breakdown: {prob_breakdown}."

        return explanation

    def set_confidence_threshold(
        self,
        qualified_threshold: float = 0.7,
        not_qualified_threshold: float = 0.7
    ):
        """
        Set confidence thresholds for classification

        If confidence is below threshold, classification changes to NEEDS_REVIEW

        Args:
            qualified_threshold: Minimum confidence for LIKELY_QUALIFIED
            not_qualified_threshold: Minimum confidence for LIKELY_NOT_QUALIFIED
        """
        self.qualified_threshold = qualified_threshold
        self.not_qualified_threshold = not_qualified_threshold

    def screen_with_threshold(self, resume_text: str) -> Dict:
        """
        Screen resume with confidence thresholds

        Args:
            resume_text: Text extracted from the resume

        Returns:
            Dictionary containing screening results
        """
        result = self.screen_resume(resume_text)

        # Apply thresholds
        if result['classification'] == 'LIKELY_QUALIFIED':
            if result['confidence'] < getattr(self, 'qualified_threshold', 0.7):
                result['classification'] = 'NEEDS_REVIEW'
                result['explanation'] = (
                    f"Classification changed to NEEDS_REVIEW due to low confidence "
                    f"({result['confidence']:.1%}). " + result['explanation']
                )
        elif result['classification'] == 'LIKELY_NOT_QUALIFIED':
            if result['confidence'] < getattr(self, 'not_qualified_threshold', 0.7):
                result['classification'] = 'NEEDS_REVIEW'
                result['explanation'] = (
                    f"Classification changed to NEEDS_REVIEW due to low confidence "
                    f"({result['confidence']:.1%}). " + result['explanation']
                )

        return result


class ExplainableScreener(ResumeScreener):
    """
    Extended screener with more detailed explanations
    """

    def __init__(
        self,
        model_path: str,
        job_code: str,
        job_criteria: Dict,
        model_name: str = 'bert-base-uncased',
        device: str = None
    ):
        """
        Initialize the explainable screener

        Args:
            model_path: Path to the trained model weights
            job_code: Job position code
            job_criteria: Dictionary containing job-specific criteria
            model_name: Name of the BERT model
            device: Device to use
        """
        super().__init__(model_path, job_code, model_name, device)
        self.job_criteria = job_criteria

    def _generate_detailed_explanation(
        self,
        prediction: Dict,
        resume_text: str
    ) -> Dict[str, str]:
        """
        Generate detailed explanation with criteria-based analysis

        Args:
            prediction: Prediction dictionary
            resume_text: Original resume text

        Returns:
            Dictionary with detailed explanations
        """
        base_explanation = self._generate_explanation(prediction)

        # Analyze against job criteria
        criteria_analysis = {}

        if 'important_sections' in self.job_criteria:
            for section in self.job_criteria['important_sections']:
                # Simple keyword matching (can be enhanced with NLP)
                section_lower = section.lower()
                if section_lower in resume_text.lower():
                    criteria_analysis[section] = f"✓ Mentioned in application"
                else:
                    criteria_analysis[section] = "✗ Not clearly mentioned"

        return {
            'summary': base_explanation,
            'criteria_analysis': criteria_analysis
        }

    def screen_with_explanation(self, resume_text: str) -> Dict:
        """
        Screen resume with detailed explanation

        Args:
            resume_text: Text extracted from the resume

        Returns:
            Dictionary containing screening results with detailed explanation
        """
        result = self.screen_resume(resume_text)

        # Add detailed explanation
        detailed_explanation = self._generate_detailed_explanation(
            result,
            resume_text
        )

        result['detailed_explanation'] = detailed_explanation

        return result
