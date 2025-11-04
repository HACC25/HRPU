"""
Views for the resume screening application
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings
import os

from .models import JobPosition, Applicant, ScreeningResult
from .utils.document_processor import DocumentProcessor
from .ml.inference import ResumeScreener


def dashboard(request):
    """Main dashboard view"""
    job_positions = JobPosition.objects.filter(is_active=True)
    recent_screenings = ScreeningResult.objects.select_related('applicant')[:10]

    context = {
        'job_positions': job_positions,
        'recent_screenings': recent_screenings,
    }
    return render(request, 'resume_screening/dashboard.html', context)


def job_position_detail(request, job_id):
    """View details for a specific job position"""
    job = get_object_or_404(JobPosition, id=job_id)
    applicants = job.applicants.all()

    # Get screening statistics
    screened_applicants = applicants.filter(screening_result__isnull=False)
    qualified_count = screened_applicants.filter(
        screening_result__classification='LIKELY_QUALIFIED'
    ).count()
    needs_review_count = screened_applicants.filter(
        screening_result__classification='NEEDS_REVIEW'
    ).count()
    not_qualified_count = screened_applicants.filter(
        screening_result__classification='LIKELY_NOT_QUALIFIED'
    ).count()

    context = {
        'job': job,
        'applicants': applicants,
        'total_applicants': applicants.count(),
        'screened_count': screened_applicants.count(),
        'qualified_count': qualified_count,
        'needs_review_count': needs_review_count,
        'not_qualified_count': not_qualified_count,
    }
    return render(request, 'resume_screening/job_detail.html', context)


def applicant_detail(request, applicant_id):
    """View details for a specific applicant"""
    applicant = get_object_or_404(Applicant, id=applicant_id)

    try:
        screening_result = applicant.screening_result
    except ScreeningResult.DoesNotExist:
        screening_result = None

    context = {
        'applicant': applicant,
        'screening_result': screening_result,
    }
    return render(request, 'resume_screening/applicant_detail.html', context)


def screen_applicant(request, applicant_id):
    """Screen a specific applicant using the trained model"""
    applicant = get_object_or_404(Applicant, id=applicant_id)

    # Check if already screened
    if hasattr(applicant, 'screening_result'):
        messages.warning(request, 'This applicant has already been screened.')
        return redirect('applicant_detail', applicant_id=applicant_id)

    # Get the model path for this job
    job_code = applicant.job_position.code
    model_path = os.path.join(
        settings.ML_MODELS_DIR,
        job_code,
        'best_model.pt'
    )

    if not os.path.exists(model_path):
        messages.error(
            request,
            f'No trained model found for job {job_code}. Please train a model first.'
        )
        return redirect('applicant_detail', applicant_id=applicant_id)

    try:
        # Extract text if not already done
        if not applicant.extracted_text:
            processor = DocumentProcessor()
            text = processor.extract_text(applicant.application_file.path)
            applicant.extracted_text = processor.clean_text(text)
            applicant.save()

        # Initialize screener
        screener = ResumeScreener(
            model_path=model_path,
            job_code=job_code
        )

        # Screen the applicant
        result = screener.screen_resume(applicant.extracted_text)

        # Save screening result
        screening_result = ScreeningResult.objects.create(
            applicant=applicant,
            classification=result['classification'],
            confidence_score=result['confidence'],
            explanation=result['explanation'],
            criteria_scores=result.get('probabilities', {}),
            model_version=f"{job_code}_v1"
        )

        messages.success(
            request,
            f'Screening complete: {result["classification"]} (Confidence: {result["confidence"]:.1%})'
        )

    except Exception as e:
        messages.error(request, f'Error during screening: {str(e)}')

    return redirect('applicant_detail', applicant_id=applicant_id)


@require_http_methods(["POST"])
def update_screening_result(request, result_id):
    """Update screening result with human review"""
    result = get_object_or_404(ScreeningResult, id=result_id)

    human_decision = request.POST.get('human_decision')
    human_notes = request.POST.get('human_notes', '')

    if human_decision in dict(ScreeningResult.CLASSIFICATION_CHOICES):
        result.human_reviewed = True
        result.human_decision = human_decision
        result.human_notes = human_notes
        result.reviewed_by = request.user.username if request.user.is_authenticated else 'anonymous'
        result.save()

        messages.success(request, 'Review saved successfully.')
    else:
        messages.error(request, 'Invalid decision value.')

    return redirect('applicant_detail', applicant_id=result.applicant.id)


# API Views (for AJAX requests)

@require_http_methods(["POST"])
def api_screen_applicant(request, applicant_id):
    """API endpoint to screen an applicant"""
    applicant = get_object_or_404(Applicant, id=applicant_id)

    # Check if already screened
    if hasattr(applicant, 'screening_result'):
        return JsonResponse({
            'error': 'Already screened',
            'result': {
                'classification': applicant.screening_result.classification,
                'confidence': applicant.screening_result.confidence_score
            }
        }, status=400)

    job_code = applicant.job_position.code
    model_path = os.path.join(settings.ML_MODELS_DIR, job_code, 'best_model.pt')

    if not os.path.exists(model_path):
        return JsonResponse({
            'error': f'No trained model found for job {job_code}'
        }, status=404)

    try:
        # Extract text if needed
        if not applicant.extracted_text:
            processor = DocumentProcessor()
            text = processor.extract_text(applicant.application_file.path)
            applicant.extracted_text = processor.clean_text(text)
            applicant.save()

        # Screen
        screener = ResumeScreener(model_path=model_path, job_code=job_code)
        result = screener.screen_resume(applicant.extracted_text)

        # Save result
        screening_result = ScreeningResult.objects.create(
            applicant=applicant,
            classification=result['classification'],
            confidence_score=result['confidence'],
            explanation=result['explanation'],
            criteria_scores=result.get('probabilities', {}),
            model_version=f"{job_code}_v1"
        )

        return JsonResponse({
            'success': True,
            'result': {
                'id': screening_result.id,
                'classification': result['classification'],
                'confidence': result['confidence'],
                'explanation': result['explanation']
            }
        })

    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)


def screening_statistics(request):
    """View showing overall screening statistics"""
    jobs = JobPosition.objects.filter(is_active=True)

    stats = []
    for job in jobs:
        applicants = job.applicants.all()
        screened = applicants.filter(screening_result__isnull=False)

        stats.append({
            'job': job,
            'total': applicants.count(),
            'screened': screened.count(),
            'qualified': screened.filter(screening_result__classification='LIKELY_QUALIFIED').count(),
            'needs_review': screened.filter(screening_result__classification='NEEDS_REVIEW').count(),
            'not_qualified': screened.filter(screening_result__classification='LIKELY_NOT_QUALIFIED').count(),
        })

    context = {
        'statistics': stats
    }
    return render(request, 'resume_screening/statistics.html', context)
