"""
Views for resume screening web application
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.http import require_http_methods, require_POST
from django.contrib import messages
from django.db.models import Q
from django.utils import timezone

from .models import Applicant, JobPosition, ScreeningResult
from .services import ResumeScreeningService


def dashboard_view(request):
    """
    Main dashboard showing applicants grouped by position
    """
    # Get filter parameters
    position_code = request.GET.get('position', 'AGLO')  # Default to AGLO
    show_archived = request.GET.get('archived', 'false') == 'true'
    filter_id = request.GET.get('filter_id', '').strip()
    sort_by = request.GET.get('sort', 'newest')

    # Get all active positions
    positions = JobPosition.objects.filter(is_active=True)

    # Build query
    query = Q(job_position__code=position_code)

    if show_archived:
        query &= Q(archived=True)
    else:
        query &= Q(archived=False)

    if filter_id:
        query &= Q(pk=int(filter_id)) if filter_id.isdigit() else Q(first_name__icontains=filter_id) | Q(last_name__icontains=filter_id)

    # Get applicants
    applicants = Applicant.objects.filter(query).select_related(
        'job_position', 'screening_result'
    )

    # Apply sorting
    if sort_by == 'newest':
        applicants = applicants.order_by('-applied_at')
    elif sort_by == 'oldest':
        applicants = applicants.order_by('applied_at')
    elif sort_by == 'id_asc':
        applicants = applicants.order_by('pk')
    elif sort_by == 'id_desc':
        applicants = applicants.order_by('-pk')

    context = {
        'applicants': applicants,
        'positions': positions,
        'current_position': position_code,
        'show_archived': show_archived,
        'filter_id': filter_id,
        'sort_by': sort_by,
    }

    return render(request, 'resume_screening/dashboard.html', context)


def upload_view(request):
    """
    Resume upload page with automatic screening
    """
    if request.method == 'POST':
        try:
            # Get form data
            first_name = request.POST.get('first_name', '').strip()
            last_name = request.POST.get('last_name', '').strip()
            email = request.POST.get('email', '').strip()
            job_position_id = request.POST.get('job_position')
            application_file = request.FILES.get('application_file')

            # Validate
            if not all([first_name, last_name, email, job_position_id, application_file]):
                messages.error(request, 'All fields are required.')
                return redirect('upload')

            # Get job position
            job_position = get_object_or_404(JobPosition, pk=job_position_id)

            # Check if applicant already exists
            if Applicant.objects.filter(email=email, job_position=job_position).exists():
                messages.error(request, 'An application with this email already exists for this position.')
                return redirect('upload')

            # Create applicant
            applicant = Applicant.objects.create(
                first_name=first_name,
                last_name=last_name,
                email=email,
                job_position=job_position,
                application_file=application_file
            )

            # Run automatic screening
            try:
                screening_result = ResumeScreeningService.screen_uploaded_resume(applicant)
                messages.success(
                    request,
                    f'Application uploaded and screened successfully! '
                    f'Classification: {screening_result.get_classification_display()}'
                )
                return redirect('applicant_review', applicant_id=applicant.pk)
            except Exception as e:
                messages.warning(
                    request,
                    f'Application uploaded but screening failed: {str(e)}. '
                    f'You can review it manually.'
                )
                return redirect('applicant_review', applicant_id=applicant.pk)

        except Exception as e:
            messages.error(request, f'Error uploading application: {str(e)}')
            return redirect('upload')

    # GET request - show form
    positions = JobPosition.objects.filter(is_active=True)
    context = {
        'positions': positions,
    }
    return render(request, 'resume_screening/upload.html', context)


def applicant_review_view(request, applicant_id):
    """
    Individual applicant review page with resume and screening results
    """
    applicant = get_object_or_404(
        Applicant.objects.select_related('job_position', 'screening_result'),
        pk=applicant_id
    )

    # Get screening result if exists
    try:
        screening_result = applicant.screening_result
    except ScreeningResult.DoesNotExist:
        screening_result = None

    # Parse criteria scores for display
    requirement_bullets = []
    if screening_result and screening_result.criteria_scores:
        req_vector = screening_result.criteria_scores.get('requirements_vector', [])
        job_config = None

        try:
            from .utils.job_config import get_job_config
            job_config = get_job_config(applicant.job_position.code)
        except:
            pass

        if job_config and req_vector:
            for i, (req_name, met) in enumerate(zip(job_config.requirements, req_vector)):
                requirement_bullets.append({
                    'name': req_name.replace('_', ' ').title(),
                    'met': bool(met),
                    'index': i
                })

    # Calculate confidence percentage
    confidence_percentage = None
    if screening_result:
        confidence_percentage = screening_result.confidence_score * 100

    context = {
        'applicant': applicant,
        'screening_result': screening_result,
        'requirement_bullets': requirement_bullets,
        'confidence_percentage': confidence_percentage,
    }

    return render(request, 'resume_screening/applicant_review.html', context)


@require_POST
def archive_applicant(request, applicant_id):
    """
    Archive an applicant (AJAX endpoint)
    """
    try:
        applicant = get_object_or_404(Applicant, pk=applicant_id)
        applicant.archived = True
        applicant.save()

        return JsonResponse({
            'success': True,
            'message': f'Applicant {applicant.full_name} archived successfully.'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=400)


@require_POST
def update_decision(request, applicant_id):
    """
    Update human review decision (AJAX endpoint)
    """
    try:
        applicant = get_object_or_404(Applicant, pk=applicant_id)

        try:
            screening_result = applicant.screening_result
        except ScreeningResult.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'No screening result found for this applicant.'
            }, status=404)

        # Get decision from POST data
        decision = request.POST.get('decision')
        notes = request.POST.get('notes', '').strip()

        if decision not in ['LIKELY_QUALIFIED', 'LIKELY_NOT_QUALIFIED', 'NEEDS_REVIEW']:
            return JsonResponse({
                'success': False,
                'message': 'Invalid decision value.'
            }, status=400)

        # Update screening result
        screening_result.human_reviewed = True
        screening_result.human_decision = decision
        screening_result.human_notes = notes
        screening_result.reviewed_at = timezone.now()
        screening_result.reviewed_by = request.user.username if request.user.is_authenticated else 'Anonymous'
        screening_result.save()

        return JsonResponse({
            'success': True,
            'message': 'Decision updated successfully.'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': str(e)
        }, status=400)


def serve_application_file(request, applicant_id):
    """
    Serve the application file for viewing
    """
    applicant = get_object_or_404(Applicant, pk=applicant_id)

    try:
        file_path = applicant.application_file.path
        return FileResponse(
            open(file_path, 'rb'),
            content_type='application/octet-stream'
        )
    except Exception as e:
        raise Http404("Application file not found.")
