from django.contrib import admin
from .models import JobPosition, Applicant, ScreeningResult, TrainingData, ModelVersion


@admin.register(JobPosition)
class JobPositionAdmin(admin.ModelAdmin):
    list_display = ('code', 'title', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('code', 'title')


@admin.register(Applicant)
class ApplicantAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'last_name', 'email', 'job_position', 'applied_at')
    list_filter = ('job_position', 'applied_at')
    search_fields = ('first_name', 'last_name', 'email')


@admin.register(ScreeningResult)
class ScreeningResultAdmin(admin.ModelAdmin):
    list_display = ('applicant', 'classification', 'confidence_score', 'human_reviewed', 'screened_at')
    list_filter = ('classification', 'human_reviewed', 'screened_at')
    search_fields = ('applicant__first_name', 'applicant__last_name', 'applicant__email')
    readonly_fields = ('screened_at',)


@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    list_display = ('job_position', 'label', 'is_active', 'created_at')
    list_filter = ('job_position', 'label', 'is_active', 'created_at')
    search_fields = ('notes',)


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('job_position', 'version', 'validation_accuracy', 'is_active', 'training_date')
    list_filter = ('job_position', 'is_active', 'training_date')
    search_fields = ('version',)
    readonly_fields = ('training_date',)
