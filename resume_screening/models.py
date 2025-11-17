from django.db import models
from django.core.validators import FileExtensionValidator


class JobPosition(models.Model):
    """Stores information about job positions"""
    title = models.CharField(max_length=200)
    code = models.CharField(max_length=50, unique=True)
    description = models.TextField()
    minimum_qualifications = models.TextField()

    # Job specification files
    class_specifications_file = models.FileField(
        upload_to='job_specs/',
        validators=[FileExtensionValidator(['pdf'])],
        null=True,
        blank=True
    )

    # Important criteria extracted from job_info.txt
    evaluation_criteria = models.JSONField(default=dict, help_text="Structured criteria for evaluating applicants")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.code} - {self.title}"


class Applicant(models.Model):
    """Stores applicant information"""
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)

    job_position = models.ForeignKey(JobPosition, on_delete=models.CASCADE, related_name='applicants')

    # Resume/Application document
    application_file = models.FileField(
        upload_to='applications/',
        validators=[FileExtensionValidator(['docx', 'pdf'])],
    )

    # Extracted text from application
    extracted_text = models.TextField(blank=True, help_text="Raw text extracted from application document")

    applied_at = models.DateTimeField(auto_now_add=True)
    archived = models.BooleanField(default=False, help_text="Whether this applicant has been archived")

    class Meta:
        ordering = ['-applied_at']
        unique_together = ['email', 'job_position']

    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.job_position.code}"

    @property
    def full_name(self):
        """Return full name of applicant"""
        return f"{self.first_name} {self.last_name}"

    @property
    def display_id(self):
        """Return display ID for applicant (using primary key)"""
        return f"{self.pk:05d}" if self.pk else "N/A"


class ScreeningResult(models.Model):
    """Stores AI screening results for applicants"""

    CLASSIFICATION_CHOICES = [
        ('LIKELY_QUALIFIED', 'Likely Qualified'),
        ('NEEDS_REVIEW', 'Needs Human Review'),
        ('LIKELY_NOT_QUALIFIED', 'Likely Not Qualified'),
    ]

    applicant = models.OneToOneField(Applicant, on_delete=models.CASCADE, related_name='screening_result')

    classification = models.CharField(max_length=30, choices=CLASSIFICATION_CHOICES)
    confidence_score = models.FloatField(help_text="Model confidence (0-1)")

    # Detailed explanation
    explanation = models.TextField(help_text="AI-generated explanation for the classification")

    # Detailed scores for each criterion
    criteria_scores = models.JSONField(
        default=dict,
        help_text="Breakdown of scores for specific job criteria"
    )

    # Model information
    model_version = models.CharField(max_length=100, help_text="Version of the model used")

    screened_at = models.DateTimeField(auto_now_add=True)

    # Human review
    human_reviewed = models.BooleanField(default=False)
    human_decision = models.CharField(max_length=30, blank=True, null=True)
    human_notes = models.TextField(blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewed_by = models.CharField(max_length=100, blank=True)

    class Meta:
        ordering = ['-screened_at']

    def __str__(self):
        return f"{self.applicant} - {self.classification}"


class TrainingData(models.Model):
    """Stores labeled training data for model training"""

    job_position = models.ForeignKey(JobPosition, on_delete=models.CASCADE, related_name='training_data')

    # Training document
    document_file = models.FileField(
        upload_to='training_data/',
        validators=[FileExtensionValidator(['docx', 'pdf'])],
    )

    extracted_text = models.TextField(help_text="Extracted text from training document")

    # Label
    label = models.CharField(
        max_length=30,
        choices=ScreeningResult.CLASSIFICATION_CHOICES,
        help_text="Ground truth label for training"
    )

    # Metadata
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True, help_text="Include in training set")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Training: {self.job_position.code} - {self.label}"


class ModelVersion(models.Model):
    """Tracks different versions of trained models"""

    job_position = models.ForeignKey(JobPosition, on_delete=models.CASCADE, related_name='model_versions')

    version = models.CharField(max_length=50)
    model_path = models.CharField(max_length=500, help_text="Path to saved model")

    # Training metadata
    training_samples = models.IntegerField()
    validation_accuracy = models.FloatField(null=True, blank=True)
    training_date = models.DateTimeField(auto_now_add=True)

    # Training configuration
    config = models.JSONField(default=dict, help_text="Model hyperparameters and training config")

    # Performance metrics
    metrics = models.JSONField(default=dict, help_text="Detailed performance metrics")

    is_active = models.BooleanField(default=False, help_text="Currently deployed model")

    class Meta:
        ordering = ['-training_date']
        unique_together = ['job_position', 'version']

    def __str__(self):
        return f"{self.job_position.code} - v{self.version}"
