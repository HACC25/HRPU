"""
URL configuration for resume_screening app
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_view, name='upload'),
    path('review/<int:applicant_id>/', views.applicant_review_view, name='applicant_review'),
    path('api/archive/<int:applicant_id>/', views.archive_applicant, name='archive_applicant'),
    path('api/update-decision/<int:applicant_id>/', views.update_decision, name='update_decision'),
    path('api/file/<int:applicant_id>/', views.serve_application_file, name='serve_application_file'),
]
