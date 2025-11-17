"""
Management command to populate job positions in the database
"""
from django.core.management.base import BaseCommand
from resume_screening.models import JobPosition


class Command(BaseCommand):
    help = 'Populate the database with initial job positions'

    def handle(self, *args, **options):
        job_positions = [
            {
                'title': 'Agricultural Loan Officer',
                'code': 'AGLO',
                'description': 'Government agricultural loan officer position responsible for processing and managing agricultural loans and financial assistance programs for farmers.',
                'minimum_qualifications': 'Bachelor\'s degree in Agriculture, Finance, or related field, experience in loan processing and agricultural finance.',
            },
            {
                'title': 'Office Assistant',
                'code': 'OFAS',
                'description': 'Office assistant position providing administrative and clerical support for government office operations.',
                'minimum_qualifications': 'High school diploma or equivalent, proficiency in office software, strong organizational and communication skills.',
            },
        ]

        created_count = 0
        updated_count = 0

        for position_data in job_positions:
            position, created = JobPosition.objects.update_or_create(
                code=position_data['code'],
                defaults={
                    'title': position_data['title'],
                    'description': position_data['description'],
                    'minimum_qualifications': position_data['minimum_qualifications'],
                    'is_active': True,
                }
            )

            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'✓ Created: {position.code} - {position.title}')
                )
            else:
                updated_count += 1
                self.stdout.write(
                    self.style.WARNING(f'→ Updated: {position.code} - {position.title}')
                )

        self.stdout.write(
            self.style.SUCCESS(
                f'\nComplete! Created {created_count} new positions, updated {updated_count} existing positions.'
            )
        )
