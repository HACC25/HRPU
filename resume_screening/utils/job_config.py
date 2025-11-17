"""
Job configuration management for resume screening system

Loads job-specific parameters from job_configs.json to make the system
job-agnostic and easily extensible to new positions.
"""
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class JobConfig:
    """Configuration for a specific job position"""
    job_code: str
    job_name: str
    num_requirements: int
    num_questions: int
    requirements: List[str]
    hybrid_thresholds: Dict[str, int]
    file_prefix: str

    def get_qualified_threshold(self) -> int:
        """Get the minimum requirements needed for LIKELY_QUALIFIED"""
        return self.hybrid_thresholds.get('qualified', self.num_requirements)

    def get_not_qualified_threshold(self) -> int:
        """Get the maximum requirements for LIKELY_NOT_QUALIFIED"""
        return self.hybrid_thresholds.get('not_qualified', 0)

    def validate(self) -> bool:
        """Validate configuration consistency"""
        if self.num_requirements != len(self.requirements):
            raise ValueError(
                f"num_requirements ({self.num_requirements}) doesn't match "
                f"requirements list length ({len(self.requirements)})"
            )

        qualified_threshold = self.get_qualified_threshold()
        not_qualified_threshold = self.get_not_qualified_threshold()

        if qualified_threshold > self.num_requirements:
            raise ValueError(
                f"qualified threshold ({qualified_threshold}) cannot exceed "
                f"num_requirements ({self.num_requirements})"
            )

        if not_qualified_threshold >= qualified_threshold:
            raise ValueError(
                f"not_qualified threshold ({not_qualified_threshold}) must be less than "
                f"qualified threshold ({qualified_threshold})"
            )

        return True


class JobConfigManager:
    """Manages loading and accessing job configurations"""

    _instance = None
    _configs: Dict[str, JobConfig] = {}
    _config_file_path: Optional[str] = None

    def __new__(cls):
        """Singleton pattern to ensure one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_configs(self, config_file_path: str = None):
        """
        Load job configurations from JSON file

        Args:
            config_file_path: Path to job_configs.json. If None, looks in project root.
        """
        if config_file_path is None:
            # Look for job_configs.json in project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            config_file_path = os.path.join(project_root, 'job_configs.json')

        if not os.path.exists(config_file_path):
            raise FileNotFoundError(
                f"Job configuration file not found: {config_file_path}\n"
                f"Please create job_configs.json in the project root."
            )

        self._config_file_path = config_file_path

        with open(config_file_path, 'r', encoding='utf-8') as f:
            configs_dict = json.load(f)

        # Convert to JobConfig objects
        self._configs = {}
        for job_code, config_data in configs_dict.items():
            config = JobConfig(
                job_code=job_code,
                job_name=config_data['job_name'],
                num_requirements=config_data['num_requirements'],
                num_questions=config_data['num_questions'],
                requirements=config_data['requirements'],
                hybrid_thresholds=config_data['hybrid_thresholds'],
                file_prefix=config_data['file_prefix']
            )
            config.validate()
            self._configs[job_code] = config

    def get_config(self, job_code: str) -> JobConfig:
        """
        Get configuration for a specific job

        Args:
            job_code: Job code (e.g., 'AGLO', 'OAIV')

        Returns:
            JobConfig object

        Raises:
            ValueError: If job_code is not found
        """
        if not self._configs:
            self.load_configs()

        if job_code not in self._configs:
            available = ', '.join(self._configs.keys())
            raise ValueError(
                f"Unknown job code: {job_code}\n"
                f"Available job codes: {available}"
            )

        return self._configs[job_code]

    def list_available_jobs(self) -> List[str]:
        """Get list of available job codes"""
        if not self._configs:
            self.load_configs()
        return list(self._configs.keys())

    def get_all_configs(self) -> Dict[str, JobConfig]:
        """Get all loaded configurations"""
        if not self._configs:
            self.load_configs()
        return self._configs.copy()


# Global instance
_manager = JobConfigManager()


def get_job_config(job_code: str) -> JobConfig:
    """
    Convenience function to get job configuration

    Args:
        job_code: Job code (e.g., 'AGLO', 'OAIV')

    Returns:
        JobConfig object

    Example:
        >>> config = get_job_config('AGLO')
        >>> print(config.num_requirements)  # 5
        >>> print(config.requirements)  # ['basic', 'credit_analysis_1y', ...]
    """
    return _manager.get_config(job_code)


def list_jobs() -> List[str]:
    """
    Get list of all available job codes

    Returns:
        List of job code strings

    Example:
        >>> jobs = list_jobs()
        >>> print(jobs)  # ['AGLO', 'OAIV']
    """
    return _manager.list_available_jobs()


def reload_configs(config_file_path: str = None):
    """
    Reload configurations from file (useful after editing job_configs.json)

    Args:
        config_file_path: Optional path to config file
    """
    _manager.load_configs(config_file_path)
