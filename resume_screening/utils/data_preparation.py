"""
Data preparation utilities for training the resume screening model
"""
import os
import json
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreparation:
    """Handles data preparation for model training"""

    @staticmethod
    def load_job_criteria(job_info_path: str) -> Dict:
        """
        Load job evaluation criteria from job_info.txt file

        Args:
            job_info_path: Path to the job_info.txt file

        Returns:
            Dictionary containing structured job criteria
        """
        with open(job_info_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the job_info.txt to extract important sections
        criteria = {
            'raw_content': content,
            'important_sections': []
        }

        # Extract section names (these are used for evaluation)
        sections = []
        for line in content.split('\n'):
            line = line.strip()
            if line.endswith(':') and not line.startswith('-'):
                sections.append(line[:-1])

        criteria['important_sections'] = sections

        return criteria

    @staticmethod
    def prepare_training_dataset(
        job_data_dir: str,
        job_code: str,
        labels_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare training dataset from job application files

        Args:
            job_data_dir: Directory containing applicant DOCX files
            job_code: Job position code (e.g., 'AGLO', 'OAIV')
            labels_file: Optional path to a JSON file containing labels for each applicant

        Returns:
            DataFrame with columns: filename, text, label, job_code
        """
        from .document_processor import DocumentProcessor

        data = []
        processor = DocumentProcessor()

        # Load labels if provided
        labels = {}
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                labels = json.load(f)

        # Process all DOCX files in the directory
        for filename in os.listdir(job_data_dir):
            if filename.endswith('_OCR.docx'):
                file_path = os.path.join(job_data_dir, filename)

                try:
                    # Extract text
                    text = processor.extract_text(file_path)
                    text = processor.clean_text(text)

                    # Get label if available
                    label = labels.get(filename, None)

                    data.append({
                        'filename': filename,
                        'text': text,
                        'label': label,
                        'job_code': job_code
                    })
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        return pd.DataFrame(data)

    @staticmethod
    def create_train_val_split(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training and validation sets

        Args:
            df: DataFrame containing the dataset
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df)
        """
        # Filter out unlabeled data
        labeled_df = df[df['label'].notna()].copy()

        if len(labeled_df) == 0:
            raise ValueError("No labeled data found in dataset")

        train_df, val_df = train_test_split(
            labeled_df,
            test_size=test_size,
            random_state=random_state,
            stratify=labeled_df['label'] if len(labeled_df['label'].unique()) > 1 else None
        )

        return train_df, val_df

    @staticmethod
    def save_processed_data(df: pd.DataFrame, output_path: str):
        """
        Save processed data to CSV

        Args:
            df: DataFrame to save
            output_path: Path to save the CSV file
        """
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

    @staticmethod
    def load_processed_data(input_path: str) -> pd.DataFrame:
        """
        Load processed data from CSV

        Args:
            input_path: Path to the CSV file

        Returns:
            DataFrame containing the processed data
        """
        return pd.read_csv(input_path)

    @staticmethod
    def get_label_distribution(df: pd.DataFrame) -> Dict[str, int]:
        """
        Get the distribution of labels in the dataset

        Args:
            df: DataFrame containing the dataset

        Returns:
            Dictionary with label counts
        """
        labeled_df = df[df['label'].notna()]
        return labeled_df['label'].value_counts().to_dict()
