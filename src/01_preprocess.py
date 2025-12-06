"""
Data Exploration and Cleaning Script

This script processes the aggregated labeled data from 00_aggregate_jsons.py,
performs data exploration, quality filtering, and creates clean train/test splits.

Input: _data/aggregated/labeled_data.csv
Output:
    - _data/final/train.csv (cleaned training data)
    - _data/final/test.csv (held-out test data)

Processing Pipeline:
    1. Load aggregated data
    2. Separate test holdout (K3I7DL + BCLHKC otp records)
    3. Remove duplicate texts (smart merge based on lead_time)
    4. Filter short texts (< 40 characters)
    5. Remove low-quality annotators (mean lead_time < 10s)
    6. Save final datasets
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class DatasetStats:
    """Container for dataset statistics."""

    total_records: int = 0
    unique_texts: int = 0
    unique_students: int = 0
    label_distribution: Dict[int, int] = field(default_factory=dict)
    mean_text_length: float = 0.0
    median_text_length: float = 0.0
    mean_lead_time: float = 0.0
    median_lead_time: float = 0.0

    def print_summary(self, title: str = "Dataset Statistics") -> None:
        """Print a formatted summary of statistics."""
        print(f"\n{title}:")
        print(f"  Total records: {self.total_records}")
        print(f"  Unique texts: {self.unique_texts}")
        print(f"  Unique students: {self.unique_students}")
        print(f"  Mean text length: {self.mean_text_length:.0f} chars")
        print(f"  Median text length: {self.median_text_length:.0f} chars")
        print(f"  Mean lead time: {self.mean_lead_time:.2f}s")
        print(f"  Median lead time: {self.median_lead_time:.2f}s")

        if self.label_distribution:
            print("\n  Label distribution:")
            total: int = sum(self.label_distribution.values())
            for label in sorted(self.label_distribution.keys()):
                count: int = self.label_distribution[label]
                pct: float = (count / total) * 100
                print(f"    Label {label}: {count:4d} ({pct:5.2f}%)")


@dataclass
class CleaningReport:
    """Container for tracking data cleaning operations."""

    initial_records: int = 0
    test_holdout_records: int = 0
    duplicates_removed: int = 0
    short_texts_removed: int = 0
    low_quality_students_removed: int = 0
    students_removed: List[str] = field(default_factory=list)
    final_train_records: int = 0
    final_test_records: int = 0

    def print_summary(self) -> None:
        """Print a formatted summary of cleaning operations."""
        print("\n" + "=" * 80)
        print("DATA CLEANING SUMMARY")
        print("=" * 80)

        print(f"\nInitial records: {self.initial_records}")
        print(f"Test holdout separated: {self.test_holdout_records}")
        print(f"Duplicates removed: {self.duplicates_removed}")
        print(f"Short texts removed: {self.short_texts_removed}")
        print(f"Low-quality student records removed: {self.low_quality_students_removed}")

        if self.students_removed:
            print(f"Students removed: {', '.join(self.students_removed)}")

        print(f"\nFinal training records: {self.final_train_records}")
        print(f"Final test records: {self.final_test_records}")
        print(f"Total final records: {self.final_train_records + self.final_test_records}")


class DataExplorer:
    """
    Handles data exploration and cleaning for readability label data.

    This class processes labeled ÁSZF (Terms and Conditions) data,
    performing quality filtering and creating train/test splits.
    """

    # Configuration constants
    COLUMNS_TO_KEEP: List[str] = [
        'student_code',
        'json_filename',
        'text',
        'label_text',
        'label_numeric',
        'annotation_created_at',
        'lead_time_seconds',
    ]

    MIN_TEXT_LENGTH: int = 50
    MIN_MEAN_LEAD_TIME: float = 10.0

    # Test set configuration
    TEST_STUDENT_CODE: str = 'K3I7DL'
    TEST_SOURCE_PATTERN: str = 'otp'
    TEST_SOURCE_STUDENT: str = 'BCLHKC'

    def __init__(self, input_path: Path, output_dir: Path):
        """
        Initialize the DataExplorer.

        Args:
            input_path: Path to the aggregated CSV file
            output_dir: Directory for output files
        """
        self.input_path: Path = input_path
        self.output_dir: Path = output_dir
        self.df: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.report: CleaningReport = CleaningReport()

    def load_data(self) -> pd.DataFrame:
        """
        Load the aggregated data from CSV.

        Returns:
            DataFrame with loaded data
        """
        print("Loading data...")
        print(f"  Input file: {self.input_path}")

        self.df = pd.read_csv(self.input_path)
        self.report.initial_records = len(self.df)

        print(f"  Loaded {len(self.df)} records")
        print(f"  Columns: {list(self.df.columns)}")

        return self.df

    def compute_stats(self, df: pd.DataFrame) -> DatasetStats:
        """
        Compute statistics for a DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            DatasetStats object with computed statistics
        """
        stats: DatasetStats = DatasetStats()
        stats.total_records = len(df)
        stats.unique_texts = df['text'].nunique()
        stats.unique_students = df['student_code'].nunique()

        # Label distribution
        for label in df['label_numeric'].unique():
            count: int = int((df['label_numeric'] == label).sum())
            stats.label_distribution[label] = count

        # Text length stats
        if 'text_length' not in df.columns:
            text_lengths: pd.Series = df['text'].str.len()
        else:
            text_lengths = df['text_length']

        stats.mean_text_length = float(text_lengths.mean())
        stats.median_text_length = float(text_lengths.median())

        # Lead time stats
        stats.mean_lead_time = float(df['lead_time_seconds'].mean())
        stats.median_lead_time = float(df['lead_time_seconds'].median())

        return stats

    def separate_test_holdout(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate test holdout data from the main dataset.

        Test set includes:
            - All records from student K3I7DL
            - BCLHKC records with 'otp' in source_file

        Returns:
            Tuple of (working_df, test_df)
        """
        print("\n" + "=" * 80)
        print("SEPARATING TEST HOLDOUT")
        print("=" * 80)

        test_condition: pd.Series = (
            (self.df['student_code'] == self.TEST_STUDENT_CODE) |
            (
                (self.df['student_code'] == self.TEST_SOURCE_STUDENT) &
                (self.df['source_file'].str.contains(self.TEST_SOURCE_PATTERN, case=False))
            )
        )

        self.df_test = self.df[test_condition].copy()
        self.df = self.df[~test_condition].copy()

        self.report.test_holdout_records = len(self.df_test)
        self.report.final_test_records = len(self.df_test)

        k3i7dl_count: int = len(self.df_test[self.df_test['student_code'] == self.TEST_STUDENT_CODE])
        otp_count: int = len(self.df_test[self.df_test['student_code'] == self.TEST_SOURCE_STUDENT])

        print(f"\nTest holdout: {len(self.df_test)} records")
        print(f"  {self.TEST_STUDENT_CODE}: {k3i7dl_count} records")
        print(f"  {self.TEST_SOURCE_STUDENT} (otp): {otp_count} records")
        print(f"\nWorking dataset: {len(self.df)} records")

        return self.df, self.df_test

    def select_columns(self) -> pd.DataFrame:
        """
        Select and rename relevant columns.

        Returns:
            DataFrame with selected columns
        """
        print("\nSelecting relevant columns...")

        self.df = self.df[self.COLUMNS_TO_KEEP].copy()
        self.df = self.df.rename(columns={'annotation_created_at': 'labeled_at'})

        print(f"  Selected {len(self.COLUMNS_TO_KEEP)} columns")

        return self.df

    def handle_duplicates(self) -> pd.DataFrame:
        """
        Handle duplicate texts by smart merging.

        Strategy:
            - If labels agree: keep first occurrence
            - If labels disagree: keep the one with higher lead_time

        Returns:
            DataFrame with duplicates resolved
        """
        print("\n" + "=" * 80)
        print("HANDLING DUPLICATE TEXTS")
        print("=" * 80)

        initial_count: int = len(self.df)
        unique_texts: int = self.df['text'].nunique()
        duplicate_count: int = initial_count - unique_texts

        print(f"\nTotal records: {initial_count}")
        print(f"Unique texts: {unique_texts}")
        print(f"Duplicate texts: {duplicate_count}")

        if duplicate_count == 0:
            print("No duplicates to process.")
            return self.df

        # Split into duplicates and non-duplicates
        duplicate_mask: pd.Series = self.df.duplicated(subset=['text'], keep=False)
        duplicates: pd.DataFrame = self.df[duplicate_mask].copy()
        non_duplicates: pd.DataFrame = self.df[~duplicate_mask].copy()

        print(f"\nProcessing {len(duplicates)} duplicate records...")

        # Process duplicates
        kept_records: List[pd.Series] = []

        for text, group in duplicates.groupby('text'):
            if len(group) == 2:
                labels: np.ndarray = group['label_numeric'].values

                if labels[0] == labels[1]:
                    # Labels agree - keep first
                    kept_records.append(group.iloc[0])
                else:
                    # Labels disagree - keep higher lead_time
                    max_idx: int = group['lead_time_seconds'].idxmax()
                    kept_records.append(group.loc[max_idx])
            else:
                # More than 2 duplicates - keep highest lead_time
                max_idx = group['lead_time_seconds'].idxmax()
                kept_records.append(group.loc[max_idx])

        # Combine non-duplicates with resolved duplicates
        kept_duplicates_df: pd.DataFrame = pd.DataFrame(kept_records)
        self.df = pd.concat([non_duplicates, kept_duplicates_df], ignore_index=True)

        self.report.duplicates_removed = initial_count - len(self.df)

        print(f"Duplicates resolved: {self.report.duplicates_removed} records removed")
        print(f"Final count: {len(self.df)} records")

        return self.df

    def add_text_length(self) -> pd.DataFrame:
        """
        Add text_length column to the DataFrame.

        Returns:
            DataFrame with text_length column added
        """
        self.df['text_length'] = self.df['text'].str.len()
        return self.df

    def filter_short_texts(self) -> pd.DataFrame:
        """
        Remove texts shorter than MIN_TEXT_LENGTH characters.

        Returns:
            Filtered DataFrame
        """
        print("\n" + "=" * 80)
        print(f"FILTERING SHORT TEXTS (< {self.MIN_TEXT_LENGTH} chars)")
        print("=" * 80)

        initial_count: int = len(self.df)

        if 'text_length' not in self.df.columns:
            self.add_text_length()

        self.df = self.df[self.df['text_length'] >= self.MIN_TEXT_LENGTH].copy()

        removed_count: int = initial_count - len(self.df)
        self.report.short_texts_removed = removed_count

        print(f"\nBefore: {initial_count} records")
        print(f"After: {len(self.df)} records")
        print(f"Removed: {removed_count} records ({removed_count / initial_count * 100:.2f}%)")

        return self.df

    def analyze_student_lead_times(self) -> pd.DataFrame:
        """
        Analyze lead times per student.

        Returns:
            DataFrame with student-level lead time statistics
        """
        student_stats: pd.DataFrame = self.df.groupby('student_code').agg({
            'lead_time_seconds': ['mean', 'median', 'count']
        }).round(2)

        student_stats.columns = ['mean_lead_time', 'median_lead_time', 'count']
        student_stats = student_stats.sort_values('mean_lead_time')

        return student_stats

    def remove_low_quality_students(self) -> pd.DataFrame:
        """
        Remove students with mean lead_time below threshold.

        Returns:
            Filtered DataFrame
        """
        print("\n" + "=" * 80)
        print(f"REMOVING LOW-QUALITY STUDENTS (mean lead_time < {self.MIN_MEAN_LEAD_TIME}s)")
        print("=" * 80)

        initial_count: int = len(self.df)

        # Calculate mean lead time per student
        student_stats: pd.DataFrame = self.df.groupby('student_code').agg({
            'lead_time_seconds': 'mean',
            'student_code': 'count'
        }).rename(columns={
            'student_code': 'count',
            'lead_time_seconds': 'mean_lead_time'
        })

        # Find students to remove
        students_to_remove: List[str] = student_stats[
            student_stats['mean_lead_time'] < self.MIN_MEAN_LEAD_TIME
        ].index.tolist()

        if students_to_remove:
            print("\nStudents to remove:")
            for student in students_to_remove:
                mean_lt: float = student_stats.loc[student, 'mean_lead_time']
                count: int = int(student_stats.loc[student, 'count'])
                print(f"  {student}: mean={mean_lt:.2f}s, records={count}")

            self.df = self.df[~self.df['student_code'].isin(students_to_remove)].copy()
            self.report.students_removed = students_to_remove
        else:
            print("\nNo students to remove.")

        removed_count: int = initial_count - len(self.df)
        self.report.low_quality_students_removed = removed_count

        print(f"\nBefore: {initial_count} records")
        print(f"After: {len(self.df)} records")
        print(f"Removed: {removed_count} records ({removed_count / initial_count * 100:.2f}%)")

        return self.df

    def save_datasets(self) -> Tuple[Path, Path]:
        """
        Save the final train and test datasets.

        Returns:
            Tuple of (train_path, test_path)
        """
        print("\n" + "=" * 80)
        print("SAVING FINAL DATASETS")
        print("=" * 80)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save training data
        train_path: Path = self.output_dir / 'train.csv'
        self.df.to_csv(train_path, index=False, encoding='utf-8')
        self.report.final_train_records = len(self.df)

        print(f"\nTraining dataset: {train_path}")
        print(f"  Records: {len(self.df)}")
        print(f"  Unique students: {self.df['student_code'].nunique()}")
        print(f"  Unique texts: {self.df['text'].nunique()}")

        # Save test data
        test_path: Path = self.output_dir / 'test.csv'
        self.df_test.to_csv(test_path, index=False, encoding='utf-8')

        print(f"\nTest dataset: {test_path}")
        print(f"  Records: {len(self.df_test)}")
        print(f"  Unique students: {self.df_test['student_code'].nunique()}")

        return train_path, test_path

    def run_exploration(self, save_output: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full data exploration and cleaning pipeline.

        Args:
            save_output: Whether to save the final datasets

        Returns:
            Tuple of (train_df, test_df)
        """
        print("=" * 80)
        print("DATA EXPLORATION AND CLEANING PIPELINE")
        print("=" * 80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Separate test holdout FIRST (before any cleaning)
        self.separate_test_holdout()

        # Step 3: Select relevant columns
        self.select_columns()

        # Step 4: Handle duplicates
        self.handle_duplicates()

        # Step 5: Add text length and filter short texts
        self.add_text_length()
        self.filter_short_texts()

        # Step 6: Remove low-quality students
        self.remove_low_quality_students()

        # Print final statistics
        print("\n" + "=" * 80)
        print("FINAL DATASET STATISTICS")
        print("=" * 80)

        train_stats: DatasetStats = self.compute_stats(self.df)
        train_stats.print_summary("Training Set")

        test_stats: DatasetStats = self.compute_stats(self.df_test)
        test_stats.print_summary("Test Set")

        # Save if requested
        if save_output:
            self.save_datasets()

        # Print cleaning report
        self.report.print_summary()

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)

        return self.df, self.df_test


def main() -> None:
    """Main entry point for the data exploration script."""
    # Define paths relative to project root
    project_root: Path = Path(__file__).parent.parent
    input_path: Path = project_root / '_data' / 'aggregated' / 'labeled_data.csv'
    output_dir: Path = project_root / '_data' / 'final'

    print("Hungarian Legal Text Readability - Data Exploration")
    print("-" * 60)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Create explorer and run pipeline
    explorer: DataExplorer = DataExplorer(input_path, output_dir)
    train_df, test_df = explorer.run_exploration(save_output=True)

    print("\nFiles ready for model training:")
    print(f"  - {output_dir / 'train.csv'}")
    print(f"  - {output_dir / 'test.csv'}")


if __name__ == '__main__':
    main()
