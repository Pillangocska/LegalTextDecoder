"""
Dataset Initialization Script

This script processes labeled JSON files from multiple students/annotators and creates
an aggregated CSV file with all the labeled data and metadata.

Input: data/original/<STUDENT_CODE>/*.json files
Output: data/aggregated/labeled_data.csv

Each student folder contains JSON files with labeled ASZF (Terms and Conditions) paragraphs
rated on a 1-5 readability scale.
"""

from pathlib import Path
import json
import csv
import re

from src.util.config_manager import config
from src.util.logger import Logger

logger = Logger("aggregate_jsons")

excluded_folders = config.get("preprocess.folders_to_exclude")
input_dir = config.get("preprocess.user_input_dir")
output_dir = config.get("preprocess.aggregated_dir")

def extract_label_number(label_text):
    """
    Extract the numeric label (1-5) from label text like '4-Ertheto'

    Args:
        label_text: String containing the label, e.g., "4-Ertheto" or "5-Konnyen ertheto"

    Returns:
        Integer label value (1-5) or None if not found
    """
    match = re.match(r'^(\d+)', label_text)
    if match:
        return int(match.group(1))
    return None


def process_json_file(json_path, student_code):
    """
    Process a single JSON file and extract labeled data

    Args:
        json_path: Path to the JSON file
        student_code: Student identifier (folder name)

    Returns:
        List of dictionaries containing the extracted data
    """
    results = []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            data = [data]

        for task in data:
            # Skip if no annotations
            if not task.get('annotations'):
                continue

            # Get the text
            text = task.get('data', {}).get('text', '')
            if not text:
                continue

            # Get file upload info (source document)
            source_file = task.get('file_upload', '')

            # Get task metadata
            task_id = task.get('id', '')
            task_inner_id = task.get('inner_id', '')
            task_created_at = task.get('created_at', '')
            task_updated_at = task.get('updated_at', '')

            # Process each annotation
            for annotation in task.get('annotations', []):
                # Extract annotation metadata
                annotation_id = annotation.get('id', '')
                completed_by = annotation.get('completed_by', '')
                created_at = annotation.get('created_at', '')
                updated_at = annotation.get('updated_at', '')
                lead_time = annotation.get('lead_time', '')

                # Extract the label from result
                result = annotation.get('result', [])
                if not result:
                    continue

                # Get the choice (label)
                choices = result[0].get('value', {}).get('choices', [])
                if not choices:
                    continue

                label_text = choices[0]
                label_numeric = extract_label_number(label_text)

                # Create a record
                record = {
                    'student_code': student_code,
                    'source_file': source_file,
                    'json_filename': Path(json_path).name,
                    'task_id': task_id,
                    'task_inner_id': task_inner_id,
                    'annotation_id': annotation_id,
                    'text': text,
                    'label_text': label_text,
                    'label_numeric': label_numeric,
                    'completed_by': completed_by,
                    'annotation_created_at': created_at,
                    'annotation_updated_at': updated_at,
                    'task_created_at': task_created_at,
                    'task_updated_at': task_updated_at,
                    'lead_time_seconds': lead_time,
                }

                results.append(record)

    except Exception as e:
        logger.error(f"Error processing {json_path}: {str(e)}", exc_info=e)

    return results


def aggregate_labeled_data(input_dir, output_dir):
    """
    Aggregate all labeled data from student folders into a single CSV

    Args:
        input_dir: Path to data/original/ directory
        output_dir: Path to data/aggregated/ directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    all_records = []

    # Iterate through student folders
    for student_folder in sorted(input_path.iterdir()):
        if not student_folder.is_dir():
            continue

        # Skip excluded folders
        if student_folder.name.lower() in excluded_folders:
            logger.info(f"Skipping excluded folder: {student_folder.name}")
            continue

        student_code = student_folder.name
        logger.info(f"Processing student: {student_code}")

        # Process all JSON files in the student folder
        json_files = list(student_folder.glob('*.json'))
        logger.info(f"  Found {len(json_files)} JSON file(s)")

        for json_file in json_files:
            records = process_json_file(json_file, student_code)
            all_records.extend(records)
            logger.info(f"    {json_file.name}: {len(records)} records")

    # Write to CSV
    output_file = output_path / 'labeled_data.csv'

    if all_records:
        fieldnames = [
            'student_code',
            'source_file',
            'json_filename',
            'task_id',
            'task_inner_id',
            'annotation_id',
            'text',
            'label_text',
            'label_numeric',
            'completed_by',
            'annotation_created_at',
            'annotation_updated_at',
            'task_created_at',
            'task_updated_at',
            'lead_time_seconds',
        ]

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

        logger.info("Aggregation complete!")
        logger.info(f"Total records: {len(all_records)}")
        logger.info(f"Output file: {output_file}")

        # Print some statistics
        unique_students = len(set(r['student_code'] for r in all_records))
        unique_texts = len(set(r['text'] for r in all_records))
        label_distribution = {}
        for record in all_records:
            label = record['label_numeric']
            label_distribution[label] = label_distribution.get(label, 0) + 1

        logger.info("Statistics:")
        logger.info(f"  Unique students: {unique_students}")
        logger.info(f"  Unique texts: {unique_texts}")
        logger.info("  Label distribution:")
        for label in sorted(label_distribution.keys()):
            if label is not None:
                logger.info(f"    Label {label}: {label_distribution[label]} records")
    else:
        logger.warning("\nNo records found to aggregate.")


if __name__ == '__main__':
    logger.info("Starting data aggregation...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("-" * 60)

    aggregate_labeled_data(input_dir, output_dir)
