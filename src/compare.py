#!/usr/bin/env python3
"""
CSV Comparison Script
Compares two CSV files line by line and reports differences.
"""

import csv
import sys
from pathlib import Path


def compare_csvs(file1_path, file2_path, show_all=False):
    """
    Compare two CSV files line by line.

    Args:
        file1_path: Path to first CSV file
        file2_path: Path to second CSV file
        show_all: If True, show all rows including matches
    """
    # Check if files exist
    if not Path(file1_path).exists():
        print(f"Error: File '{file1_path}' not found")
        return
    if not Path(file2_path).exists():
        print(f"Error: File '{file2_path}' not found")
        return

    # Read both CSV files
    with open(file1_path, 'r', encoding='utf-8') as f1:
        reader1 = list(csv.reader(f1))

    with open(file2_path, 'r', encoding='utf-8') as f2:
        reader2 = list(csv.reader(f2))

    # Get file lengths
    len1, len2 = len(reader1), len(reader2)
    max_len = max(len1, len2)

    print(f"Comparing: {file1_path} ({len1} rows) vs {file2_path} ({len2} rows)")
    print("=" * 80)

    differences_found = 0
    matches_found = 0

    # Compare line by line
    for i in range(max_len):
        row1 = reader1[i] if i < len1 else None
        row2 = reader2[i] if i < len2 else None

        # Check if rows are different
        if row1 != row2:
            differences_found += 1
            print(f"\n[DIFFERENCE] Row {i + 1}:")

            if row1 is None:
                print(f"  File 1: <missing>")
                print(f"  File 2: {row2}")
            elif row2 is None:
                print(f"  File 1: {row1}")
                print(f"  File 2: <missing>")
            else:
                print(f"  File 1: {row1}")
                print(f"  File 2: {row2}")

                # Show column-by-column differences
                max_cols = max(len(row1), len(row2))
                for col in range(max_cols):
                    val1 = row1[col] if col < len(row1) else "<missing>"
                    val2 = row2[col] if col < len(row2) else "<missing>"
                    if val1 != val2:
                        print(f"    Column {col + 1}: '{val1}' != '{val2}'")
        else:
            matches_found += 1
            if show_all:
                print(f"\n[MATCH] Row {i + 1}: {row1}")

    # Summary
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total rows compared: {max_len}")
    print(f"  Matching rows: {matches_found}")
    print(f"  Different rows: {differences_found}")

    if differences_found == 0:
        print("\n✓ Files are identical!")
    else:
        print(f"\n✗ Files differ in {differences_found} row(s)")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python compare_csvs.py <file1.csv> <file2.csv> [--show-all]")
        print("\nOptions:")
        print("  --show-all    Show all rows including matches")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    show_all = "--show-all" in sys.argv

    compare_csvs(file1, file2, show_all)


if __name__ == "__main__":
    main()
