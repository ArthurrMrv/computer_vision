#!/usr/bin/env python3
"""
CSV Column Translator Script

Reads a CSV file and converts a specified column to lowercase string values.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def translate_column_to_lowercase(csv_path, column_name, output_path=None, inplace=False):
    """
    Read a CSV file and convert a specified column to lowercase string values.
    
    Args:
        csv_path: Path to the input CSV file
        column_name: Name of the column to convert to lowercase
        output_path: Optional path for output CSV file. If None and inplace=False, 
                    creates a new file with '_lowercase' suffix
        inplace: If True, overwrites the original file
    
    Returns:
        DataFrame with the converted column
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully read CSV file: {csv_path}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"✗ Error: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        sys.exit(1)
    
    # Check if column exists
    if column_name not in df.columns:
        print(f"✗ Error: Column '{column_name}' not found in CSV file.")
        print(f"  Available columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)
    
    # Show original column info
    print(f"\nColumn '{column_name}' info:")
    print(f"  Data type: {df[column_name].dtype}")
    print(f"  Sample values (before): {df[column_name].head(5).tolist()}")
    
    # Convert to lowercase string
    df[column_name] = df[column_name].astype(str).str.lower()
    
    # Show converted column info
    print(f"\nAfter conversion:")
    print(f"  Data type: {df[column_name].dtype}")
    print(f"  Sample values (after): {df[column_name].head(5).tolist()}")
    
    # Determine output path
    if inplace:
        output_path = csv_path
        print(f"\n✓ Converting in-place (overwriting original file)")
    elif output_path is None:
        # Create new filename with _lowercase suffix
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_lowercase{csv_file.suffix}"
        print(f"\n✓ Saving to new file: {output_path}")
    else:
        print(f"\n✓ Saving to: {output_path}")
    
    # Save the result
    try:
        df.to_csv(output_path, index=False)
        print(f"✓ Successfully saved CSV file: {output_path}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error saving CSV file: {e}")
        sys.exit(1)
    
    return df


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert a CSV column to lowercase string values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert 'Label' column and save to new file
  python translator.py data.csv Label
  
  # Convert 'Label' column and overwrite original file
  python translator.py data.csv Label --inplace
  
  # Convert 'Label' column and save to specific output file
  python translator.py data.csv Label --output output.csv
        """
    )
    
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to the input CSV file'
    )
    
    parser.add_argument(
        'column_name',
        type=str,
        help='Name of the column to convert to lowercase'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: creates new file with _lowercase suffix)'
    )
    
    parser.add_argument(
        '-i', '--inplace',
        action='store_true',
        help='Overwrite the original file (use with caution)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.inplace and args.output:
        print("✗ Error: Cannot use both --inplace and --output options together.")
        sys.exit(1)
    
    # Run the translation
    translate_column_to_lowercase(
        csv_path=args.csv_path,
        column_name=args.column_name,
        output_path=args.output,
        inplace=args.inplace
    )


if __name__ == '__main__':
    #main()
    translate_column_to_lowercase("predictions_V5.csv", "Label", output_path="predictions_V5_lowercase.csv")

