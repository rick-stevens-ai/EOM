#!/usr/bin/env python3
"""
Script to extract summary tables from dataset extraction .txt files.
Creates CSV files with dataset information preserving original columns.
"""

import pandas as pd
import re
from pathlib import Path
import argparse

def parse_dataset_file(file_path):
    """
    Parse a dataset extraction .txt file and extract dataset information.
    
    Args:
        file_path (str): Path to the .txt file
        
    Returns:
        list: List of dictionaries containing dataset information
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    datasets = []
    
    # Find all dataset entries
    dataset_pattern = r'(\d+)\.\s*\*\*DATASET\s*#\d+:\*\*\s*([^\n]+)'
    dataset_matches = re.finditer(dataset_pattern, content)
    
    for match in dataset_matches:
        dataset_num = match.group(1)
        dataset_title = match.group(2).strip()
        
        # Find the start and end of this dataset entry
        start_pos = match.start()
        
        # Find the next dataset or end of file
        next_dataset = re.search(r'\n\d+\.\s*\*\*DATASET\s*#\d+:', content[start_pos + 1:])
        if next_dataset:
            end_pos = start_pos + 1 + next_dataset.start()
        else:
            # Look for end marker or end of file
            end_marker = re.search(r'\n---\s*$', content[start_pos:])
            if end_marker:
                end_pos = start_pos + end_marker.start()
            else:
                end_pos = len(content)
        
        dataset_text = content[start_pos:end_pos]
        
        # Extract information using bullet points
        dataset_info = {
            'Dataset_Number': dataset_num,
            'Dataset_Title': dataset_title,
            'Type': extract_field(dataset_text, r'-\s*\*\*Type:\*\*\s*([^\n]+)'),
            'Scale': extract_field(dataset_text, r'-\s*\*\*Scale:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)'),
            'Aggregate_Scale': extract_field(dataset_text, r'-\s*\*\*Aggregate Scale:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)'),
            'Creation': extract_field(dataset_text, r'-\s*\*\*Creation:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)'),
            'Owner': extract_field(dataset_text, r'-\s*\*\*Owner:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)'),
            'Access': extract_field(dataset_text, r'-\s*\*\*Access:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)'),
            'Open_Access_Status': extract_field(dataset_text, r'-\s*\*\*Open Access Status:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)'),
            'Usage': extract_field(dataset_text, r'-\s*\*\*Usage:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)'),
            'Additional_Info': extract_field(dataset_text, r'-\s*\*\*Additional Info:\*\*\s*([^\n]+(?:\n(?!-\s*\*\*)[^\n]+)*)')
        }
        
        # Add source file information
        dataset_info['Source_File'] = Path(file_path).stem
        
        datasets.append(dataset_info)
    
    return datasets

def extract_field(text, pattern):
    """
    Extract a field from dataset text using regex pattern.
    
    Args:
        text (str): Text to search
        pattern (str): Regex pattern
        
    Returns:
        str: Extracted field value or empty string
    """
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        value = match.group(1).strip()
        # Clean up multiline values
        value = re.sub(r'\n\s*', ' ', value)
        return value
    return ''

def process_all_files(input_dir, output_file):
    """
    Process all dataset extraction .txt files in a directory.
    
    Args:
        input_dir (str): Directory containing .txt files
        output_file (str): Output CSV file path
    """
    input_path = Path(input_dir)
    txt_files = list(input_path.glob("*_dataset_extraction.txt"))
    
    all_datasets = []
    
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        datasets = parse_dataset_file(txt_file)
        all_datasets.extend(datasets)
    
    if all_datasets:
        df = pd.DataFrame(all_datasets)
        df.to_csv(output_file, index=False)
        print(f"Extracted {len(all_datasets)} datasets from {len(txt_files)} files")
        print(f"Summary table saved to: {output_file}")
    else:
        print("No datasets found in any files")

def main():
    parser = argparse.ArgumentParser(description="Extract summary tables from dataset extraction .txt files")
    parser.add_argument("-i", "--input", default=".", help="Input directory containing .txt files (default: current directory)")
    parser.add_argument("-o", "--output", default="datasets_summary.csv", help="Output CSV file (default: datasets_summary.csv)")
    
    args = parser.parse_args()
    
    process_all_files(args.input, args.output)

if __name__ == "__main__":
    main()