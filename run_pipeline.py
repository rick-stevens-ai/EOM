#!/usr/bin/env python3
"""
Pipeline script to run extract_summary_tables, analyze_datasets, and md2pdf in sequence.

This script automates the complete workflow:
1. Extract summary tables from dataset extraction files
2. Analyze datasets using LLM 
3. Convert analysis report to PDF

Usage:
    python3 run_pipeline.py [options]

Options:
    --input-dir DIR         Input directory for dataset extraction files (default: current directory)
    --model MODEL          Model to use for analysis (default: gpt-4.1)
    --config CONFIG        Model configuration file (default: argo_model_servers.yaml)
    --chunk-size SIZE      Number of datasets per chunk (default: 200)
    --workers NUM          Number of parallel workers (default: 4)
    --output-prefix PREFIX Prefix for output files (default: datasets_summary)
    --skip-extract         Skip the extract_summary_tables step
    --skip-analyze         Skip the analyze_datasets step
    --skip-pdf             Skip the md2pdf step
    --verbose              Enable verbose output
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, verbose=False):
    """Run a command and handle errors."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    print(f"Step: {description}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if verbose and result.stdout:
            print(result.stdout)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete dataset analysis pipeline")
    parser.add_argument("--input-dir", default=".", help="Input directory for dataset extraction files")
    parser.add_argument("--model", default="gpt-4.1", help="Model to use for analysis")
    parser.add_argument("--config", default="argo_model_servers.yaml", help="Model configuration file")
    parser.add_argument("--chunk-size", type=int, default=200, help="Number of datasets per chunk")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output-prefix", default="datasets_summary", help="Prefix for output files")
    parser.add_argument("--skip-extract", action="store_true", help="Skip the extract_summary_tables step")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip the analyze_datasets step")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip the md2pdf step")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Determine file names
    csv_file = f"{args.output_prefix}.csv"
    md_file = f"{args.output_prefix}_analysis_report.md"
    pdf_file = f"{args.output_prefix}_analysis_report.pdf"
    
    success = True
    
    print("=" * 60)
    print("Dataset Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Extract summary tables
    if not args.skip_extract:
        cmd = ["python3", "extract_summary_tables.py", "-i", args.input_dir, "-o", csv_file]
        if not run_command(cmd, "Extracting summary tables", args.verbose):
            success = False
    else:
        print("⏭  Skipping extract_summary_tables step")
        if not os.path.exists(csv_file):
            print(f"✗ CSV file {csv_file} not found, but extract step was skipped")
            success = False
    
    # Step 2: Analyze datasets
    if success and not args.skip_analyze:
        cmd = [
            "python3", "analyze_datasets.py", csv_file,
            "--model", args.model,
            "--config", args.config,
            "--chunk-size", str(args.chunk_size),
            "--workers", str(args.workers),
            "--output", md_file
        ]
        if not run_command(cmd, "Analyzing datasets with LLM", args.verbose):
            success = False
    elif not args.skip_analyze:
        print("⏭  Skipping analyze_datasets step due to previous failure")
    else:
        print("⏭  Skipping analyze_datasets step")
        if not os.path.exists(md_file):
            print(f"✗ Markdown file {md_file} not found, but analyze step was skipped")
            success = False
    
    # Step 3: Convert to PDF
    if success and not args.skip_pdf:
        cmd = ["python3", "md2pdf.py", md_file]
        if not run_command(cmd, "Converting markdown to PDF", args.verbose):
            success = False
    elif not args.skip_pdf:
        print("⏭  Skipping md2pdf step due to previous failure")
    else:
        print("⏭  Skipping md2pdf step")
    
    print("=" * 60)
    if success:
        print("✓ Pipeline completed successfully!")
        print(f"  CSV file: {csv_file}")
        if not args.skip_analyze:
            print(f"  Markdown report: {md_file}")
        if not args.skip_pdf:
            print(f"  PDF report: {pdf_file}")
    else:
        print("✗ Pipeline failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()