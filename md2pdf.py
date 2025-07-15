#!/usr/bin/env python3
"""
Markdown to PDF Converter

This script converts Markdown files to PDF using the markdown-pdf Node.js package.
It provides a simple command-line interface for batch conversion.

Requirements:
- markdown-pdf Node.js package (install with: npm install -g markdown-pdf)

Usage:
    python3 md2pdf.py input.md [output.pdf]
    python3 md2pdf.py *.md                    # Convert all .md files
    python3 md2pdf.py --batch directory/      # Convert all .md files in directory
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path


def check_markdown_pdf_installed():
    """Check if markdown-pdf is installed and available."""
    try:
        result = subprocess.run(['markdown-pdf', '--version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def convert_markdown_to_pdf(input_file, output_file=None):
    """
    Convert a single markdown file to PDF.
    
    Args:
        input_file (str): Path to the input markdown file
        output_file (str, optional): Path to the output PDF file
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    input_path = Path(input_file)
    
    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        return False
    
    if input_path.suffix.lower() != '.md':
        print(f"Warning: '{input_file}' does not have a .md extension.")
    
    # Determine output file name
    if output_file is None:
        output_file = input_path.with_suffix('.pdf')
    
    output_path = Path(output_file)
    
    print(f"Converting '{input_file}' to '{output_path}'...")
    
    try:
        # Run markdown-pdf command
        result = subprocess.run(['markdown-pdf', str(input_path)], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # markdown-pdf creates PDF with same name as MD file
            default_pdf = input_path.with_suffix('.pdf')
            
            # If user specified a different output name, move the file
            if str(output_path) != str(default_pdf):
                if default_pdf.exists():
                    default_pdf.rename(output_path)
            
            print(f"✅ Successfully converted to '{output_path}'")
            return True
        else:
            print(f"❌ Error converting '{input_file}':")
            print(f"   {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout while converting '{input_file}'")
        return False
    except Exception as e:
        print(f"❌ Unexpected error converting '{input_file}': {e}")
        return False


def batch_convert(directory_or_pattern):
    """
    Convert multiple markdown files to PDF.
    
    Args:
        directory_or_pattern (str): Directory path or glob pattern
        
    Returns:
        tuple: (successful_count, total_count)
    """
    path = Path(directory_or_pattern)
    
    if path.is_dir():
        # Convert all .md files in directory
        md_files = list(path.glob('*.md'))
    else:
        # Treat as glob pattern
        md_files = [Path(f) for f in glob.glob(directory_or_pattern)]
    
    if not md_files:
        print(f"No markdown files found in '{directory_or_pattern}'")
        return 0, 0
    
    print(f"Found {len(md_files)} markdown file(s) to convert...")
    
    successful = 0
    for md_file in md_files:
        if convert_markdown_to_pdf(str(md_file)):
            successful += 1
    
    return successful, len(md_files)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Markdown files to PDF using markdown-pdf',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s report.md                    # Convert single file
  %(prog)s report.md output.pdf         # Convert with custom output name
  %(prog)s *.md                         # Convert all .md files in current dir
  %(prog)s --batch documents/           # Convert all .md files in directory
  %(prog)s --batch "reports/*.md"       # Convert using glob pattern
        """
    )
    
    parser.add_argument('input', nargs='?', 
                       help='Input markdown file or glob pattern')
    parser.add_argument('output', nargs='?',
                       help='Output PDF file (optional, for single file conversion)')
    parser.add_argument('--batch', metavar='DIR_OR_PATTERN',
                       help='Batch convert all .md files in directory or matching pattern')
    parser.add_argument('--check', action='store_true',
                       help='Check if markdown-pdf is installed')
    
    args = parser.parse_args()
    
    # Check installation
    if args.check or not args.input and not args.batch:
        if check_markdown_pdf_installed():
            print("✅ markdown-pdf is installed and available")
            if args.check:
                return 0
        else:
            print("❌ markdown-pdf is not installed or not available")
            print("Install it with: npm install -g markdown-pdf")
            return 1
    
    # Validate arguments
    if not args.input and not args.batch:
        parser.print_help()
        return 1
    
    if not check_markdown_pdf_installed():
        print("❌ markdown-pdf is not installed. Install it with:")
        print("npm install -g markdown-pdf")
        return 1
    
    # Batch conversion
    if args.batch:
        successful, total = batch_convert(args.batch)
        print(f"\nConversion complete: {successful}/{total} files converted successfully")
        return 0 if successful == total else 1
    
    # Single file conversion or glob pattern
    if '*' in args.input or '?' in args.input:
        # Glob pattern
        successful, total = batch_convert(args.input)
        print(f"\nConversion complete: {successful}/{total} files converted successfully")
        return 0 if successful == total else 1
    else:
        # Single file
        success = convert_markdown_to_pdf(args.input, args.output)
        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())