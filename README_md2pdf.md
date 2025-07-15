# Markdown to PDF Converter

A Python script that converts Markdown files to PDF using the `markdown-pdf` Node.js package.

## Prerequisites

1. **Node.js** and **npm** must be installed
2. **markdown-pdf** package must be installed globally:
   ```bash
   npm install -g markdown-pdf
   ```

## Installation Check

To verify that all dependencies are installed:
```bash
python3 md2pdf.py --check
```

## Usage

### Convert a single file
```bash
# Convert with automatic naming (creates report.pdf)
python3 md2pdf.py report.md

# Convert with custom output name
python3 md2pdf.py report.md custom_name.pdf
```

### Batch convert multiple files
```bash
# Convert all .md files in current directory
python3 md2pdf.py *.md

# Convert all .md files in a specific directory
python3 md2pdf.py --batch documents/

# Convert files matching a pattern
python3 md2pdf.py --batch "reports/*.md"
```

### Examples

```bash
# Single file conversion
python3 md2pdf.py datasets_summary_analysis_report.md

# Batch convert all markdown files in current directory
python3 md2pdf.py --batch ./

# Convert specific pattern
python3 md2pdf.py --batch "*_report.md"
```

## Features

- ✅ Single file conversion
- ✅ Batch conversion of multiple files
- ✅ Glob pattern support
- ✅ Custom output file naming
- ✅ Error handling and validation
- ✅ Progress reporting
- ✅ Dependency checking

## Output

The script will:
- Validate input files exist and have .md extension
- Show progress for each conversion
- Report success/failure for each file
- Provide summary statistics for batch operations

## Error Handling

The script handles:
- Missing input files
- Invalid file extensions (with warnings)
- markdown-pdf installation issues
- Conversion timeouts
- File permission errors

## Notes

- PDF files are created in the same directory as the source Markdown files
- Existing PDF files will be overwritten
- The script uses the same conversion method that successfully generated your analysis report