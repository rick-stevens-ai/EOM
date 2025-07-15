# EOM (Extract-O-Matic)

**Comprehensive Bioinformatics Text Mining and Analysis Toolkit**

A powerful suite of AI-powered tools for extracting, analyzing, and processing scientific information from bioinformatics literature. This toolkit combines multiple extraction modes with advanced data analysis capabilities.

## 🚀 Features

### Core Extraction Capabilities
- **9 Extraction Modes:**
  - Workflow extraction and BV-BRC service mapping
  - Scientific problem summarization
  - Experimental protocol identification
  - Dataset extraction with comprehensive metadata
  - Hypothesis extraction with Wisteria JSON compatibility
  - Document classification (TOOL vs PROBLEM)
  - Workflow calculus notation generation
  - Automated protocol mapping for robotics/AGI

### Advanced Processing
- **Rich UI** with progress tracking and visual feedback
- **Parallel processing** support for high-throughput analysis
- **Batch mode** for processing multiple files
- **LaTeX and PDF** text extraction
- **Token-based and character-based** intelligent chunking
- **Document classification** and filtering

### Analysis Pipeline
- **Dataset summarization** from extraction results
- **LLM-powered analysis** with domain classification
- **Parallel chunk processing** for large datasets
- **PDF report generation** from analysis results
- **Complete pipeline automation**

## 📦 Toolkit Components

| Tool | Purpose | Key Features |
|------|---------|-------------|
| `extract-o-matic.py` | Main extraction engine | 9 modes, parallel processing, rich UI |
| `extract_summary_tables.py` | CSV generation from extractions | Structured data export |
| `analyze_datasets.py` | LLM-powered dataset analysis | Domain classification, trend analysis |
| `md2pdf.py` | Markdown to PDF conversion | Batch processing, report generation |
| `run_pipeline.py` | Complete workflow automation | End-to-end processing |
| `run_pipeline.sh` | Shell wrapper script | Simple execution interface |

## 🛠 Installation

### Core Dependencies
```bash
pip install openai tiktoken rich pyyaml pandas
```

### Optional Dependencies
```bash
# For LaTeX and PDF processing
pip install pylatexenc PyPDF2

# For PDF generation (requires Node.js)
npm install -g markdown-pdf
```

### Configuration
Copy and configure the model server settings:
```bash
cp argo_model_servers.yaml model_servers.yaml
# Edit model_servers.yaml with your API configurations
```

## 🎯 Quick Start

### Single File Extraction
```bash
# Extract datasets from a paper
python extract-o-matic.py paper.pdf --mode dataset_extraction --model gpt-4.1 --output datasets.txt

# Extract hypotheses with JSON output
python extract-o-matic.py paper.txt --mode hypothesis_extraction --model gpt-4.1 --output hypotheses.json
```

### Batch Processing
```bash
# Process multiple papers
python extract-o-matic.py papers/ --mode dataset_extraction --model gpt-4.1 --output results_dir/ --workers 4
```

### Complete Pipeline
```bash
# Run full analysis pipeline
python run_pipeline.py --input-dir paper_extractions/ --model gpt-4.1 --workers 4

# Or use the shell wrapper
./run_pipeline.sh --input-dir paper_extractions/ --model gpt-4.1
```

## 📊 Analysis Workflow

1. **Extract Information**: Use extract-o-matic.py to extract datasets/workflows/hypotheses
2. **Generate Summary**: Create CSV summaries with extract_summary_tables.py  
3. **Analyze Data**: Run LLM analysis with analyze_datasets.py
4. **Generate Report**: Convert markdown results to PDF with md2pdf.py

### Pipeline Example
```bash
# Complete automated workflow
python run_pipeline.py \
  --input-dir ./extractions \
  --model gpt-4.1 \
  --chunk-size 200 \
  --workers 4 \
  --output-prefix project_analysis
```

## ⚙️ Advanced Configuration

### Extraction Modes
- `workflow` - Extract bioinformatics workflows and BV-BRC mappings
- `problem_summary` - Identify scientific objectives and research questions  
- `dataset_extraction` - Comprehensive dataset metadata extraction
- `hypothesis_extraction` - Scientific hypothesis extraction with Wisteria JSON
- `experimental_protocol` - Laboratory protocol identification
- `automated_protocol` - Robotics/AGI automation mapping
- `classification` - Document type classification
- `bvbrc_mapping` - Map problems to BV-BRC tools
- `workflow_calculus` - Generate workflow calculus notation

### Model Configuration
Edit `model_servers.yaml` to configure AI models:
```yaml
servers:
- server: your_server
  shortname: model_name
  openai_api_key: your_key
  openai_api_base: https://api.your-server.com/v1
  openai_model: model_id
```

### Performance Tuning
```bash
# High-throughput processing
python extract-o-matic.py papers/ \
  --workers 8 \
  --chunk-size 3000 \
  --character-limit 200000 \
  --mode dataset_extraction
```

## 📈 Output Formats

### Text Extraction Results
- Structured text files with extraction results
- Individual files per paper in batch mode
- Rich formatting with progress indicators

### Dataset Analysis
- **CSV files** with structured dataset information
- **JSON files** with detailed analysis results  
- **Markdown reports** with comprehensive analysis
- **PDF reports** for presentation/sharing

### Hypothesis Extraction
- **Wisteria-compatible JSON** for hypothesis management
- Structured hypothesis data with scientific rigor analysis
- References and experimental validation plans

## 🔧 Troubleshooting

### Common Issues
- **Missing dependencies**: Install optional packages for full functionality
- **Model configuration**: Verify API keys and endpoints in model_servers.yaml
- **Memory issues**: Reduce chunk-size or worker count for large documents
- **PDF conversion**: Ensure Node.js and markdown-pdf are installed

### Debug Mode
```bash
# Enable verbose output
python extract-o-matic.py paper.txt --mode dataset_extraction --model gpt-4.1 --workers 1

# Keep debug files for analysis
python analyze_datasets.py --keep-debug datasets.csv
```

## 📄 License

Open source bioinformatics toolkit for scientific literature analysis and workflow extraction.

## 🤝 Contributing

This toolkit combines features from multiple extraction scripts and provides a unified interface for bioinformatics text mining. Contributions welcome for additional extraction modes and analysis capabilities.