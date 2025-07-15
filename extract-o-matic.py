#!/usr/bin/env python3
"""
extract-o-matic.py - Extract-o-matic Bioinformatics Workflow Extraction Script

This script combines the best features from multiple extraction scripts:
- YAML-based model configuration from model_servers.yaml
- Rich UI and progress tracking from make_v21.py
- Workflow extraction prompts from new_extract_v3.py
- Workflow Calculus notation from l33_process_summary_to_workflow.py
- Knowledge graph extraction from graph-o-matic.py (graph-o-matic.py is now deprecated)

Features:
- 11 extraction modes including enhanced knowledge_graph mode
- Enhanced experimental_protocol mode with post-processing consolidation
- RDF output generation for knowledge graphs
- Semantic relationship extraction with confidence scoring
- Knowledge graph connectivity optimization (eliminates isolated nodes)
- Cross-chunk relationship detection for better graph connectivity
- Open problems extraction with structured JSON output
- Parallel processing and batch mode support

Usage:
    python extract-o-matic.py input_file [options]
    python extract-o-matic.py paper.pdf --mode knowledge_graph --model gpt-4 --output graph.rdf
    python extract-o-matic.py paper.txt --mode experimental_protocol --model gpt-4 --output protocols.txt
"""

import os
import sys
import argparse
import time
import yaml
import glob
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Rich console imports
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table

# OpenAI and text processing imports
from openai import OpenAI
import tiktoken

# LaTeX processing (optional)
try:
    from pylatexenc.latex2text import LatexNodes2Text
    LATEX_AVAILABLE = True
except ImportError:
    LATEX_AVAILABLE = False

# PDF processing (optional)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Global console instance
console = Console()

class ExtractionMode(Enum):
    """Extraction mode options."""
    WORKFLOW = "workflow"
    PROBLEM_SUMMARY = "problem_summary"
    BVBRC_MAPPING = "bvbrc_mapping"
    WORKFLOW_CALCULUS = "workflow_calculus"
    CLASSIFICATION = "classification"
    EXPERIMENTAL_PROTOCOL = "experimental_protocol"
    AUTOMATED_PROTOCOL = "automated_protocol"
    DATASET_EXTRACTION = "dataset_extraction"
    HYPOTHESIS_EXTRACTION = "hypothesis_extraction"
    OPEN_PROBLEMS = "open_problems"
    KNOWLEDGE_GRAPH = "knowledge_graph"

@dataclass
class ModelConfig:
    """Model configuration from YAML file."""
    server: str
    shortname: str
    openai_api_key: str
    openai_api_base: str
    openai_model: str

@dataclass
class ExtractionConfig:
    """Configuration for extraction process."""
    input_file: str
    output_file: str
    mode: ExtractionMode
    model_config: ModelConfig
    chunk_size: int
    max_tokens: int
    temperature: float
    character_limit: int
    guidance_files: Dict[str, str]
    require_problem: bool = False
    require_tool: bool = False
    batch_mode: bool = False
    input_files: List[str] = None
    worker_count: int = 1

class ExtractOMatic:
    """Extract-o-matic main extraction class that handles all extraction modes."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.console = Console()
        self.client = None
        self.tokenizer = None
        self.total_chunks = 0
        self.processed_chunks = 0
        self.results = []
        self.results_lock = Lock()
        self.progress_lock = Lock()
        
        # Initialize OpenAI client
        self._initialize_client()
        
        # Initialize tokenizer
        self._initialize_tokenizer()
        
        # Load guidance files
        self.guidance = self._load_guidance_files()
    
    def _initialize_client(self):
        """Initialize OpenAI client with model configuration."""
        try:
            self.client = OpenAI(
                api_key=self.config.model_config.openai_api_key,
                base_url=self.config.model_config.openai_api_base
            )
            self.console.print(f"✅ Initialized client for {self.config.model_config.shortname}")
        except Exception as e:
            self.console.print(f"❌ Failed to initialize client: {e}")
            sys.exit(1)
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for text chunking."""
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
            self.console.print("✅ Initialized tokenizer")
        except Exception as e:
            self.console.print(f"❌ Failed to initialize tokenizer: {e}")
            # Fallback to basic chunking
            self.tokenizer = None
    
    def _load_guidance_files(self) -> Dict[str, str]:
        """Load guidance files for different extraction modes."""
        guidance = {}
        
        for name, path in self.config.guidance_files.items():
            if path and os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        guidance[name] = f.read()
                    self.console.print(f"✅ Loaded {name} guidance from {path}")
                except Exception as e:
                    self.console.print(f"⚠️  Failed to load {name} guidance: {e}")
            else:
                self.console.print(f"⚠️  {name} guidance file not found: {path}")
        
        return guidance
    
    def _read_input_file(self) -> str:
        """Read and preprocess input file."""
        return self._read_single_file(self.config.input_file)
    
    def _read_single_file(self, file_path: str) -> str:
        """Read and preprocess a single file based on its type."""
        try:
            file_type = get_file_type(file_path)
            
            if file_type == 'pdf':
                content = read_pdf_file(file_path)
            else:
                # Read text/markdown files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            if not content:
                self.console.print(f"⚠️  No content extracted from {file_path}")
                return ""
            
            # Clean LaTeX if available
            if LATEX_AVAILABLE:
                content = LatexNodes2Text().latex_to_text(content)
            
            # Check if content exceeds character limit but don't truncate
            if len(content) > self.config.character_limit:
                self.console.print(f"📄 Large document detected: {len(content)} characters for {file_path}")
                self.console.print(f"   Will be processed in chunks (limit: {self.config.character_limit})")
            
            return content
        except Exception as e:
            self.console.print(f"❌ Failed to read file {file_path}: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks with overlap to ensure complete coverage."""
        if not text.strip():
            return []
            
        if self.tokenizer:
            return self._chunk_text_with_tokens(text)
        else:
            return self._chunk_text_with_characters(text)
    
    def _chunk_text_with_tokens(self, text: str) -> List[str]:
        """Token-based chunking with overlap."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Use 20% overlap to ensure context preservation
        overlap_size = max(100, self.config.chunk_size // 5)  # Minimum 100 tokens overlap
        
        i = 0
        while i < len(tokens):
            # Calculate end position for this chunk
            end_pos = min(i + self.config.chunk_size, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[i:end_pos]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # If this is the last chunk, break
            if end_pos >= len(tokens):
                break
            
            # Move to next chunk position with overlap
            i = end_pos - overlap_size
        
        return chunks
    
    def _chunk_text_with_characters(self, text: str) -> List[str]:
        """Character-based chunking with overlap and sentence boundary awareness."""
        chunk_size_chars = self.config.chunk_size * 4  # Rough estimate (4 chars per token)
        chunks = []
        
        # Use 20% overlap
        overlap_size = max(400, chunk_size_chars // 5)  # Minimum 400 characters overlap
        
        i = 0
        while i < len(text):
            # Calculate end position for this chunk
            end_pos = min(i + chunk_size_chars, len(text))
            
            # Try to break at sentence boundaries to preserve context
            if end_pos < len(text):
                # Look for sentence endings within the last 200 characters
                search_start = max(end_pos - 200, i)
                sentence_endings = []
                
                # Find sentence endings (period, exclamation, question mark followed by space/newline)
                for j in range(search_start, end_pos):
                    if text[j] in '.!?' and j + 1 < len(text) and text[j + 1] in ' \n\t':
                        sentence_endings.append(j + 1)
                
                # Use the last sentence ending if found
                if sentence_endings:
                    end_pos = sentence_endings[-1]
                # If no sentence endings, look for paragraph breaks
                elif '\n\n' in text[search_start:end_pos]:
                    para_breaks = [search_start + m.start() + 2 for m in re.finditer(r'\n\n', text[search_start:end_pos])]
                    if para_breaks:
                        end_pos = para_breaks[-1]
            
            # Extract chunk
            chunk_text = text[i:end_pos].strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append(chunk_text)
            
            # If this is the last chunk, break
            if end_pos >= len(text):
                break
            
            # Move to next chunk position with overlap
            i = end_pos - overlap_size
        
        return chunks
    
    def _validate_chunking(self, original_text: str, chunks: List[str]) -> bool:
        """Validate that chunking covers the entire document."""
        if not chunks:
            return len(original_text.strip()) == 0
        
        # Check if the first chunk starts near the beginning of the document
        first_chunk_start = original_text.find(chunks[0][:100])  # Use first 100 chars to find start
        if first_chunk_start > 1000:  # Allow some flexibility for whitespace differences
            self.console.print(f"⚠️  Warning: First chunk doesn't start near document beginning (offset: {first_chunk_start})")
        
        # Check if the last chunk ends near the end of the document
        last_chunk_end = original_text.rfind(chunks[-1][-100:])  # Use last 100 chars to find end
        if last_chunk_end != -1:
            remaining_text = original_text[last_chunk_end + len(chunks[-1][-100:]):].strip()
            if len(remaining_text) > 500:  # Allow some flexibility
                self.console.print(f"⚠️  Warning: {len(remaining_text)} characters may not be covered at document end")
        
        return True
    
    def _get_prompt_for_mode(self, mode: ExtractionMode, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate appropriate prompt based on extraction mode."""
        
        if mode == ExtractionMode.WORKFLOW:
            return self._get_workflow_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.PROBLEM_SUMMARY:
            return self._get_problem_summary_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.BVBRC_MAPPING:
            return self._get_bvbrc_mapping_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.WORKFLOW_CALCULUS:
            return self._get_workflow_calculus_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.CLASSIFICATION:
            return self._get_classification_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.EXPERIMENTAL_PROTOCOL:
            return self._get_experimental_protocol_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.AUTOMATED_PROTOCOL:
            return self._get_automated_protocol_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.DATASET_EXTRACTION:
            return self._get_dataset_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.HYPOTHESIS_EXTRACTION:
            return self._get_hypothesis_extraction_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.OPEN_PROBLEMS:
            return self._get_open_problems_prompt(chunk_text, chunk_idx, total_chunks)
        elif mode == ExtractionMode.KNOWLEDGE_GRAPH:
            return self._get_knowledge_graph_prompt(chunk_text, chunk_idx, total_chunks)
        else:
            raise ValueError(f"Unknown extraction mode: {mode}")
    
    def _get_workflow_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate workflow extraction prompt (from extract_workflow.py)."""
        return f"""Please extract the bioinformatics workflow used in the following text (part {chunk_idx + 1} of {total_chunks}). 
Identify the primary objective of the author in using the tool, and indicate which services in BV-BRC would be needed to replicate that workflow.

Text:
{chunk_text}"""
    
    def _get_problem_summary_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate problem summary prompt (from extract_problem_summary.py)."""
        return f"""
Identify the primary scientific objective(s) of the author(s) and identify the key bioinformatics analysis needed to meet these objectives.

Please rephrase the goal as a series of questions that could be answered using a powerful bioinformatics environment such as BV-BRC. These questions should be specific search terms and analysis that could be done to meet the goal.

Consider if this problem would make a good tutorial for using the system or a good demonstration for an AI to demonstrate using the tools in an automatic fashion. If you think this is a good example for that, then finish the output with the words "GOOD EXAMPLE".

If the analysis mentioned in the workflow can be done without uploading new data to BV-BRC, then finish the output with the words "POSSIBLY GREAT EXAMPLE"

Text:
{chunk_text}"""
    
    def _get_bvbrc_mapping_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate BV-BRC mapping prompt (from compare_to_bvbrc.py)."""
        bvbrc_services = """
BV-BRC contains the following services and tools:

1. Genome Assembly Service: Allows assembly of bacterial genomes using multiple assemblers for comparison.
2. Genome Annotation Service: Annotates bacterial genomes with RASTtk and viral genomes with VIGOR4.
3. Comprehensive Genome Analysis Service: Performs assembly, annotation, and comparative analysis of bacterial genomes.
4. BLAST Service: Searches against public or private genomes using DNA or protein sequences to find matches.
5. Primer Design Service: Designs primers from input sequences using Primer3 for PCR, hybridization, and sequencing.
6. Similar Genome Finder Service: Finds similar bacterial genomes based on genome distance estimation using Mash/MinHash.
7. Genome Alignment Service: Produces whole-genome alignments of bacterial genomes using progressiveMauve.
8. Variation Analysis Service: Identifies sequence variations by comparing samples to a reference genome.
9. Tn-Seq Analysis Service: Identifies essential genomic regions from transposon insertion sequencing data.
10. Phylogenetic Tree Service: Constructs custom phylogenetic trees using Codon Tree and RAxML methods.
11. Gene Tree Service: Builds phylogenetic trees based on user-selected genomes, genes, or proteins using FastTree or RAxML.
12. MSA and SNP/Variation Analysis Service: Aligns sequences and analyzes SNP variations from input data.
13. Meta-CATS: Compares aligned sequence positions and identifies statistically significant variations.
14. Proteome Comparison Service: Compares protein sequences across multiple genomes using BLASTP.
15. Comparative Systems Service: Compares protein families and pathways across up to 500 bacterial genomes.
16. Taxonomic Classification Service: Classifies metagenomic reads into taxonomic bins using Kraken 2.
17. Metagenomic Binning Service: Bins metagenomic reads or contigs into genome sets from environmental samples.
18. Metagenomic Read Mapping Service: Maps metagenomic reads to genes related to antibiotic resistance and virulence.
19. RNA-Seq Analysis Service: Analyzes RNA-Seq data for differential gene expression using Tuxedo or HISAT2.
20. Expression Import Service: Uploads and analyzes pre-processed differential gene expression datasets.
21. Fastq Utilities Service: Provides FASTQ file operations including trimming, quality checking, and alignment.
22. ID Mapper Tool: Maps BV-BRC identifiers to external databases or vice versa.
23. SARS-CoV-2 Genome Assembly and Annotation Service: Performs assembly, annotation, and variation analysis of SARS-CoV-2 genomes.
24. SARS-CoV-2 Wastewater Analysis Service: Analyzes wastewater samples for SARS-CoV-2 variants using Freyja.
25. Sequence Submission Service: Validates and submits virus sequences to NCBI Genbank.
26. HA Subtype Numbering Conversion Service: Converts HA protein sequence numbering to align with subtype references.
27. Subspecies Classification Service: Assigns viral genotypes/subtypes based on reference tree positions.
28. Genome Browser: Provides a graphical representation of genomic feature alignments.
29. Circular Genome Viewer: Visualizes genome alignments in an interactive circular format.
30. Compare Region Viewer: Identifies and displays proteins from the same family across different genomes.
31. Archaeopteryx.js Phylogenetic Tree Viewer: Interactive display of phylogenetic trees with customization options.
32. Multiple Sequence Alignment Viewer: Visualizes multiple sequence alignments with linked phylogenetic trees.
33. Protein Family Sorter: Examines distribution of protein families across selected genomes for pan-genome analysis.
34. Pathway Comparison Tool: Identifies metabolic pathways across genomes and visualizes them using KEGG maps.
35. Subsystems Data and Viewer: Summarizes gene functionality across genomes with heatmaps and pie charts.
"""
        
        return f"""
{bvbrc_services}

Please analyze the scientific problem and the specific questions in the following text, and make specific suggestions for how the BV-BRC collection of tools could be used to answer the questions. Be as specific as possible about the queries and prompts that would be used to answer the questions using the tools contained in BV-BRC. Where the questions need other tools please indicate so.

Text:
{chunk_text}"""
    
    def _get_workflow_calculus_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate workflow calculus prompt (from l33_process_summary_to_workflow.py)."""
        notation_guidance = self.guidance.get('notation', '[Notation guidance not available]')
        rubric_guidance = self.guidance.get('rubric', '[Rubric guidance not available]')
        
        return f"""
Please read and process the following section of a scientific article.

**Task A – High‑level summary**
1. Summarise the main goal of the paper.
2. Explain briefly how the workflow you will design addresses that goal.

**Task B – Workflow extraction**
For *each* research question or analysis described (even implicitly) in the text, emit a **compact workflow** using the Workflow Calculus v0.9 **exactly** as specified in the Notation Guidance below.
Follow the Extraction **Rubric** strictly – fill the mandatory fields and strive for rubric score 3 where information allows.

* Keep the notation pure UTF‑8 (no LaTeX, no code‑blocks).
* Compose transformations with ∘ or ⇒, parallelism with ∥, iteration with […] as needed.
* Provide an enumerated list *after* the workflows that defines every function / tool symbol you introduced (one‑line each).

---
### Rubric Guidance
{rubric_guidance}
---
### Notation Guidance
{notation_guidance}
---

#### Text chunk {chunk_idx + 1}/{total_chunks}
{chunk_text}
""".strip()
    
    def _get_classification_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate classification prompt (from classify.py)."""
        return f"""Your task is to analyze the provided text and output either 'TOOL' or 'PROBLEM' based on the criteria provided.

- If the text is about a bioinformatics tool or database (not a specific use of that on a problem), output 'TOOL'.
- If the text is about using one or more tools in a scientific workflow to solve a specific problem, output 'PROBLEM'.

Only output 'TOOL' or 'PROBLEM', nothing else.

Here is the text:

{chunk_text}"""
    
    def _get_experimental_protocol_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate experimental protocol extraction prompt."""
        return f"""Please extract all experimental protocols and laboratory procedures mentioned in the following text (part {chunk_idx + 1} of {total_chunks}). 

Focus ONLY on wet lab experimental protocols and procedures, NOT computational steps or bioinformatics analyses.

Use standard protocol names when possible. Examples include:
- PCR amplification
- DNA extraction
- Gel electrophoresis
- Bacterial culture
- Protein purification
- Western blot
- ELISA
- Flow cytometry
- Microscopy
- Sequencing library preparation
- Cell transformation
- Plasmid isolation
- Restriction enzyme digestion
- Ligation
- Transfection
- Immunofluorescence
- RT-PCR
- qPCR
- Cloning
- Mutagenesis

Return the results as a compact bulleted list using standard protocol names. If no experimental protocols are found, return "No experimental protocols identified."

Text:
{chunk_text}"""
    
    def _get_automated_protocol_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate automated protocol mapping prompt."""
        return f"""Please analyze the following scientific text (part {chunk_idx + 1} of {total_chunks}) and create a detailed automation plan for reproducing this research using three types of automated systems:

**FIXED ROBOT SYSTEMS** (Liquid handlers, plate readers, automated pipettes, etc.):
- What wet lab steps could be performed by fixed laboratory automation?
- Specify equipment needed (e.g., Hamilton STAR, Tecan EVO, automated plate readers)
- Detail liquid handling volumes, plate formats, incubation times
- Include quality control checkpoints

**HUMANOID ROBOT SYSTEMS** (General-purpose robots with human-like manipulation):
- What physical manipulations require human-like dexterity?
- Specify manual procedures that could be automated (microscopy setup, gel loading, equipment operation)
- Detail the required sensing and manipulation capabilities
- Include safety protocols and error handling

**AGI COMPUTATIONAL SYSTEMS** (Advanced AI for analysis and decision-making):
- What computational workflows and data analysis could be fully automated?
- Specify bioinformatics pipelines, statistical analyses, and interpretation steps
- Detail decision trees for experimental parameter optimization
- Include automated hypothesis generation and experimental design

For each system type, provide:
1. **Specific Steps**: Numbered list of automated procedures
2. **Required Capabilities**: Technical specifications needed
3. **Integration Points**: How the three systems coordinate
4. **Quality Control**: Automated validation and error detection
5. **Timeline**: Estimated automation time vs. manual time

**Integration Plan**: Describe how all three systems would work together to fully automate the research from start to publication-ready results.

Text:
{chunk_text}"""
    
    def _get_dataset_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate dataset extraction prompt."""
        return f"""Please analyze the following scientific text (part {chunk_idx + 1} of {total_chunks}) and identify ALL datasets that are mentioned, used, produced, or cited. For each dataset, extract as much information as possible using the following structured format:

**DATASET IDENTIFICATION AND CLASSIFICATION:**
- Dataset name/title (if provided)
- Dataset type (e.g., genomic sequences, RNA-seq data, protein structures, phylogenetic trees, experimental measurements, survey data, etc.)
- Data format (e.g., FASTQ, FASTA, BAM, VCF, CSV, JSON, etc.)
- Scale/size (number of samples, sequences, records, file sizes, etc.)

**DATA CREATION AND PROVENANCE:**
- How was the data generated/collected? (experimental methods, computational analysis, literature curation, etc.)
- Who created/owns the data? (authors, institutions, organizations)
- When was the data created? (dates, time periods)
- What instruments/platforms were used? (sequencing platforms, experimental equipment, software tools)
- Sample information (organism, tissue type, conditions, treatments, etc.)

**ACCESS AND AVAILABILITY:**
- Where is the data stored/available? (databases, repositories, URLs, DOIs)
- Access restrictions (public, restricted, upon request, etc.)
- Open access status: Determine if the dataset is OPEN ACCESS, RESTRICTED ACCESS, or UNKNOWN based on:
  * Public repositories (NCBI, EBI, etc.) = OPEN ACCESS
  * Upon request/authorization required = RESTRICTED ACCESS
  * Commercial databases requiring licenses = RESTRICTED ACCESS
  * Supplementary materials in open journals = OPEN ACCESS
  * No clear access method mentioned = UNKNOWN
- How to access the data? (direct download, API, database query, contact information)
- Accession numbers or identifiers (SRA, GEO, GenBank, PDB, etc.)
- File names or specific identifiers mentioned

**REFERENCES AND CITATIONS:**
- Citations to papers describing the dataset
- References to database entries or submissions
- URLs, DOIs, or other persistent identifiers
- Version information if applicable

**USAGE CONTEXT:**
- How was the dataset used in this study? (input data, comparison, validation, etc.)
- Was the dataset modified or processed? If so, how?
- What analysis was performed on the dataset?

**AGGREGATE SCALE ANALYSIS:**
- Provide an aggregate summary of the dataset scale in terms of:
  * Total computational/storage requirements (if mentioned)
  * Combined sample sizes across all components
  * Temporal scope (time periods covered)
  * Spatial scope (geographic regions, organisms, conditions covered)
  * Comparative scale (small/medium/large for the field)

**ADDITIONAL METADATA:**
- License or usage terms if mentioned
- Quality metrics or validation information
- Related datasets or dependencies
- Any limitations or caveats mentioned

Please organize your response as a numbered list of datasets. If no datasets are identified, respond with "No datasets identified in this text segment."

For each dataset, use this format:
**DATASET #:** [Brief descriptive name]
- **Type:** [Dataset type and format]
- **Scale:** [Size/scope information]
- **Aggregate Scale:** [Overall scope and scale assessment]
- **Creation:** [How and when created]
- **Owner:** [Who owns/created it]
- **Access:** [How to access, URLs, identifiers]
- **Open Access Status:** [OPEN ACCESS/RESTRICTED ACCESS/UNKNOWN]
- **Usage:** [How used in this study]
- **Additional Info:** [Any other relevant details]

Text:
{chunk_text}"""
    
    def _get_hypothesis_extraction_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate hypothesis extraction prompt for Wisteria-compatible JSON output."""
        return f"""Please analyze the following scientific text (part {chunk_idx + 1} of {total_chunks}) and extract ALL hypotheses mentioned, proposed, tested, or discussed. For each hypothesis, provide a comprehensive analysis following the scientific rigor framework.

**CRITICAL REQUIREMENTS:**
1. **Identify ALL hypotheses**: Look for explicit hypotheses, implicit assumptions, proposed mechanisms, theoretical predictions, and working models
2. **Extract complete information**: Title, detailed description, experimental validation plans, computational approaches
3. **Scientific rigor analysis**: Evaluate each hypothesis against the five hallmarks of strong scientific hypotheses
4. **Provide supporting references**: Include all relevant citations mentioned in the text
5. **Format as structured data**: Organize information for JSON conversion

**For each hypothesis identified, provide:**

**HYPOTHESIS IDENTIFICATION:**
- **Title**: Clear, descriptive title (if not explicitly stated, create one based on the main claim)
- **Description**: Comprehensive paragraph describing the hypothesis, its mechanism, key predictions, and scientific rationale
- **Source**: How the hypothesis appears in the text (explicitly stated, implicit assumption, proposed mechanism, etc.)

**EXPERIMENTAL VALIDATION:**
- **Proposed experiments**: Detailed experimental plan to test the hypothesis
- **Methods and techniques**: Specific experimental methods, measurements, controls, and expected outcomes
- **Timeline and feasibility**: Estimated timeline and practical considerations

**COMPUTATIONAL APPROACH:**
- **Theory and computation**: Computational modeling, simulations, theoretical frameworks, and analytical approaches
- **Tools and software**: Specific computational tools, databases, or platforms mentioned
- **Predictions and modeling**: What computational predictions support or could test the hypothesis

**SCIENTIFIC RIGOR ANALYSIS:**
Evaluate each hypothesis against these five hallmarks:

1. **Testability (Falsifiability)**: Can the hypothesis be experimentally tested? What would falsify it?
2. **Specificity and Clarity**: Are the variables, relationships, and predictions clearly defined?
3. **Grounded in Prior Knowledge**: How does it build on established scientific knowledge?
4. **Predictive Power & Novel Insight**: What new insights or novel predictions does it offer?
5. **Parsimony (Principle of Simplicity)**: Does it explain phenomena with minimal assumptions?

**SUPPORTING EVIDENCE:**
- **References**: All citations mentioned in relation to this hypothesis
- **Empirical support**: Existing evidence supporting or contradicting the hypothesis
- **Knowledge gaps**: What information is needed to better evaluate the hypothesis

**RESEARCH CONTEXT:**
- **Research goal**: The broader research question or problem this hypothesis addresses
- **Significance**: Why this hypothesis matters for the field
- **Applications**: Potential practical applications or implications

**OUTPUT FORMAT:**
For each hypothesis, use this exact structure:

**HYPOTHESIS #:** [Brief descriptive title]
- **Description:** [Detailed paragraph description]
- **Experimental Validation:** [Comprehensive experimental plan]
- **Theory and Computation:** [Computational approaches and modeling]
- **Testability:** [Analysis of falsifiability and testing approaches]
- **Specificity:** [Analysis of clarity and precision of claims]
- **Grounded Knowledge:** [Analysis of foundation in prior knowledge]
- **Predictive Power:** [Analysis of novel insights and predictions]
- **Parsimony:** [Analysis of simplicity and minimal assumptions]
- **References:** [List of supporting citations with annotations]
- **Research Context:** [Broader significance and applications]
- **Source in Text:** [How this hypothesis appears in the text]

If no clear hypotheses are identified, respond with "No hypotheses identified in this text segment."

**IMPORTANT NOTES:**
- Look beyond explicitly stated hypotheses - include proposed mechanisms, theoretical predictions, and working models
- Even preliminary or speculative ideas should be captured if they represent testable propositions
- Consider both the authors' hypotheses and any alternative hypotheses they discuss
- Include hypotheses that are mentioned from other research, not just those proposed by the current authors

Text:
{chunk_text}"""
    
    def _get_open_problems_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate open problems extraction prompt for Wisteria-compatible JSON output."""
        return f"""Please analyze the following scientific text (part {chunk_idx + 1} of {total_chunks}) and identify ALL open problems, unsolved questions, research gaps, future work directions, and challenges mentioned by the authors. Read very carefully for statements about what remains unknown, unexplored, or needs further investigation.

**CRITICAL REQUIREMENTS:**
1. **Identify ALL open problems**: Look for explicit statements about unsolved problems, research gaps, limitations, future work directions, and challenges
2. **Extract complete information**: Title, detailed description, significance, potential approaches
3. **Categorize by type**: Scientific challenges, technical limitations, methodological gaps, knowledge gaps
4. **Provide context**: Why this is an open problem and what solving it would accomplish
5. **Format as structured data**: Organize information for JSON conversion

**For each open problem identified, provide:**

**PROBLEM IDENTIFICATION:**
- **Title**: Clear, descriptive title of the open problem
- **Description**: Comprehensive paragraph describing the problem, what makes it challenging, and current limitations
- **Type**: Category (scientific challenge, technical limitation, methodological gap, knowledge gap, computational challenge, etc.)
- **Source**: How the problem appears in the text (explicitly stated, implied limitation, future work suggestion, etc.)

**PROBLEM CONTEXT:**
- **Current state**: What is currently known or possible
- **Gap or limitation**: What is missing, unknown, or not yet achievable
- **Significance**: Why solving this problem matters for the field
- **Impact**: What new capabilities or understanding would result from solving this

**POTENTIAL APPROACHES:**
- **Suggested methods**: Any approaches mentioned by authors or that could address the problem
- **Required resources**: What tools, data, or capabilities would be needed
- **Research directions**: Specific research questions or investigations that could help
- **Technical requirements**: Computational, experimental, or methodological needs

**RESEARCH SCOPE:**
- **Difficulty level**: Estimated complexity (straightforward, challenging, fundamental research needed)
- **Timeline**: Potential timeframe if mentioned (short-term, long-term, uncertain)
- **Dependencies**: What other advances or developments are needed first
- **Interdisciplinary aspects**: What fields or expertise areas are relevant

**SUPPORTING EVIDENCE:**
- **References**: Any citations related to this problem
- **Related work**: Existing attempts or partial solutions mentioned
- **Examples**: Specific examples or cases illustrating the problem

**OUTPUT FORMAT:**
For each open problem, use this exact structure:

**OPEN PROBLEM #:** [Brief descriptive title]
- **Description:** [Detailed paragraph description]
- **Type:** [Category of problem]
- **Current State:** [What is currently known/possible]
- **Gap:** [What is missing or limited]
- **Significance:** [Why this matters]
- **Impact:** [What solving it would accomplish]
- **Potential Approaches:** [Suggested methods or research directions]
- **Difficulty:** [Estimated complexity level]
- **Timeline:** [Potential timeframe if mentioned]
- **Dependencies:** [Prerequisites or related advances needed]
- **References:** [Related citations]
- **Source in Text:** [How this problem appears in the text]

If no open problems are identified, respond with "No open problems identified in this text segment."

**IMPORTANT NOTES:**
- Look for phrases like "remains unclear", "future work", "limitations", "challenges", "unknown", "needs investigation"
- Include both major research challenges and smaller technical problems
- Consider computational limitations, experimental challenges, and theoretical gaps
- Include problems that authors acknowledge but don't necessarily propose to solve
- Look for statements about what would be useful but isn't currently possible

Text:
{chunk_text}"""
    
    def _get_knowledge_graph_prompt(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Generate knowledge graph extraction prompt for RDF output with connectivity focus."""
        return f"""Please analyze the following scientific text (part {chunk_idx + 1} of {total_chunks}) and extract a knowledge graph representing the key entities, relationships, and concepts mentioned.

**CRITICAL REQUIREMENTS:**
1. **PRIORITIZE CONNECTIVITY**: Focus on relationships between entities rather than isolated descriptions. Every entity should be connected to at least one other entity through a relationship.
2. **Extract ALL entities**: Identify genes, proteins, organisms, methods, tools, databases, diseases, phenotypes, pathways, compounds, etc.
3. **Extract relationships**: Identify how entities relate to each other (e.g., "gene encodes protein", "protein interacts with protein", "method analyzes data")
4. **Use standard terminology**: When possible, use standard names and identifiers from relevant ontologies/databases
5. **Provide context**: Include the experimental or analytical context for relationships
6. **Format as structured triples**: Organize as Subject-Predicate-Object relationships
7. **AVOID ISOLATED NODES**: Do not list entities without relationships. If an entity is mentioned, find its connections to other entities.

**For each entity-relationship-entity triple, provide:**

**ENTITY IDENTIFICATION:**
- **Subject**: The source entity (gene, protein, organism, method, etc.)
- **Predicate**: The relationship type (encodes, interacts_with, analyzes, regulates, etc.)
- **Object**: The target entity
- **Context**: Experimental or analytical context for this relationship
- **Evidence**: Supporting evidence from the text
- **Confidence**: High/Medium/Low based on how explicitly stated in the text

**ENTITY TYPES TO EXTRACT:**
- **Biological entities**: genes, proteins, RNA, DNA sequences, organisms, cell types, tissues
- **Chemical entities**: compounds, drugs, metabolites, substrates
- **Phenotypic entities**: diseases, symptoms, traits, phenotypes
- **Methodological entities**: experimental methods, computational tools, databases, algorithms
- **Data entities**: datasets, measurements, results, statistics
- **Conceptual entities**: pathways, processes, functions, mechanisms

**RELATIONSHIP TYPES TO EXTRACT:**
- **Biological relationships**: encodes, regulates, interacts_with, binds_to, catalyzes
- **Methodological relationships**: uses, analyzes, measures, compares, validates
- **Causal relationships**: causes, leads_to, results_in, affects, influences
- **Structural relationships**: part_of, contains, located_in, composed_of
- **Functional relationships**: participates_in, involved_in, required_for, activates, inhibits

**OUTPUT FORMAT:**
For each relationship, use this exact structure:

**TRIPLE #:** [Sequential number]
- **Subject:** [Entity name] (Type: [entity type])
- **Predicate:** [Relationship type]
- **Object:** [Entity name] (Type: [entity type])
- **Context:** [Experimental/analytical context]
- **Evidence:** [Supporting text from the document]
- **Confidence:** [High/Medium/Low]
- **Source:** [Location in text where this relationship was identified]

**IMPORTANT NOTES:**
- **CONNECTIVITY IS PARAMOUNT**: Every entity must be connected to at least one other entity. Do not extract isolated entities.
- Focus on factual relationships explicitly stated or strongly implied in the text
- Include both direct experimental findings and methodological relationships
- For ambiguous relationships, mark confidence as Low but still include them
- Use standardized entity names when possible (e.g., official gene symbols, standard method names)
- Include quantitative relationships when measurements are provided
- Consider temporal relationships (before/after, during, following)
- **Look for indirect connections**: If entities seem unrelated, consider methodological or contextual connections
- **Prefer connected subgraphs**: Create clusters of related entities rather than scattered individual relationships

Text:
{chunk_text}"""
    
    def _consolidate_experimental_protocols(self, results: List[str]) -> List[str]:
        """Consolidate experimental protocol results and remove 'no protocol found' messages."""
        if self.config.mode != ExtractionMode.EXPERIMENTAL_PROTOCOL:
            return results
            
        # Combine all results into a single text
        combined_results = "\n\n".join(results)
        
        # Create consolidation prompt
        consolidation_prompt = f"""Please analyze the following experimental protocol extractions from different sections of a research paper and create a consolidated, coherent summary of all unique protocols mentioned.

Your tasks:
1. Remove any lines that say "No experimental protocols identified" or similar
2. Combine duplicate or very similar protocols into single entries
3. Use standard protocol names when possible
4. Organize protocols in a logical order (e.g., sample preparation, molecular techniques, analytical methods)
5. Return results as a clean, consolidated bulleted list

Here are the protocol extractions to consolidate:

{combined_results}

Please provide a consolidated summary of all unique experimental protocols mentioned across all sections:"""

        try:
            self.console.print("🔄 Consolidating experimental protocols...")
            consolidated_result = self._call_api(consolidation_prompt)
            return [consolidated_result]
        except Exception as e:
            self.console.print(f"⚠️  Failed to consolidate protocols: {e}")
            return results
    
    def _consolidate_knowledge_graph(self, results: List[str]) -> List[str]:
        """Consolidate knowledge graph results to improve connectivity and remove isolated nodes."""
        if self.config.mode != ExtractionMode.KNOWLEDGE_GRAPH:
            return results
            
        # Combine all results into a single text
        combined_results = "\n\n".join(results)
        
        # Create consolidation prompt focused on connectivity
        consolidation_prompt = f"""Please analyze the following knowledge graph extractions from different sections of a research paper and create a consolidated, well-connected knowledge graph.

**CRITICAL CONSOLIDATION TASKS:**
1. **ELIMINATE ISOLATED NODES**: Remove or connect any entities that appear without relationships
2. **MERGE DUPLICATE ENTITIES**: Combine similar or identical entities mentioned in different sections
3. **DISCOVER IMPLICIT CONNECTIONS**: Look for relationships between entities that appear in different sections but are related through the research context
4. **CREATE CONNECTED SUBGRAPHS**: Organize entities into connected clusters rather than scattered individual relationships
5. **STANDARDIZE ENTITY NAMES**: Use consistent naming for the same entities across sections
6. **MAINTAIN RELATIONSHIP QUALITY**: Preserve confidence scores and evidence, but improve connectivity

**CONNECTIVITY IMPROVEMENT STRATEGIES:**
- Look for methodological connections (e.g., "Method X analyzes Entity Y")
- Find contextual relationships (e.g., entities used in the same experimental context)
- Identify hierarchical relationships (e.g., "Gene X part_of Pathway Y")
- Connect through intermediate entities (e.g., "Gene X encodes Protein Y, Protein Y interacts_with Protein Z")
- Link through experimental conditions or datasets

**OUTPUT REQUIREMENTS:**
- Every entity should have at least one relationship
- Create a connected graph where you can navigate from any entity to others
- Maintain the structured triple format with confidence scores
- Remove any standalone entity descriptions
- Prioritize the most confident and well-supported relationships

Here are the knowledge graph extractions to consolidate and connect:

{combined_results}

Please provide a consolidated, well-connected knowledge graph where every entity is connected to at least one other entity:"""

        try:
            self.console.print("🔄 Consolidating knowledge graph for connectivity...")
            consolidated_result = self._call_api(consolidation_prompt)
            return [consolidated_result]
        except Exception as e:
            self.console.print(f"⚠️  Failed to consolidate knowledge graph: {e}")
            return results
    
    def _extract_cross_chunk_relationships(self, results: List[str]) -> str:
        """Extract relationships between entities mentioned in different chunks."""
        if len(results) < 2:
            return ""
            
        # Extract entity names from all chunks
        all_entities = set()
        chunk_entities = []
        
        for result in results:
            entities_in_chunk = set()
            lines = result.split('\n')
            for line in lines:
                if '**Subject:**' in line:
                    # Extract entity name from subject line
                    entity_match = re.search(r'\*\*Subject:\*\*\s*([^(]+)', line)
                    if entity_match:
                        entity = entity_match.group(1).strip()
                        all_entities.add(entity)
                        entities_in_chunk.add(entity)
                elif '**Object:**' in line:
                    # Extract entity name from object line
                    entity_match = re.search(r'\*\*Object:\*\*\s*([^(]+)', line)
                    if entity_match:
                        entity = entity_match.group(1).strip()
                        all_entities.add(entity)
                        entities_in_chunk.add(entity)
            chunk_entities.append(entities_in_chunk)
        
        # Find entities that appear in multiple chunks
        cross_chunk_entities = set()
        for i, chunk1_entities in enumerate(chunk_entities):
            for j, chunk2_entities in enumerate(chunk_entities[i+1:], i+1):
                common_entities = chunk1_entities.intersection(chunk2_entities)
                cross_chunk_entities.update(common_entities)
        
        if not cross_chunk_entities:
            return ""
            
        # Create prompt to find cross-chunk relationships
        entity_list = ", ".join(sorted(cross_chunk_entities))
        cross_chunk_prompt = f"""The following entities appear in multiple sections of the research paper: {entity_list}

Please identify additional relationships between these entities that span across different sections of the paper. Look for:
- Methodological connections (methods used to study multiple entities)
- Experimental context connections (entities used in the same experiments)
- Pathway or process connections (entities involved in the same biological processes)
- Causal or temporal connections (entities in experimental sequences)

Combined knowledge graph data:
{chr(10).join(results)}

Provide additional cross-section relationships in the same triple format:"""

        try:
            self.console.print("🔄 Extracting cross-chunk relationships...")
            cross_chunk_result = self._call_api(cross_chunk_prompt)
            return cross_chunk_result
        except Exception as e:
            self.console.print(f"⚠️  Failed to extract cross-chunk relationships: {e}")
            return ""
    
    def _validate_graph_connectivity(self, knowledge_graph_text: str) -> Dict[str, Any]:
        """Validate connectivity of the knowledge graph and provide metrics."""
        lines = knowledge_graph_text.split('\n')
        entities = set()
        relationships = []
        isolated_entities = set()
        
        current_triple = {}
        for line in lines:
            line = line.strip()
            if '**Subject:**' in line:
                subject_match = re.search(r'\*\*Subject:\*\*\s*([^(]+)', line)
                if subject_match:
                    current_triple['subject'] = subject_match.group(1).strip()
                    entities.add(current_triple['subject'])
            elif '**Object:**' in line:
                object_match = re.search(r'\*\*Object:\*\*\s*([^(]+)', line)
                if object_match:
                    current_triple['object'] = object_match.group(1).strip()
                    entities.add(current_triple['object'])
            elif '**Predicate:**' in line:
                predicate_match = re.search(r'\*\*Predicate:\*\*\s*(.+)', line)
                if predicate_match:
                    current_triple['predicate'] = predicate_match.group(1).strip()
                    
                    # If we have a complete triple, record it
                    if 'subject' in current_triple and 'object' in current_triple:
                        relationships.append(current_triple.copy())
                        current_triple = {}
        
        # Find connected entities
        connected_entities = set()
        for rel in relationships:
            if 'subject' in rel and 'object' in rel:
                connected_entities.add(rel['subject'])
                connected_entities.add(rel['object'])
        
        # Find isolated entities
        isolated_entities = entities - connected_entities
        
        # Calculate connectivity metrics
        total_entities = len(entities)
        connected_count = len(connected_entities)
        isolated_count = len(isolated_entities)
        connectivity_ratio = connected_count / total_entities if total_entities > 0 else 0
        
        return {
            'total_entities': total_entities,
            'connected_entities': connected_count,
            'isolated_entities': isolated_count,
            'connectivity_ratio': connectivity_ratio,
            'relationships_count': len(relationships),
            'isolated_entity_list': list(isolated_entities)
        }
    
    def _convert_to_rdf(self, knowledge_graph_text: str, source_file: str) -> str:
        """Convert knowledge graph text to RDF format."""
        self.console.print(f"🔄 Converting to RDF format for {Path(source_file).name}")
        
        # Basic RDF structure
        source_name = Path(source_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        rdf_header = f"""@prefix : <http://example.org/extract-o-matic/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix bio: <http://purl.obolibrary.org/obo/> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .

# Knowledge Graph extracted from {source_name}
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

:document_{source_name} rdf:type :Document ;
    dc:source "{source_file}" ;
    dc:created "{datetime.now().isoformat()}" ;
    :extractedBy "extract-o-matic" .

"""
        
        # Convert knowledge graph text to RDF triples
        # This is a simplified conversion - in practice, you'd want more sophisticated parsing
        rdf_triples = []
        lines = knowledge_graph_text.split('\n')
        entity_counter = 0
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Simple pattern matching for entities and relationships
                if '→' in line or '->' in line:
                    # Parse relationship patterns
                    separator = '→' if '→' in line else '->'
                    parts = line.split(separator)
                    if len(parts) == 2:
                        subject = parts[0].strip()
                        predicate_object = parts[1].strip()
                        
                        # Clean and create RDF-safe identifiers
                        subject_id = self._to_rdf_identifier(subject)
                        
                        # Create triple
                        rdf_triples.append(f":entity_{subject_id} rdfs:label \"{subject}\" .")
                        rdf_triples.append(f":entity_{subject_id} :relatedTo \"{predicate_object}\" .")
                        entity_counter += 1
                
                elif ':' in line:
                    # Parse attribute patterns
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        entity = parts[0].strip()
                        attribute = parts[1].strip()
                        
                        entity_id = self._to_rdf_identifier(entity)
                        rdf_triples.append(f":entity_{entity_id} rdfs:label \"{entity}\" .")
                        rdf_triples.append(f":entity_{entity_id} :hasProperty \"{attribute}\" .")
                        entity_counter += 1
        
        # Combine header and triples
        rdf_content = rdf_header + '\n'.join(rdf_triples) + '\n'
        return rdf_content

    def _to_rdf_identifier(self, text: str) -> str:
        """Convert text to RDF-safe identifier."""
        import re
        # Remove special characters and replace spaces with underscores
        identifier = re.sub(r'[^\w\s-]', '', text)
        identifier = re.sub(r'\s+', '_', identifier)
        identifier = identifier.lower()
        return identifier[:50]  # Limit length

    def _save_rdf_file(self, rdf_content: str, source_file: str) -> None:
        """Save RDF content to file."""
        source_name = Path(source_file).stem
        model_name = self.config.model_config.shortname
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine output file path
        if self.config.output_file.endswith('/'):
            output_file = Path(self.config.output_file) / f"knowledge_graph_{source_name}_{model_name}_{timestamp}.rdf"
        else:
            output_file = Path(f"knowledge_graph_{source_name}_{model_name}_{timestamp}.rdf")
        
        try:
            # Save RDF file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(rdf_content)
            
            self.console.print(f"✅ Saved knowledge graph RDF to {output_file}")
            
        except Exception as e:
            self.console.print(f"❌ Failed to save RDF file: {e}")
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to extract information."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes scientific texts for bioinformatics workflows."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.console.print(f"❌ API call failed: {e}")
            return f"ERROR: {e}"
    
    def _process_chunk_parallel(self, chunk_data: Tuple[int, str, int, int]) -> Tuple[int, str]:
        """Process a single chunk in parallel execution."""
        chunk_idx, chunk_text, chunk_num, total_chunks = chunk_data
        
        try:
            # Generate prompt for current mode
            prompt = self._get_prompt_for_mode(self.config.mode, chunk_text, chunk_num, total_chunks)
            
            # Make API call
            result = self._call_api(prompt)
            
            # Update progress counter (thread-safe)
            with self.progress_lock:
                self.processed_chunks += 1
            
            return (chunk_idx, result)
        except Exception as e:
            error_msg = f"ERROR processing chunk {chunk_idx}: {e}"
            with self.progress_lock:
                self.processed_chunks += 1
            return (chunk_idx, error_msg)
    
    def _classify_document(self, text: str) -> str:
        """Classify document as TOOL or PROBLEM."""
        try:
            # Use a sample of the document for classification (first 2000 characters)
            sample_text = text[:2000] if len(text) > 2000 else text
            
            prompt = f"""Your task is to analyze the provided text and output either 'TOOL' or 'PROBLEM' based on the criteria provided.

- If the text is about a bioinformatics tool or database (not a specific use of that on a problem), output 'TOOL'.
- If the text is about using one or more tools in a scientific workflow to solve a specific problem, output 'PROBLEM'.

Only output 'TOOL' or 'PROBLEM', nothing else.

Here is the text:

{sample_text}"""
            
            response = self.client.chat.completions.create(
                model=self.config.model_config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies scientific texts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0,
            )
            
            result = response.choices[0].message.content.strip().upper()
            
            # Extract TOOL or PROBLEM from the response
            if 'TOOL' in result:
                return 'TOOL'
            elif 'PROBLEM' in result:
                return 'PROBLEM'
            else:
                return 'UNKNOWN'
                
        except Exception as e:
            self.console.print(f"❌ Classification failed: {e}")
            return 'ERROR'
    
    def extract(self) -> List[str]:
        """Main extraction method."""
        if self.config.mode == ExtractionMode.HYPOTHESIS_EXTRACTION:
            return self._extract_hypotheses()
        elif self.config.mode == ExtractionMode.OPEN_PROBLEMS:
            return self._extract_open_problems()
        elif self.config.batch_mode:
            return self._extract_batch()
        else:
            return self._extract_single()
    
    def _extract_hypotheses(self) -> List[str]:
        """Extract hypotheses and generate Wisteria-compatible JSON files."""
        self.console.print("🔬 Starting hypothesis extraction for Wisteria compatibility")
        
        if self.config.batch_mode:
            return self._extract_hypotheses_batch()
        else:
            return self._extract_hypotheses_single()
    
    def _extract_hypotheses_single(self) -> List[str]:
        """Extract hypotheses from a single file and generate JSON."""
        self.console.print(f"📄 Processing {self.config.input_file}")
        
        # Read input file
        text = self._read_input_file()
        
        # Check if document classification is required
        if self.config.require_problem or self.config.require_tool:
            classification = self._classify_document(text)
            if self.config.require_problem and classification == 'TOOL':
                self.console.print("❌ Document classified as TOOL - skipping extraction")
                return ["Document classified as TOOL - extraction skipped"]
            elif self.config.require_tool and classification == 'PROBLEM':
                self.console.print("❌ Document classified as PROBLEM - skipping extraction")
                return ["Document classified as PROBLEM - extraction skipped"]
        
        # Extract hypotheses from text
        hypotheses_text = self._process_text_for_hypotheses(text, self.config.input_file)
        
        # Convert to Wisteria JSON format
        json_data = self._convert_to_wisteria_json(hypotheses_text, self.config.input_file)
        
        # Save JSON file
        self._save_wisteria_json(json_data, self.config.input_file)
        
        return [hypotheses_text]
    
    def _extract_hypotheses_batch(self) -> List[str]:
        """Extract hypotheses from multiple files and generate JSON files."""
        self.console.print(f"📁 Processing {len(self.config.input_files)} files")
        
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            files_task = progress.add_task("Processing files", total=len(self.config.input_files))
            
            for file_path in self.config.input_files:
                progress.update(files_task, description=f"Processing {Path(file_path).name}")
                
                # Read file
                text = self._read_single_file(file_path)
                
                if not text:
                    self.console.print(f"⚠️  Skipping empty file: {file_path}")
                    progress.advance(files_task)
                    continue
                
                # Check if document classification is required
                if self.config.require_problem or self.config.require_tool:
                    classification = self._classify_document(text)
                    if self.config.require_problem and classification == 'TOOL':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as TOOL")
                        progress.advance(files_task)
                        continue
                    elif self.config.require_tool and classification == 'PROBLEM':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as PROBLEM")
                        progress.advance(files_task)
                        continue
                
                # Extract hypotheses from text
                hypotheses_text = self._process_text_for_hypotheses(text, file_path, show_progress=False)
                
                # Convert to Wisteria JSON format
                json_data = self._convert_to_wisteria_json(hypotheses_text, file_path)
                
                # Save JSON file
                self._save_wisteria_json(json_data, file_path)
                
                # Add to results
                all_results.append(f"\n{'='*60}\nFILE: {Path(file_path).name}\n{'='*60}\n")
                all_results.append(hypotheses_text)
                
                progress.advance(files_task)
        
        self.results = all_results
        return all_results
    
    def _process_text_for_hypotheses(self, text: str, file_path: str, show_progress: bool = True) -> str:
        """Process text to extract hypotheses using the AI model."""
        # Chunk text
        chunks = self._chunk_text(text)
        
        # Validate chunking coverage
        self._validate_chunking(text, chunks)
        
        self.console.print(f"📄 Processing {len(chunks)} chunks from {Path(file_path).name}")
        if len(chunks) > 1:
            self.console.print(f"   Document length: {len(text)} characters, chunked with overlap for complete coverage")
        
        # Process chunks
        chunk_results = []
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Extracting hypotheses", total=len(chunks))
                
                for idx, chunk in enumerate(chunks):
                    progress.update(task, description=f"Processing chunk {idx + 1}/{len(chunks)}")
                    
                    # Generate prompt for hypothesis extraction
                    prompt = self._get_hypothesis_extraction_prompt(chunk, idx, len(chunks))
                    
                    # Make API call
                    result = self._call_api(prompt)
                    
                    # Store result
                    chunk_results.append(result)
                    
                    progress.advance(task)
        else:
            # Process without progress bar (for batch mode)
            for idx, chunk in enumerate(chunks):
                # Generate prompt for hypothesis extraction
                prompt = self._get_hypothesis_extraction_prompt(chunk, idx, len(chunks))
                
                # Make API call
                result = self._call_api(prompt)
                
                # Store result
                chunk_results.append(result)
        
        # Combine all chunk results
        return "\n\n".join(chunk_results)
    
    def _convert_to_wisteria_json(self, hypotheses_text: str, file_path: str) -> Dict[str, Any]:
        """Convert extracted hypotheses text to Wisteria-compatible JSON format."""
        # Parse hypotheses from the extracted text
        hypotheses = self._parse_hypotheses_from_text(hypotheses_text)
        
        # Create metadata
        timestamp = datetime.now().isoformat()
        metadata = {
            "session_type": "extract_o_matic",
            "research_goal_source": "extracted_from_paper",
            "research_goal": f"Hypotheses extracted from {Path(file_path).name}",
            "model": self.config.model_config.shortname,
            "model_name": self.config.model_config.openai_model,
            "num_unique_hypotheses": len(hypotheses),
            "total_hypothesis_versions": len(hypotheses),
            "timestamp": timestamp,
            "session_time_seconds": 0,
            "hypothesis_types": {
                "original": len(hypotheses),
                "improvements": 0,
                "new_alternatives": 0
            },
            "source_document": Path(file_path).name,
            "extraction_method": "extract_o_matic_hypothesis_extraction"
        }
        
        return {
            "metadata": metadata,
            "hypotheses": hypotheses
        }
    
    def _parse_hypotheses_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse the AI-generated hypotheses text into structured data."""
        hypotheses = []
        
        # Split text into sections for each hypothesis
        # Look for pattern "**HYPOTHESIS #:** [title]"
        hypothesis_sections = re.split(r'\*\*HYPOTHESIS \d+:\*\*', text)
        
        # Skip the first section (usually empty or intro text)
        if len(hypothesis_sections) > 1:
            hypothesis_sections = hypothesis_sections[1:]
        
        for idx, section in enumerate(hypothesis_sections):
            if not section.strip():
                continue
                
            hypothesis = self._parse_single_hypothesis(section, idx + 1)
            if hypothesis:
                hypotheses.append(hypothesis)
        
        # If no structured hypotheses found, try to extract from unstructured text
        if not hypotheses:
            hypotheses = self._extract_hypotheses_from_unstructured_text(text)
        
        return hypotheses
    
    def _parse_single_hypothesis(self, section: str, hypothesis_number: int) -> Optional[Dict[str, Any]]:
        """Parse a single hypothesis section into structured data."""
        try:
            lines = section.strip().split('\n')
            
            # Extract title (first non-empty line)
            title = ""
            for line in lines:
                if line.strip():
                    title = line.strip()
                    break
            
            # Extract structured fields
            description = self._extract_field_from_section(section, ["Description:", "**Description:**"])
            experimental_validation = self._extract_field_from_section(section, ["Experimental Validation:", "**Experimental Validation:**"])
            theory_computation = self._extract_field_from_section(section, ["Theory and Computation:", "**Theory and Computation:**"])
            
            # Extract hallmarks
            testability = self._extract_field_from_section(section, ["Testability:", "**Testability:**"])
            specificity = self._extract_field_from_section(section, ["Specificity:", "**Specificity:**"])
            grounded_knowledge = self._extract_field_from_section(section, ["Grounded Knowledge:", "**Grounded Knowledge:**"])
            predictive_power = self._extract_field_from_section(section, ["Predictive Power:", "**Predictive Power:**"])
            parsimony = self._extract_field_from_section(section, ["Parsimony:", "**Parsimony:**"])
            
            # Extract references
            references_text = self._extract_field_from_section(section, ["References:", "**References:**"])
            references = self._parse_references(references_text)
            
            # Extract research context
            research_context = self._extract_field_from_section(section, ["Research Context:", "**Research Context:**"])
            
            # Create hypothesis object
            timestamp = datetime.now().isoformat()
            
            hypothesis = {
                "title": title or f"Extracted Hypothesis {hypothesis_number}",
                "description": description or "No description extracted",
                "experimental_validation": experimental_validation or "No experimental validation plan extracted",
                "theory_and_computation": theory_computation or "No computational approach extracted",
                "hallmarks": {
                    "testability": testability or "Analysis not available",
                    "specificity": specificity or "Analysis not available",
                    "grounded_knowledge": grounded_knowledge or "Analysis not available",
                    "predictive_power": predictive_power or "Analysis not available",
                    "parsimony": parsimony or "Analysis not available"
                },
                "references": references,
                "hypothesis_number": hypothesis_number,
                "version": "1.0",
                "type": "extracted",
                "generation_timestamp": timestamp,
                "research_context": research_context or "No context extracted",
                "extraction_source": "extract_o_matic"
            }
            
            return hypothesis
            
        except Exception as e:
            self.console.print(f"⚠️  Error parsing hypothesis {hypothesis_number}: {e}")
            return None
    
    def _extract_field_from_section(self, section: str, field_markers: List[str]) -> str:
        """Extract content for a specific field from a hypothesis section."""
        for marker in field_markers:
            if marker in section:
                # Find the start of the field
                start_idx = section.find(marker)
                if start_idx == -1:
                    continue
                    
                # Find the content after the marker
                content_start = start_idx + len(marker)
                content = section[content_start:].strip()
                
                # Find the end of this field (next field marker or end of section)
                next_field_markers = [
                    "- **Description:**", "- **Experimental Validation:**", "- **Theory and Computation:**",
                    "- **Testability:**", "- **Specificity:**", "- **Grounded Knowledge:**",
                    "- **Predictive Power:**", "- **Parsimony:**", "- **References:**",
                    "- **Research Context:**", "- **Source in Text:**"
                ]
                
                end_idx = len(content)
                for next_marker in next_field_markers:
                    marker_pos = content.find(next_marker)
                    if marker_pos != -1 and marker_pos < end_idx:
                        end_idx = marker_pos
                
                field_content = content[:end_idx].strip()
                
                # Clean up the content (remove bullet points, extra whitespace)
                field_content = re.sub(r'^-\s*', '', field_content)
                field_content = re.sub(r'\n\s*\n', '\n', field_content)
                
                return field_content
        
        return ""
    
    def _parse_references(self, references_text: str) -> List[Dict[str, str]]:
        """Parse references from text into structured format."""
        references = []
        
        if not references_text:
            return references
        
        # Try to split by common citation patterns
        citation_lines = references_text.split('\n')
        
        for line in citation_lines:
            line = line.strip()
            if not line or line.startswith('-') and len(line) < 10:
                continue
                
            # Clean up line
            line = re.sub(r'^-\s*', '', line)
            
            # Simple heuristic: if line contains author name and year, treat as citation
            if re.search(r'\(\d{4}\)', line) or re.search(r'\d{4}', line):
                references.append({
                    "citation": line,
                    "annotation": "Reference extracted from source document"
                })
        
        return references
    
    def _extract_hypotheses_from_unstructured_text(self, text: str) -> List[Dict[str, Any]]:
        """Fallback method to extract basic hypothesis information from unstructured text."""
        hypotheses = []
        
        # Look for sentences that might contain hypotheses
        hypothesis_indicators = [
            "hypothesis", "hypothesize", "propose", "suggest", "predict", 
            "expect", "assume", "postulate", "theory", "model"
        ]
        
        sentences = re.split(r'[.!?]+', text)
        hypothesis_number = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence contains hypothesis indicators
            lower_sentence = sentence.lower()
            if any(indicator in lower_sentence for indicator in hypothesis_indicators):
                timestamp = datetime.now().isoformat()
                
                hypothesis = {
                    "title": f"Extracted Hypothesis {hypothesis_number}",
                    "description": sentence,
                    "experimental_validation": "No experimental validation plan extracted",
                    "theory_and_computation": "No computational approach extracted",
                    "hallmarks": {
                        "testability": "Requires further analysis",
                        "specificity": "Requires further analysis",
                        "grounded_knowledge": "Requires further analysis",
                        "predictive_power": "Requires further analysis",
                        "parsimony": "Requires further analysis"
                    },
                    "references": [],
                    "hypothesis_number": hypothesis_number,
                    "version": "1.0",
                    "type": "extracted_unstructured",
                    "generation_timestamp": timestamp,
                    "research_context": "Extracted from unstructured text",
                    "extraction_source": "extract_o_matic_fallback"
                }
                
                hypotheses.append(hypothesis)
                hypothesis_number += 1
                
                # Limit to reasonable number of hypotheses
                if hypothesis_number > 10:
                    break
        
        return hypotheses
    
    def _save_wisteria_json(self, json_data: Dict[str, Any], source_file: str):
        """Save hypothesis data in Wisteria-compatible JSON format."""
        try:
            # Generate output filename
            source_name = Path(source_file).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model_config.shortname
            
            if self.config.batch_mode:
                # For batch mode, create individual files in output directory
                output_dir = Path(self.config.output_file)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"hypotheses_{source_name}_{model_name}_{timestamp}.json"
            else:
                # For single file mode, use specified output or generate name
                if self.config.output_file.endswith('.json'):
                    output_file = Path(self.config.output_file)
                else:
                    output_file = Path(f"hypotheses_{source_name}_{model_name}_{timestamp}.json")
            
            # Save JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            num_hypotheses = len(json_data.get('hypotheses', []))
            self.console.print(f"✅ Saved {num_hypotheses} hypotheses to {output_file}")
            
        except Exception as e:
            self.console.print(f"❌ Failed to save Wisteria JSON: {e}")
    
    def _extract_open_problems(self) -> List[str]:
        """Extract open problems and generate Wisteria-compatible JSON files."""
        self.console.print("🔍 Starting open problems extraction for Wisteria compatibility")
        
        if self.config.batch_mode:
            return self._extract_open_problems_batch()
        else:
            return self._extract_open_problems_single()
    
    def _extract_open_problems_single(self) -> List[str]:
        """Extract open problems from a single file and generate JSON."""
        self.console.print(f"📄 Processing {self.config.input_file}")
        
        # Read input file
        text = self._read_input_file()
        
        # Check if document classification is required
        if self.config.require_problem or self.config.require_tool:
            classification = self._classify_document(text)
            if self.config.require_problem and classification == 'TOOL':
                self.console.print("❌ Document classified as TOOL - skipping extraction")
                return ["Document classified as TOOL - extraction skipped"]
            elif self.config.require_tool and classification == 'PROBLEM':
                self.console.print("❌ Document classified as PROBLEM - skipping extraction")
                return ["Document classified as PROBLEM - extraction skipped"]
        
        # Extract open problems from text
        open_problems_text = self._process_text_for_open_problems(text, self.config.input_file)
        
        # Convert to Wisteria JSON format
        json_data = self._convert_open_problems_to_wisteria_json(open_problems_text, self.config.input_file)
        
        # Save JSON file
        self._save_open_problems_wisteria_json(json_data, self.config.input_file)
        
        return [open_problems_text]
    
    def _extract_open_problems_batch(self) -> List[str]:
        """Extract open problems from multiple files and generate JSON files."""
        self.console.print(f"📁 Processing {len(self.config.input_files)} files")
        
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            files_task = progress.add_task("Processing files", total=len(self.config.input_files))
            
            for file_path in self.config.input_files:
                progress.update(files_task, description=f"Processing {Path(file_path).name}")
                
                # Read file
                text = self._read_single_file(file_path)
                
                if not text:
                    self.console.print(f"⚠️  Skipping empty file: {file_path}")
                    progress.advance(files_task)
                    continue
                
                # Check if document classification is required
                if self.config.require_problem or self.config.require_tool:
                    classification = self._classify_document(text)
                    if self.config.require_problem and classification == 'TOOL':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as TOOL")
                        progress.advance(files_task)
                        continue
                    elif self.config.require_tool and classification == 'PROBLEM':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as PROBLEM")
                        progress.advance(files_task)
                        continue
                
                # Extract open problems from text
                open_problems_text = self._process_text_for_open_problems(text, file_path, show_progress=False)
                
                # Convert to Wisteria JSON format
                json_data = self._convert_open_problems_to_wisteria_json(open_problems_text, file_path)
                
                # Save JSON file
                self._save_open_problems_wisteria_json(json_data, file_path)
                
                # Add to results
                all_results.append(f"\n{'='*60}\nFILE: {Path(file_path).name}\n{'='*60}\n")
                all_results.append(open_problems_text)
                
                progress.advance(files_task)
        
        self.results = all_results
        return all_results
    
    def _process_text_for_open_problems(self, text: str, file_path: str, show_progress: bool = True) -> str:
        """Process text to extract open problems using the AI model."""
        # Chunk text
        chunks = self._chunk_text(text)
        
        # Validate chunking coverage
        self._validate_chunking(text, chunks)
        
        self.console.print(f"📄 Processing {len(chunks)} chunks from {Path(file_path).name}")
        if len(chunks) > 1:
            self.console.print(f"   Document length: {len(text)} characters, chunked with overlap for complete coverage")
        
        # Process chunks
        chunk_results = []
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Extracting open problems", total=len(chunks))
                
                for idx, chunk in enumerate(chunks):
                    progress.update(task, description=f"Processing chunk {idx + 1}/{len(chunks)}")
                    
                    # Generate prompt for open problems extraction
                    prompt = self._get_open_problems_prompt(chunk, idx, len(chunks))
                    
                    # Make API call
                    result = self._call_api(prompt)
                    
                    # Store result
                    chunk_results.append(result)
                    
                    progress.advance(task)
        else:
            # Process without progress bar (for batch mode)
            for idx, chunk in enumerate(chunks):
                # Generate prompt for open problems extraction
                prompt = self._get_open_problems_prompt(chunk, idx, len(chunks))
                
                # Make API call
                result = self._call_api(prompt)
                
                # Store result
                chunk_results.append(result)
        
        # Combine all chunk results
        return "\n\n".join(chunk_results)
    
    def _convert_open_problems_to_wisteria_json(self, open_problems_text: str, file_path: str) -> Dict[str, Any]:
        """Convert extracted open problems text to Wisteria-compatible JSON format."""
        # Parse open problems from the extracted text
        open_problems = self._parse_open_problems_from_text(open_problems_text)
        
        # Create metadata
        timestamp = datetime.now().isoformat()
        metadata = {
            "session_type": "extract_o_matic",
            "research_goal_source": "extracted_from_paper",
            "research_goal": f"Open problems extracted from {Path(file_path).name}",
            "model": self.config.model_config.shortname,
            "model_name": self.config.model_config.openai_model,
            "num_unique_problems": len(open_problems),
            "total_problem_versions": len(open_problems),
            "timestamp": timestamp,
            "session_time_seconds": 0,
            "problem_types": {
                "extracted": len(open_problems)
            },
            "source_document": Path(file_path).name,
            "extraction_method": "extract_o_matic_open_problems_extraction"
        }
        
        return {
            "metadata": metadata,
            "open_problems": open_problems
        }
    
    def _parse_open_problems_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse the AI-generated open problems text into structured data."""
        open_problems = []
        
        # Split text into sections for each open problem
        # Look for pattern "**OPEN PROBLEM #:** [title]"
        problem_sections = re.split(r'\*\*OPEN PROBLEM \d+:\*\*', text)
        
        # Skip the first section (usually empty or intro text)
        if len(problem_sections) > 1:
            problem_sections = problem_sections[1:]
        
        for idx, section in enumerate(problem_sections):
            if not section.strip():
                continue
                
            open_problem = self._parse_single_open_problem(section, idx + 1)
            if open_problem:
                open_problems.append(open_problem)
        
        # If no structured open problems found, try to extract from unstructured text
        if not open_problems:
            open_problems = self._extract_open_problems_from_unstructured_text(text)
        
        return open_problems
    
    def _parse_single_open_problem(self, section: str, problem_number: int) -> Optional[Dict[str, Any]]:
        """Parse a single open problem section into structured data."""
        try:
            lines = section.strip().split('\n')
            
            # Extract title (first non-empty line)
            title = ""
            for line in lines:
                if line.strip():
                    title = line.strip()
                    break
            
            # Extract structured fields (removing markdown formatting)
            description = self._extract_clean_field_from_section(section, ["Description:", "**Description:**"])
            problem_type = self._extract_clean_field_from_section(section, ["Type:", "**Type:**"])
            current_state = self._extract_clean_field_from_section(section, ["Current State:", "**Current State:**"])
            gap = self._extract_clean_field_from_section(section, ["Gap:", "**Gap:**"])
            significance = self._extract_clean_field_from_section(section, ["Significance:", "**Significance:**"])
            impact = self._extract_clean_field_from_section(section, ["Impact:", "**Impact:**"])
            potential_approaches = self._extract_clean_field_from_section(section, ["Potential Approaches:", "**Potential Approaches:**"])
            difficulty = self._extract_clean_field_from_section(section, ["Difficulty:", "**Difficulty:**"])
            timeline = self._extract_clean_field_from_section(section, ["Timeline:", "**Timeline:**"])
            dependencies = self._extract_clean_field_from_section(section, ["Dependencies:", "**Dependencies:**"])
            
            # Extract references
            references_text = self._extract_clean_field_from_section(section, ["References:", "**References:**"])
            references = self._parse_references(references_text)
            
            # Extract source
            source_in_text = self._extract_clean_field_from_section(section, ["Source in Text:", "**Source in Text:**"])
            
            # Create open problem object
            timestamp = datetime.now().isoformat()
            
            open_problem = {
                "title": title or f"Extracted Open Problem {problem_number}",
                "description": description or "No description extracted",
                "type": problem_type or "Unknown",
                "current_state": current_state or "No current state information",
                "gap": gap or "No gap analysis available",
                "significance": significance or "Significance not specified",
                "impact": impact or "Impact not specified",
                "potential_approaches": potential_approaches or "No approaches suggested",
                "difficulty": difficulty or "Difficulty not assessed",
                "timeline": timeline or "Timeline not specified",
                "dependencies": dependencies or "No dependencies identified",
                "references": references,
                "problem_number": problem_number,
                "version": "1.0",
                "type_category": "extracted",
                "generation_timestamp": timestamp,
                "source_in_text": source_in_text or "Not specified",
                "extraction_source": "extract_o_matic"
            }
            
            return open_problem
            
        except Exception as e:
            self.console.print(f"⚠️  Error parsing open problem {problem_number}: {e}")
            return None
    
    def _extract_clean_field_from_section(self, section: str, field_markers: List[str]) -> str:
        """Extract content for a specific field from an open problem section, removing markdown formatting."""
        field_content = self._extract_field_from_section(section, field_markers)
        
        if field_content:
            # Remove markdown formatting
            field_content = re.sub(r'\*\*(.*?)\*\*', r'\1', field_content)  # Remove bold
            field_content = re.sub(r'\*(.*?)\*', r'\1', field_content)      # Remove italics
            field_content = re.sub(r'`(.*?)`', r'\1', field_content)        # Remove code blocks
            field_content = re.sub(r'#+\s*', '', field_content)             # Remove headers
            field_content = re.sub(r'^\s*-\s*', '', field_content, flags=re.MULTILINE)  # Remove bullet points
            field_content = field_content.strip()
        
        return field_content
    
    def _extract_open_problems_from_unstructured_text(self, text: str) -> List[Dict[str, Any]]:
        """Fallback method to extract basic open problem information from unstructured text."""
        open_problems = []
        
        # Look for sentences that might contain open problems
        problem_indicators = [
            "remains unclear", "future work", "limitation", "challenge", "unknown", 
            "needs investigation", "open question", "unsolved", "further research",
            "not yet understood", "requires", "would be useful", "would benefit"
        ]
        
        sentences = re.split(r'[.!?]+', text)
        problem_number = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence contains problem indicators
            lower_sentence = sentence.lower()
            if any(indicator in lower_sentence for indicator in problem_indicators):
                timestamp = datetime.now().isoformat()
                
                open_problem = {
                    "title": f"Extracted Open Problem {problem_number}",
                    "description": sentence,
                    "type": "Unstructured extraction",
                    "current_state": "Not specified",
                    "gap": "Requires further analysis",
                    "significance": "Requires further analysis",
                    "impact": "Requires further analysis",
                    "potential_approaches": "No approaches suggested",
                    "difficulty": "Unknown",
                    "timeline": "Unknown",
                    "dependencies": "Unknown",
                    "references": [],
                    "problem_number": problem_number,
                    "version": "1.0",
                    "type_category": "extracted_unstructured",
                    "generation_timestamp": timestamp,
                    "source_in_text": "Extracted from unstructured text",
                    "extraction_source": "extract_o_matic_fallback"
                }
                
                open_problems.append(open_problem)
                problem_number += 1
                
                # Limit to reasonable number of problems
                if problem_number > 15:
                    break
        
        return open_problems
    
    def _save_open_problems_wisteria_json(self, json_data: Dict[str, Any], source_file: str):
        """Save open problems data in Wisteria-compatible JSON format."""
        try:
            # Generate output filename
            source_name = Path(source_file).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.config.model_config.shortname
            
            if self.config.batch_mode:
                # For batch mode, create individual files in output directory
                output_dir = Path(self.config.output_file)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"open_problems_{source_name}_{model_name}_{timestamp}.json"
            else:
                # For single file mode, use specified output or generate name
                if self.config.output_file.endswith('.json'):
                    output_file = Path(self.config.output_file)
                else:
                    output_file = Path(f"open_problems_{source_name}_{model_name}_{timestamp}.json")
            
            # Save JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            num_problems = len(json_data.get('open_problems', []))
            self.console.print(f"✅ Saved {num_problems} open problems to {output_file}")
            
        except Exception as e:
            self.console.print(f"❌ Failed to save open problems Wisteria JSON: {e}")
    
    def _extract_single(self) -> List[str]:
        """Extract from a single file."""
        self.console.print(f"🚀 Starting extraction in {self.config.mode.value} mode")
        
        # Read input file
        text = self._read_input_file()
        
        # Check if document classification is required
        if self.config.require_problem or self.config.require_tool:
            self.console.print("🔍 Classifying document first...")
            classification = self._classify_document(text)
            
            if self.config.require_problem:
                if classification == 'TOOL':
                    self.console.print("❌ Document classified as TOOL - skipping extraction (use --require-problem flag only for PROBLEM papers)")
                    return ["Document classified as TOOL - extraction skipped"]
                elif classification == 'PROBLEM':
                    self.console.print("✅ Document classified as PROBLEM - proceeding with extraction")
                elif classification == 'ERROR':
                    self.console.print("⚠️  Classification failed - proceeding with extraction anyway")
                else:
                    self.console.print(f"⚠️  Unknown classification result '{classification}' - proceeding with extraction anyway")
                    
            elif self.config.require_tool:
                if classification == 'PROBLEM':
                    self.console.print("❌ Document classified as PROBLEM - skipping extraction (use --require-tool flag only for TOOL papers)")
                    return ["Document classified as PROBLEM - extraction skipped"]
                elif classification == 'TOOL':
                    self.console.print("✅ Document classified as TOOL - proceeding with extraction")
                elif classification == 'ERROR':
                    self.console.print("⚠️  Classification failed - proceeding with extraction anyway")
                else:
                    self.console.print(f"⚠️  Unknown classification result '{classification}' - proceeding with extraction anyway")
        
        # Chunk text
        chunks = self._chunk_text(text)
        self.total_chunks = len(chunks)
        
        # Validate chunking coverage
        self._validate_chunking(text, chunks)
        
        self.console.print(f"📄 Processing {self.total_chunks} chunks from {self.config.input_file}")
        if len(chunks) > 1:
            self.console.print(f"   Document length: {len(text)} characters, chunked with overlap for complete coverage")
        
        if self.config.worker_count > 1:
            # Parallel processing
            self.console.print(f"🔄 Using {self.config.worker_count} workers for parallel processing")
            results = self._process_chunks_parallel(chunks)
        else:
            # Sequential processing
            results = self._process_chunks_sequential(chunks)
        
        # Consolidate experimental protocols if in experimental_protocol mode
        if self.config.mode == ExtractionMode.EXPERIMENTAL_PROTOCOL:
            results = self._consolidate_experimental_protocols(results)
        
        # Process knowledge graph if in knowledge_graph mode
        if self.config.mode == ExtractionMode.KNOWLEDGE_GRAPH:
            # Step 1: Extract cross-chunk relationships
            cross_chunk_relationships = self._extract_cross_chunk_relationships(results)
            if cross_chunk_relationships:
                results.append(cross_chunk_relationships)
            
            # Step 2: Consolidate for connectivity
            results = self._consolidate_knowledge_graph(results)
            
            # Step 3: Combine all results
            combined_result = "\n\n".join(results)
            
            # Step 4: Validate connectivity and show metrics
            connectivity_metrics = self._validate_graph_connectivity(combined_result)
            self.console.print(f"📊 Knowledge Graph Connectivity Metrics:")
            self.console.print(f"   Total entities: {connectivity_metrics['total_entities']}")
            self.console.print(f"   Connected entities: {connectivity_metrics['connected_entities']}")
            self.console.print(f"   Isolated entities: {connectivity_metrics['isolated_entities']}")
            self.console.print(f"   Connectivity ratio: {connectivity_metrics['connectivity_ratio']:.2%}")
            self.console.print(f"   Total relationships: {connectivity_metrics['relationships_count']}")
            
            if connectivity_metrics['isolated_entities'] > 0:
                self.console.print(f"⚠️  Isolated entities found: {', '.join(connectivity_metrics['isolated_entity_list'][:5])}")
                if len(connectivity_metrics['isolated_entity_list']) > 5:
                    self.console.print(f"   ... and {len(connectivity_metrics['isolated_entity_list']) - 5} more")
            
            # Step 5: Convert to RDF format
            rdf_content = self._convert_to_rdf(combined_result, self.config.input_file)
            
            # Step 6: Save RDF file
            self._save_rdf_file(rdf_content, self.config.input_file)
            
            # Return the knowledge graph text
            results = [combined_result]
        
        return results
    
    def _process_chunks_sequential(self, chunks: List[str]) -> List[str]:
        """Process chunks sequentially (original behavior)."""
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=self.total_chunks)
            
            for idx, chunk in enumerate(chunks):
                progress.update(task, description=f"Processing chunk {idx + 1}/{self.total_chunks}")
                
                # Generate prompt for current mode
                prompt = self._get_prompt_for_mode(self.config.mode, chunk, idx, self.total_chunks)
                
                # Make API call
                result = self._call_api(prompt)
                
                # Store result
                self.results.append(result)
                self.processed_chunks += 1
                
                progress.advance(task)
        
        return self.results
    
    def _process_chunks_parallel(self, chunks: List[str]) -> List[str]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        # Prepare chunk data for parallel processing
        chunk_data = [(idx, chunk, idx, self.total_chunks) for idx, chunk in enumerate(chunks)]
        
        # Initialize results list with placeholders
        results = [None] * len(chunks)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=self.total_chunks)
            
            with ThreadPoolExecutor(max_workers=self.config.worker_count) as executor:
                # Submit all tasks
                future_to_chunk = {executor.submit(self._process_chunk_parallel, data): data for data in chunk_data}
                
                # Process completed tasks
                for future in as_completed(future_to_chunk):
                    chunk_idx, result = future.result()
                    results[chunk_idx] = result
                    progress.advance(task)
        
        # Store results in the main results list
        self.results = results
        return self.results
    
    def _extract_batch(self) -> List[str]:
        """Extract from multiple files in batch mode."""
        self.console.print(f"🚀 Starting batch extraction in {self.config.mode.value} mode")
        self.console.print(f"📁 Processing {len(self.config.input_files)} files")
        
        if self.config.worker_count > 1:
            # Parallel file processing
            self.console.print(f"🔄 Using {self.config.worker_count} workers for parallel processing")
            return self._process_files_parallel()
        else:
            # Sequential file processing
            return self._process_files_sequential()
    
    def _process_files_sequential(self) -> List[str]:
        """Process files sequentially (original behavior)."""
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            files_task = progress.add_task("Processing files", total=len(self.config.input_files))
            
            for file_idx, file_path in enumerate(self.config.input_files):
                progress.update(files_task, description=f"Processing {Path(file_path).name}")
                
                # Read file
                text = self._read_single_file(file_path)
                
                if not text:
                    self.console.print(f"⚠️  Skipping empty file: {file_path}")
                    progress.advance(files_task)
                    continue
                
                # Check if document classification is required
                if self.config.require_problem or self.config.require_tool:
                    classification = self._classify_document(text)
                    
                    if self.config.require_problem and classification == 'TOOL':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as TOOL")
                        progress.advance(files_task)
                        continue
                    elif self.config.require_tool and classification == 'PROBLEM':
                        self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as PROBLEM")
                        progress.advance(files_task)
                        continue
                
                # Add file header to results
                file_header = f"\n{'='*60}\nFILE: {Path(file_path).name}\n{'='*60}\n"
                all_results.append(file_header)
                
                # Chunk text
                chunks = self._chunk_text(text)
                
                # Validate chunking coverage
                self._validate_chunking(text, chunks)
                
                # Process chunks for this file
                for idx, chunk in enumerate(chunks):
                    # Generate prompt for current mode
                    prompt = self._get_prompt_for_mode(self.config.mode, chunk, idx, len(chunks))
                    
                    # Make API call
                    result = self._call_api(prompt)
                    
                    # Store result
                    all_results.append(result)
                    self.processed_chunks += 1
                
                progress.advance(files_task)
        
        self.results = all_results
        return all_results
    
    def _process_files_parallel(self) -> List[str]:
        """Process files in parallel using ThreadPoolExecutor."""
        # Prepare all chunk data from all files
        all_chunk_data = []
        file_results_mapping = {}
        chunk_idx = 0
        
        for file_idx, file_path in enumerate(self.config.input_files):
            # Read file
            text = self._read_single_file(file_path)
            
            if not text:
                self.console.print(f"⚠️  Skipping empty file: {file_path}")
                continue
            
            # Check if document classification is required
            if self.config.require_problem or self.config.require_tool:
                classification = self._classify_document(text)
                
                if self.config.require_problem and classification == 'TOOL':
                    self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as TOOL")
                    continue
                elif self.config.require_tool and classification == 'PROBLEM':
                    self.console.print(f"⚠️  Skipping {Path(file_path).name} - classified as PROBLEM")
                    continue
            
            # Chunk text
            chunks = self._chunk_text(text)
            
            # Validate chunking coverage
            self._validate_chunking(text, chunks)
            
            # Record file info and chunk ranges
            file_start_idx = chunk_idx
            file_results_mapping[file_idx] = {
                'file_path': file_path,
                'start_idx': file_start_idx,
                'end_idx': file_start_idx + len(chunks),
                'chunk_count': len(chunks)
            }
            
            # Add chunk data for this file
            for local_idx, chunk in enumerate(chunks):
                all_chunk_data.append((chunk_idx, chunk, local_idx, len(chunks)))
                chunk_idx += 1
        
        if not all_chunk_data:
            self.console.print("⚠️  No files to process after filtering")
            return []
        
        # Initialize results list with placeholders
        results = [None] * len(all_chunk_data)
        
        # Update total chunks for progress tracking
        self.total_chunks = len(all_chunk_data)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing chunks", total=self.total_chunks)
            
            with ThreadPoolExecutor(max_workers=self.config.worker_count) as executor:
                # Submit all tasks
                future_to_chunk = {executor.submit(self._process_chunk_parallel, data): data for data in all_chunk_data}
                
                # Process completed tasks
                for future in as_completed(future_to_chunk):
                    chunk_idx, result = future.result()
                    results[chunk_idx] = result
                    progress.advance(task)
        
        # Reconstruct results organized by file
        all_results = []
        for file_idx in sorted(file_results_mapping.keys()):
            file_info = file_results_mapping[file_idx]
            
            # Add file header
            file_header = f"\n{'='*60}\nFILE: {Path(file_info['file_path']).name}\n{'='*60}\n"
            all_results.append(file_header)
            
            # Add results for this file
            for i in range(file_info['start_idx'], file_info['end_idx']):
                all_results.append(results[i])
        
        self.results = all_results
        return all_results
    
    def save_results(self):
        """Save extraction results to output file or directory."""
        try:
            if self.config.batch_mode:
                # Directory mode: save individual files
                self._save_results_to_directory()
            else:
                # Single file mode: save to single output file
                self._save_results_to_file()
            
        except Exception as e:
            self.console.print(f"❌ Failed to save results: {e}")
    
    def _save_results_to_file(self):
        """Save results to a single output file."""
        # Combine all results
        combined_result = "\n\n".join(self.results)
        
        # Save to file
        with open(self.config.output_file, 'w', encoding='utf-8') as f:
            f.write(combined_result)
        
        self.console.print(f"✅ Results saved to {self.config.output_file}")
        
        # Display preview
        self.console.print(Panel(
            Markdown(combined_result[:1000] + "..." if len(combined_result) > 1000 else combined_result),
            title="Extraction Results Preview",
            expand=False
        ))
    
    def _save_results_to_directory(self):
        """Save results to individual files in a directory."""
        # Ensure output directory exists
        output_dir = Path(self.config.output_file)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse results by file
        current_file = None
        current_results = []
        files_processed = 0
        
        for result in self.results:
            if result.startswith("\n" + "="*60 + "\nFILE:"):
                # Save previous file's results if any
                if current_file and current_results:
                    self._save_individual_file_results(output_dir, current_file, current_results)
                    files_processed += 1
                
                # Extract new filename
                lines = result.split('\n')
                file_line = next((line for line in lines if line.startswith('FILE:')), None)
                if file_line:
                    current_file = file_line.replace('FILE: ', '').strip()
                    current_results = []
            else:
                # Add to current file's results
                if current_file:
                    current_results.append(result)
        
        # Save the last file's results
        if current_file and current_results:
            self._save_individual_file_results(output_dir, current_file, current_results)
            files_processed += 1
        
        self.console.print(f"✅ Results saved to {files_processed} files in {output_dir}")
        
        # Display directory summary
        self.console.print(Panel(
            f"📁 Output directory: {output_dir}\n"
            f"📄 Files created: {files_processed}\n"
            f"🔧 Mode: {self.config.mode.value}",
            title="Directory Output Summary",
            expand=False
        ))
    
    def _save_individual_file_results(self, output_dir: Path, filename: str, results: List[str]):
        """Save results for an individual file."""
        # Create output filename based on input filename and mode
        base_name = Path(filename).stem
        output_filename = f"{base_name}_{self.config.mode.value}.txt"
        output_path = output_dir / output_filename
        
        # Combine results for this file
        combined_result = "\n\n".join(results)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Extract-o-matic Results\n")
            f.write(f"# Source: {filename}\n")
            f.write(f"# Mode: {self.config.mode.value}\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(combined_result)
        
        self.console.print(f"  💾 {output_filename}")

def discover_files(input_path: str, recursive: bool = True) -> List[str]:
    """Discover .txt, .md, and .pdf files in a directory or return single file."""
    supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
    
    if os.path.isfile(input_path):
        # Single file - check if supported
        file_ext = Path(input_path).suffix.lower()
        if file_ext in supported_extensions:
            return [input_path]
        else:
            console.print(f"❌ Unsupported file type: {file_ext}")
            return []
    
    elif os.path.isdir(input_path):
        # Directory - find all supported files
        files = []
        
        if recursive:
            # Recursive search using Path.rglob()
            path_obj = Path(input_path)
            for ext in supported_extensions:
                pattern = f"**/*{ext}"
                files.extend([str(p) for p in path_obj.rglob(f"*{ext}")])
        else:
            # Non-recursive search using glob
            for ext in supported_extensions:
                pattern = os.path.join(input_path, f"*{ext}")
                files.extend(glob.glob(pattern))
        
        # Sort for consistent processing order
        files.sort()
        
        if not files:
            search_type = "recursively" if recursive else "in directory"
            console.print(f"❌ No supported files found {search_type}: {input_path}")
            console.print(f"Supported extensions: {', '.join(supported_extensions)}")
        
        return files
    
    else:
        console.print(f"❌ Path not found: {input_path}")
        return []

def get_file_type(file_path: str) -> str:
    """Determine file type based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.md', '.markdown']:
        return 'markdown'
    else:
        return 'text'

def read_pdf_file(file_path: str) -> str:
    """Extract text from PDF file."""
    if not PDF_AVAILABLE:
        console.print("❌ PDF processing requires PyPDF2: pip install PyPDF2")
        return ""
    
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        console.print(f"❌ Failed to read PDF {file_path}: {e}")
        return ""

def load_model_config(config_file: str, model_shortname: str) -> ModelConfig:
    """Load model configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find model by shortname
        for server in config['servers']:
            if server['shortname'] == model_shortname:
                # Handle environment variables in API key (following PullR pattern)
                openai_api_key_config = server['openai_api_key']
                openai_api_key = None
                
                if openai_api_key_config == "${OPENAI_API_KEY}":
                    openai_api_key = os.environ.get('OPENAI-API-KEY') or os.environ.get('OPENAI_API_KEY')
                    if not openai_api_key:
                        console.print("❌ OpenAI API key is configured to use environment variable "
                                    "'OPENAI-API-KEY' or 'OPENAI_API_KEY', but neither is set.")
                        sys.exit(1)
                elif openai_api_key_config:
                    openai_api_key = openai_api_key_config
                else:
                    console.print(f"❌ 'openai_api_key' not specified for model {model_shortname}")
                    sys.exit(1)
                
                return ModelConfig(
                    server=server['server'],
                    shortname=server['shortname'],
                    openai_api_key=openai_api_key,
                    openai_api_base=server['openai_api_base'],
                    openai_model=server['openai_model']
                )
        
        # If not found, list available models
        available_models = [server['shortname'] for server in config['servers']]
        raise ValueError(f"Model '{model_shortname}' not found. Available models: {available_models}")
        
    except Exception as e:
        console.print(f"❌ Failed to load model config: {e}")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract-o-matic bioinformatics workflow extraction script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extraction Modes:
  workflow         - Extract bioinformatics workflows and BV-BRC service requirements
  problem_summary  - Identify scientific objectives and generate research questions
  bvbrc_mapping    - Map scientific problems to specific BV-BRC tools
  workflow_calculus - Generate compact Workflow Calculus descriptions
  classification   - Classify text as TOOL or PROBLEM
  experimental_protocol - Extract experimental protocols and laboratory procedures
  automated_protocol - Map research to automated protocol using fixed robots, humanoid robots, and AGI
  dataset_extraction - Extract comprehensive information about datasets used, produced, or cited
  hypothesis_extraction - Extract hypotheses and generate Wisteria-compatible JSON files
  open_problems - Extract open problems and research gaps with Wisteria-compatible JSON files
  knowledge_graph - Extract semantic relationships and convert to RDF format

Examples:
  python extract-o-matic.py paper.txt --mode workflow --model scout
  python extract-o-matic.py papers/ --mode workflow --model scout --output results_dir/
  python extract-o-matic.py papers/ --mode workflow --model scout --no-recursive
  python extract-o-matic.py paper.pdf --mode bvbrc_mapping --model llama --output results.txt
  python extract-o-matic.py paper.txt --mode workflow_calculus --notation notation.txt --rubric rubric.txt
  python extract-o-matic.py papers/ --mode experimental_protocol --model scout --output protocols_dir/
  python extract-o-matic.py paper.txt --mode automated_protocol --model scout --output automation_plan.txt
  python extract-o-matic.py paper.txt --mode dataset_extraction --model scout --output datasets.txt
  python extract-o-matic.py papers/ --mode dataset_extraction --model scout --output datasets_dir/
  python extract-o-matic.py paper.txt --mode hypothesis_extraction --model scout --output hypotheses.json
  python extract-o-matic.py papers/ --mode hypothesis_extraction --model scout --output hypotheses_dir/
  python extract-o-matic.py paper.txt --mode open_problems --model scout --output open_problems.json
  python extract-o-matic.py papers/ --mode open_problems --model scout --output open_problems_dir/
  python extract-o-matic.py paper.txt --mode knowledge_graph --model scout --output knowledge_graph.rdf
  python extract-o-matic.py papers/ --mode knowledge_graph --model scout --output knowledge_graphs_dir/
  python extract-o-matic.py paper.txt --mode workflow --require-problem --model scout
  python extract-o-matic.py papers/ --mode workflow --require-tool --model scout
  python extract-o-matic.py paper.txt --mode workflow --model scout --workers 4
  python extract-o-matic.py papers/ --mode workflow --model scout --workers 8 --output results_dir/
        """
    )
    
    parser.add_argument('input_path', help='Input file or directory to process (.txt, .md, .pdf files)')
    parser.add_argument('--mode', type=str, 
                       choices=[mode.value for mode in ExtractionMode], 
                       default='workflow', help='Extraction mode (default: workflow)')
    parser.add_argument('--output', default='extraction_results.txt', 
                       help='Output file or directory (default: extraction_results.txt). For directory input, use directory path ending with / to create individual files per input file.')
    parser.add_argument('--model', default='scout', 
                       help='Model shortname from config file (default: scout)')
    parser.add_argument('--config', default='model_servers.yaml', 
                       help='Model configuration file (default: model_servers.yaml)')
    parser.add_argument('--chunk-size', type=int, default=3000, 
                       help='Chunk size in tokens/characters (default: 3000)')
    parser.add_argument('--max-tokens', type=int, default=2000, 
                       help='Maximum tokens per API response (default: 2000)')
    parser.add_argument('--temperature', type=float, default=0.0, 
                       help='API temperature (default: 0.0)')
    parser.add_argument('--character-limit', type=int, default=100000, 
                       help='Maximum characters to process (default: 100000)')
    parser.add_argument('--notation', default='notation.txt', help='Notation guidance file (default: notation.txt)')
    parser.add_argument('--rubric', default='rubric.txt', help='Rubric guidance file (default: rubric.txt)')
    parser.add_argument('--require-problem', action='store_true', 
                       help='Only proceed with extraction if paper is classified as PROBLEM (runs classification first)')
    parser.add_argument('--require-tool', action='store_true', 
                       help='Only proceed with extraction if paper is classified as TOOL (runs classification first)')
    parser.add_argument('--no-recursive', action='store_true', 
                       help='Do not search subdirectories recursively (default: recursive search enabled)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker threads for parallel processing (default: 1)')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate mutually exclusive flags
    if args.require_problem and args.require_tool:
        console.print("❌ Error: --require-problem and --require-tool flags cannot be used together")
        sys.exit(1)
    
    # Discover input files
    recursive = not args.no_recursive
    input_files = discover_files(args.input_path, recursive=recursive)
    if not input_files:
        sys.exit(1)
    
    # Determine if batch mode
    batch_mode = len(input_files) > 1
    
    # Welcome message
    classification_filter = "None"
    if args.require_problem:
        classification_filter = "PROBLEM only"
    elif args.require_tool:
        classification_filter = "TOOL only"
    
    console.print(Panel(
        "[bold blue]Extract-o-matic Bioinformatics Workflow Extraction Script[/bold blue]\n"
        f"Mode: {args.mode}\n"
        f"Model: {args.model}\n"
        f"Input: {args.input_path}\n"
        f"Files to process: {len(input_files)}\n"
        f"Batch mode: {'Yes' if batch_mode else 'No'}\n"
        f"Recursive search: {'Yes' if recursive else 'No'}\n"
        f"Workers: {args.workers}\n"
        f"Output: {args.output}\n"
        f"Classification Filter: {classification_filter}",
        title="Configuration",
        expand=False
    ))
    
    # Show discovered files if batch mode
    if batch_mode:
        console.print("📁 Discovered files:")
        for file_path in input_files:
            file_type = get_file_type(file_path)
            console.print(f"  • {Path(file_path).name} ({file_type})")
    
    # Load model configuration
    model_config = load_model_config(args.config, args.model)
    
    # Setup guidance files
    guidance_files = {
        'notation': args.notation,
        'rubric': args.rubric
    }
    
    # Create extraction configuration
    config = ExtractionConfig(
        input_file=input_files[0] if not batch_mode else "",
        output_file=args.output,
        mode=ExtractionMode(args.mode),
        model_config=model_config,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        character_limit=args.character_limit,
        guidance_files=guidance_files,
        require_problem=args.require_problem,
        require_tool=args.require_tool,
        batch_mode=batch_mode,
        input_files=input_files,
        worker_count=args.workers
    )
    
    # Create extractor and run
    extractor = ExtractOMatic(config)
    results = extractor.extract()
    extractor.save_results()
    
    # Summary
    if batch_mode:
        console.print(Panel(
            f"✅ Batch extraction complete!\n"
            f"📁 Processed {len(input_files)} files\n"
            f"📊 Processed {extractor.processed_chunks} chunks total\n"
            f"💾 Results saved to {args.output}",
            title="Summary",
            expand=False
        ))
    else:
        console.print(Panel(
            f"✅ Extraction complete!\n"
            f"📊 Processed {extractor.processed_chunks} chunks\n"
            f"💾 Results saved to {args.output}",
            title="Summary",
            expand=False
        ))

if __name__ == "__main__":
    main()