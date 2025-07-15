#!/usr/bin/env python3
"""
Dataset Analysis Tool - LLM-powered CSV Analysis

This script analyzes the datasets_summary.csv file using LLM to provide:
- High-level classification of dataset types
- Summary of the top 100 kinds of data
- Aggregated analysis by scientific domain (physics, materials science, chemistry, biology, earth sciences)
- Summary of largest datasets

Usage:
    python analyze_datasets.py [--model <model_shortname>] [--config <config_file>] [--chunk-size <size>] [--output <output_file>]

Examples:
    python analyze_datasets.py --model gpt4 --output analysis_report.md
    python analyze_datasets.py --model llama --config model_servers.yaml --chunk-size 50
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table


DEFAULT_MODEL_CONFIG_FILE = '../model_servers.yaml'
DEFAULT_CHUNK_SIZE = 200
DEFAULT_WORKERS = 4
DEFAULT_OUTPUT_FILE = 'dataset_analysis_report.md'

console = Console()


class DatasetAnalyzer:
    def __init__(self, model_shortname: str, config_file: str = DEFAULT_MODEL_CONFIG_FILE):
        self.model_shortname = model_shortname
        self.config_file = config_file
        self.client = None
        self.model_config = None
        self._load_model_config()
        
    def _load_model_config(self):
        """Load model configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Handle both formats: 'models' dict and 'servers' list
            if 'models' in config:
                models = config['models']
                if self.model_shortname not in models:
                    console.print(f"[red]Error: Model '{self.model_shortname}' not found in {self.config_file}")
                    sys.exit(1)
                self.model_config = models[self.model_shortname]
                # Ensure we have server info for 'models' format
                if 'server' not in self.model_config:
                    self.model_config['server'] = self.model_config.get('base_url', 'unknown')
                if 'shortname' not in self.model_config:
                    self.model_config['shortname'] = self.model_shortname
                
                # Initialize OpenAI client
                self.client = OpenAI(
                    api_key=self.model_config.get('api_key', ''),
                    base_url=self.model_config.get('base_url', 'https://api.openai.com/v1')
                )
                
            elif 'servers' in config:
                servers = config['servers']
                model_found = None
                
                for server in servers:
                    if server.get('shortname') == self.model_shortname:
                        model_found = server
                        break
                        
                if not model_found:
                    console.print(f"[red]Error: Model '{self.model_shortname}' not found in {self.config_file}")
                    console.print(f"[yellow]Available models: {[s.get('shortname') for s in servers]}")
                    sys.exit(1)
                    
                # Convert server format to model config format
                self.model_config = {
                    'model': model_found.get('openai_model', self.model_shortname),
                    'api_key': model_found.get('openai_api_key', 'no_key'),
                    'base_url': model_found.get('openai_api_base', 'https://api.openai.com/v1'),
                    'server': model_found.get('server', 'unknown'),
                    'shortname': model_found.get('shortname', self.model_shortname)
                }
                
                # Initialize OpenAI client
                self.client = OpenAI(
                    api_key=self.model_config['api_key'],
                    base_url=self.model_config['base_url']
                )
            else:
                console.print(f"[red]Error: Invalid configuration format in {self.config_file}")
                console.print("[yellow]Expected 'models' or 'servers' section")
                sys.exit(1)
            
        except FileNotFoundError:
            console.print(f"[red]Error: Configuration file {self.config_file} not found")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}")
            sys.exit(1)
            
    def get_model_display_name(self) -> str:
        """Get a display name showing both model and server"""
        shortname = self.model_config.get('shortname', 'unknown')
        server = self.model_config.get('server', 'unknown')
        model = self.model_config.get('model', 'unknown')
        return f"{shortname} (on {server}: {model})"
            
    def load_csv_data(self, csv_file: str) -> List[Dict[str, str]]:
        """Load and parse the CSV file"""
        console.print(f"[blue]Loading CSV data from {csv_file}...")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                
            console.print(f"[green]Loaded {len(data)} dataset entries")
            return data
            
        except Exception as e:
            console.print(f"[red]Error loading CSV: {e}")
            sys.exit(1)
            
    def chunk_data(self, data: List[Dict[str, str]], chunk_size: int) -> List[List[Dict[str, str]]]:
        """Split data into chunks for processing"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks
        
    def analyze_chunk_worker(self, chunk: List[Dict[str, str]], chunk_idx: int) -> Dict[str, Any]:
        """Analyze a chunk of datasets using LLM"""
        
        # Prepare chunk summary for LLM
        chunk_text = f"Dataset Chunk {chunk_idx + 1} ({len(chunk)} datasets):\n\n"
        
        for i, dataset in enumerate(chunk, 1):
            chunk_text += f"Dataset {i}:\n"
            chunk_text += f"Title: {dataset.get('Dataset_Title', 'N/A')}\n"
            chunk_text += f"Type: {dataset.get('Type', 'N/A')}\n"
            chunk_text += f"Scale: {dataset.get('Scale', 'N/A')}\n"
            chunk_text += f"Creation: {dataset.get('Creation', 'N/A')}\n"
            chunk_text += f"Usage: {dataset.get('Usage', 'N/A')}\n"
            chunk_text += f"Additional Info: {dataset.get('Additional_Info', 'N/A')[:200]}...\n\n"
            
        prompt = f"""Analyze this chunk of scientific datasets. Classify each dataset by:
1. Domain: Physics, Materials Science, Chemistry, Biology, Earth Sciences, Computer Science, Mathematics, Other
2. Data Type: Experimental, Computational, Literature, Imaging, Spectral, Survey, Other  
3. Scale: Small, Medium, Large, Very Large

Return ONLY valid JSON with this exact format:

```json
{{
    "datasets": [
        {{"id": 1, "title": "Short title", "domain": "Physics", "data_type": "Experimental", "scale": "Medium", "keywords": ["key1", "key2"]}}
    ],
    "summary": {{
        "domain_distribution": {{"Physics": 3, "Chemistry": 2}},
        "data_type_distribution": {{"Experimental": 4, "Computational": 1}},
        "scale_distribution": {{"Medium": 3, "Small": 2}},
        "key_insights": ["Pattern 1", "Pattern 2"]
    }}
}}
```

Use short titles, valid JSON only, double quotes, no trailing commas.

Datasets:
{chunk_text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a scientific data analyst expert at classifying and analyzing research datasets across multiple scientific domains."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            
            # Multiple attempts to extract JSON
            json_result = self._extract_json_from_response(result_text, chunk_idx)
            if json_result:
                return json_result
            else:
                console.print(f"[yellow]Warning: Failed to parse JSON in chunk {chunk_idx + 1}, using fallback")
                return self._create_fallback_analysis(chunk)
                
        except Exception as e:
            console.print(f"[red]Error analyzing chunk {chunk_idx + 1}: {e}")
            return self._create_fallback_analysis(chunk)
            
    def _extract_json_from_response(self, response_text: str, chunk_idx: int) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with multiple strategies"""
        
        # Strategy 1: Look for ```json blocks
        start_idx = response_text.find('```json')
        if start_idx != -1:
            end_idx = response_text.find('```', start_idx + 7)
            if end_idx != -1:
                json_text = response_text[start_idx + 7:end_idx].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]JSON parse error in chunk {chunk_idx + 1} (strategy 1): {str(e)[:100]}")
        
        # Strategy 2: Look for { } blocks (complete JSON objects)
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                # Try to parse each potential JSON block
                potential_json = json.loads(match)
                # Validate it has expected structure
                if isinstance(potential_json, dict) and ('datasets' in potential_json or 'summary' in potential_json):
                    return potential_json
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Look for larger nested JSON blocks
        brace_count = 0
        start_pos = None
        
        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos is not None:
                    try:
                        json_text = response_text[start_pos:i+1]
                        potential_json = json.loads(json_text)
                        if isinstance(potential_json, dict) and ('datasets' in potential_json or 'summary' in potential_json):
                            return potential_json
                    except json.JSONDecodeError:
                        continue
        
        # Strategy 4: Try to fix common JSON issues and parse
        cleaned_text = response_text
        # Remove markdown formatting
        cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        # Remove extra text before/after JSON
        lines = cleaned_text.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('{') or in_json:
                in_json = True
                json_lines.append(line)
                if line.endswith('}') and line.count('}') >= line.count('{'):
                    break
                    
        if json_lines:
            try:
                json_text = '\n'.join(json_lines)
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass
        
        # Strategy 5: Aggressive JSON repair - fix common issues
        try:
            repaired_json = self._repair_json(response_text)
            if repaired_json:
                potential_json = json.loads(repaired_json)
                if isinstance(potential_json, dict) and ('datasets' in potential_json or 'summary' in potential_json):
                    console.print(f"[green]JSON repaired successfully for chunk {chunk_idx + 1}")
                    return potential_json
        except json.JSONDecodeError:
            pass
        
        # Strategy 6: Save raw response for debugging
        debug_file = f"debug_chunk_{chunk_idx + 1}_response.txt"
        try:
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"Raw response for chunk {chunk_idx + 1}:\n")
                f.write("=" * 50 + "\n")
                f.write(response_text)
            console.print(f"[yellow]Raw response saved to {debug_file} for debugging")
        except Exception:
            pass
        
        # If all strategies fail, return None
        console.print(f"[red]All JSON extraction strategies failed for chunk {chunk_idx + 1}")
        return None
        
    def _repair_json(self, text: str) -> Optional[str]:
        """Attempt to repair common JSON formatting issues"""
        
        # Find potential JSON block
        start_brace = text.find('{')
        if start_brace == -1:
            return None
            
        # Extract everything from first { to end
        json_part = text[start_brace:]
        
        # Find the matching closing brace
        brace_count = 0
        end_pos = None
        for i, char in enumerate(json_part):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
                    
        if end_pos:
            json_text = json_part[:end_pos]
        else:
            json_text = json_part
            
        # Common repairs
        repairs = [
            # Fix missing commas before closing braces/brackets
            (r'"\s*\n\s*}', '"\n}'),
            (r'"\s*\n\s*]', '"\n]'),
            (r'}\s*\n\s*}', '}\n}'),
            (r']\s*\n\s*}', ']\n}'),
            (r'}\s*\n\s*]', '}\n]'),
            # Fix trailing commas
            (r',(\s*[}\]])', r'\1'),
            # Fix missing commas between objects
            (r'}\s*\n\s*{', '},\n{'),
            (r']\s*\n\s*"', '],\n"'),
            (r'"\s*\n\s*"', '",\n"'),
            # Fix unescaped quotes in strings
            (r'(?<!\\)"(?=[^,}\]:])', '\\"'),
            # Fix newlines in strings
            (r'(?<=")([^"]*)\n([^"]*?)(?=")', r'\1\\n\2'),
        ]
        
        repaired = json_text
        for pattern, replacement in repairs:
            try:
                repaired = re.sub(pattern, replacement, repaired, flags=re.MULTILINE)
            except Exception:
                continue
                
        # Try to add missing commas more aggressively
        lines = repaired.split('\n')
        repaired_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            repaired_lines.append(line)
            
            # If current line ends with " or } or ], and next line starts with " or {
            if i < len(lines) - 1:
                next_stripped = lines[i + 1].strip()
                if (stripped.endswith('"') or stripped.endswith('}') or stripped.endswith(']')) and \
                   (next_stripped.startswith('"') or next_stripped.startswith('{')):
                    # Check if we need a comma
                    if not stripped.endswith(',') and not next_stripped.startswith('}') and not next_stripped.startswith(']'):
                        repaired_lines[-1] = line.rstrip() + ','
                        
        return '\n'.join(repaired_lines)
            
    def _create_fallback_analysis(self, chunk: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create a basic analysis when LLM analysis fails"""
        
        # Try to make intelligent guesses based on keywords in titles/descriptions
        fallback_datasets = []
        domain_counts = Counter()
        data_type_counts = Counter()
        scale_counts = Counter()
        
        for i, dataset in enumerate(chunk):
            title = dataset.get('Dataset_Title', '').lower()
            data_type = dataset.get('Type', '').lower()
            usage = dataset.get('Usage', '').lower()
            
            # Simple domain classification based on keywords
            domain = "Other"
            if any(word in title + data_type + usage for word in ['physic', 'quantum', 'particle', 'energy']):
                domain = "Physics"
            elif any(word in title + data_type + usage for word in ['material', 'crystal', 'metal', 'polymer']):
                domain = "Materials Science"
            elif any(word in title + data_type + usage for word in ['chemical', 'molecule', 'reaction', 'compound']):
                domain = "Chemistry"
            elif any(word in title + data_type + usage for word in ['bio', 'cell', 'protein', 'gene', 'organism']):
                domain = "Biology"
            elif any(word in title + data_type + usage for word in ['earth', 'climate', 'environment', 'geology']):
                domain = "Earth Sciences"
            
            # Simple data type classification
            dtype = "Other"
            if any(word in data_type for word in ['experimental', 'measurement', 'test']):
                dtype = "Experimental"
            elif any(word in data_type for word in ['computational', 'simulation', 'model']):
                dtype = "Computational"
            elif any(word in data_type for word in ['literature', 'review', 'survey']):
                dtype = "Literature"
            elif any(word in data_type for word in ['spectral', 'spectrum', 'xray', 'nmr']):
                dtype = "Spectral"
            
            # Simple scale classification
            scale_text = dataset.get('Scale', '').lower() + dataset.get('Aggregate_Scale', '').lower()
            scale = "Medium"  # default
            if any(word in scale_text for word in ['small', 'few', 'limited']):
                scale = "Small"
            elif any(word in scale_text for word in ['large', 'big', 'extensive', 'massive']):
                scale = "Large"
            
            fallback_datasets.append({
                "id": i + 1,
                "title": dataset.get('Dataset_Title', 'Unknown')[:50],  # Truncate long titles
                "domain": domain,
                "data_type": dtype,
                "scale": scale,
                "keywords": []
            })
            
            domain_counts[domain] += 1
            data_type_counts[dtype] += 1
            scale_counts[scale] += 1
        
        return {
            "datasets": fallback_datasets,
            "summary": {
                "domain_distribution": dict(domain_counts),
                "data_type_distribution": dict(data_type_counts),
                "scale_distribution": dict(scale_counts),
                "key_insights": ["Analysis completed using fallback method - LLM parsing failed"]
            }
        }
        
    def aggregate_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all chunks"""
        console.print("[blue]Aggregating results...")
        
        all_datasets = []
        domain_counts = Counter()
        data_type_counts = Counter()
        scale_counts = Counter()
        all_keywords = Counter()
        all_insights = []
        
        for result in chunk_results:
            all_datasets.extend(result.get('datasets', []))
            
            summary = result.get('summary', {})
            domain_counts.update(summary.get('domain_distribution', {}))
            data_type_counts.update(summary.get('data_type_distribution', {}))
            scale_counts.update(summary.get('scale_distribution', {}))
            all_insights.extend(summary.get('key_insights', []))
            
            # Collect keywords
            for dataset in result.get('datasets', []):
                all_keywords.update(dataset.get('keywords', []))
                
        return {
            'total_datasets': len(all_datasets),
            'datasets': all_datasets,
            'domain_distribution': dict(domain_counts),
            'data_type_distribution': dict(data_type_counts),
            'scale_distribution': dict(scale_counts),
            'top_keywords': dict(all_keywords.most_common(100)),
            'insights': all_insights
        }
        
    def process_chunks_parallel(self, chunks: List[List[Dict[str, str]]], num_workers: int) -> List[Dict[str, Any]]:
        """Process chunks in parallel using ThreadPoolExecutor"""
        console.print(f"[blue]Processing {len(chunks)} chunks using {num_workers} workers...")
        
        chunk_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Analyzing chunks...", total=len(chunks))
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all chunk analysis tasks
                future_to_chunk = {
                    executor.submit(self.analyze_chunk_worker, chunk, i): (chunk, i) 
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk, chunk_idx = future_to_chunk[future]
                    try:
                        result = future.result()
                        chunk_results.append((chunk_idx, result))
                        progress.update(task, description=f"Completed chunk {chunk_idx+1}/{len(chunks)}")
                        progress.advance(task)
                    except Exception as e:
                        console.print(f"[red]Error processing chunk {chunk_idx + 1}: {e}")
                        # Add fallback result
                        chunk_results.append((chunk_idx, self._create_fallback_analysis(chunk)))
                        progress.advance(task)
        
        # Sort results by chunk index to maintain order
        chunk_results.sort(key=lambda x: x[0])
        return [result for _, result in chunk_results]
        
    def consolidate_summaries(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate multiple chunk summaries using divide-and-conquer approach"""
        console.print("[blue]Consolidating chunk summaries...")
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # If we have many results, consolidate in batches first
        if len(chunk_results) > 8:
            console.print(f"[blue]Performing hierarchical consolidation of {len(chunk_results)} chunks...")
            batch_size = 4
            consolidated_batches = []
            
            for i in range(0, len(chunk_results), batch_size):
                batch = chunk_results[i:i + batch_size]
                if len(batch) > 1:
                    consolidated_batch = self._consolidate_batch(batch, f"batch_{i//batch_size + 1}")
                    consolidated_batches.append(consolidated_batch)
                else:
                    consolidated_batches.append(batch[0])
            
            # Final consolidation of batches
            return self._consolidate_batch(consolidated_batches, "final")
        else:
            # Direct consolidation for smaller numbers
            return self._consolidate_batch(chunk_results, "final")
            
    def _consolidate_batch(self, batch_results: List[Dict[str, Any]], batch_name: str) -> Dict[str, Any]:
        """Consolidate a batch of results using LLM"""
        console.print(f"[blue]Consolidating {batch_name} batch ({len(batch_results)} summaries)...")
        
        # Prepare consolidation prompt
        summaries_text = ""
        total_datasets = 0
        
        for i, result in enumerate(batch_results, 1):
            summary = result.get('summary', {})
            total_datasets += len(result.get('datasets', []))
            
            summaries_text += f"\nSummary {i} ({len(result.get('datasets', []))} datasets):\n"
            summaries_text += f"Domain Distribution: {summary.get('domain_distribution', {})}\n"
            summaries_text += f"Data Type Distribution: {summary.get('data_type_distribution', {})}\n"
            summaries_text += f"Scale Distribution: {summary.get('scale_distribution', {})}\n"
            summaries_text += f"Key Insights: {summary.get('key_insights', [])}\n"
            
        prompt = f"""Consolidate these {len(batch_results)} dataset analysis summaries into a single comprehensive summary.

Total datasets across all summaries: {total_datasets}

{summaries_text}

Please provide a consolidated analysis in valid JSON format. Ensure the JSON is properly formatted with no syntax errors:

```json
{{
    "summary": {{
        "domain_distribution": {{"Physics": 10, "Chemistry": 8, "Biology": 5}},
        "data_type_distribution": {{"Experimental": 15, "Computational": 8}},
        "scale_distribution": {{"Small": 12, "Medium": 8, "Large": 3}},
        "key_insights": ["insight 1", "insight 2", "insight 3"],
        "emerging_patterns": ["pattern 1", "pattern 2"],
        "research_trends": ["trend 1", "trend 2"]
    }}
}}
```

IMPORTANT: 
- Use only valid JSON syntax
- Use double quotes for all strings
- Ensure all brackets and braces are properly closed
- Do not include trailing commas
- Merge and sum the distributions, consolidate insights, and identify overarching patterns."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior research analyst consolidating multiple dataset analysis summaries into comprehensive insights."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response using robust method
            json_result = self._extract_json_from_response(result_text, f"{batch_name}_consolidation")
            if json_result:
                # Combine all datasets from batch
                all_datasets = []
                for result in batch_results:
                    all_datasets.extend(result.get('datasets', []))
                
                return {
                    'datasets': all_datasets,
                    'summary': json_result.get('summary', {}),
                    'total_datasets': len(all_datasets)
                }
            else:
                console.print(f"[yellow]Warning: Failed to parse JSON in {batch_name} consolidation")
                return self._fallback_consolidation(batch_results)
                
        except Exception as e:
            console.print(f"[red]Error consolidating {batch_name} batch: {e}")
            return self._fallback_consolidation(batch_results)
            
    def _fallback_consolidation(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback consolidation when LLM fails"""
        console.print("[yellow]Using fallback consolidation...")
        
        all_datasets = []
        domain_counts = Counter()
        data_type_counts = Counter()
        scale_counts = Counter()
        all_insights = []
        
        for result in batch_results:
            all_datasets.extend(result.get('datasets', []))
            summary = result.get('summary', {})
            domain_counts.update(summary.get('domain_distribution', {}))
            data_type_counts.update(summary.get('data_type_distribution', {}))
            scale_counts.update(summary.get('scale_distribution', {}))
            all_insights.extend(summary.get('key_insights', []))
            
        return {
            'datasets': all_datasets,
            'summary': {
                'domain_distribution': dict(domain_counts),
                'data_type_distribution': dict(data_type_counts),
                'scale_distribution': dict(scale_counts),
                'key_insights': all_insights[:20],  # Limit insights
                'emerging_patterns': ['Fallback analysis - manual review needed'],
                'research_trends': ['Consolidation failed - requires manual analysis']
            },
            'total_datasets': len(all_datasets)
        }
        
    def generate_final_summary(self, consolidated_results: Dict[str, Any]) -> str:
        """Generate a comprehensive summary using LLM"""
        console.print("[blue]Generating final comprehensive summary...")
        
        summary = consolidated_results.get('summary', {})
        
        # Extract top keywords from datasets if available
        all_keywords = Counter()
        for dataset in consolidated_results.get('datasets', []):
            all_keywords.update(dataset.get('keywords', []))
        
        prompt = f"""Based on the analysis of {consolidated_results.get('total_datasets', 0)} scientific datasets, create a comprehensive summary report.

**Data Summary:**
- Total datasets: {consolidated_results.get('total_datasets', 0)}
- Domain distribution: {summary.get('domain_distribution', {})}
- Data type distribution: {summary.get('data_type_distribution', {})}
- Scale distribution: {summary.get('scale_distribution', {})}
- Top keywords: {list(all_keywords.keys())[:50] if all_keywords else 'Not available'}

**Key insights:**
{chr(10).join(summary.get('key_insights', [])[:20])}

**Emerging patterns:**
{chr(10).join(summary.get('emerging_patterns', [])[:10])}

**Research trends:**
{chr(10).join(summary.get('research_trends', [])[:10])}

Please provide:

1. **Executive Summary**: High-level overview of the dataset collection
2. **Scientific Domain Analysis**: Detailed breakdown by domain (physics, materials science, chemistry, biology, earth sciences, etc.)
3. **Data Type Characteristics**: Analysis of experimental vs computational vs literature datasets
4. **Scale and Scope**: Analysis of dataset sizes and research scope
5. **Emerging Trends**: Key research areas, methodologies, and technologies
6. **Top 100 Data Types**: Summary of the most common types of datasets
7. **Largest Datasets**: Analysis of the biggest and most comprehensive datasets
8. **Recommendations**: Suggested focus areas for future research

Format as a well-structured markdown report."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior research analyst creating comprehensive reports on scientific dataset collections."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=6000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            console.print(f"[red]Error generating final summary: {e}")
            return self._create_fallback_summary(consolidated_results)
            
    def _create_fallback_summary(self, consolidated_results: Dict[str, Any]) -> str:
        """Create a basic summary when LLM fails"""
        summary = consolidated_results.get('summary', {})
        
        # Extract keywords
        all_keywords = Counter()
        for dataset in consolidated_results.get('datasets', []):
            all_keywords.update(dataset.get('keywords', []))
            
        return f"""# Dataset Analysis Report

## Summary
- Total datasets analyzed: {consolidated_results.get('total_datasets', 0)}
- Domain distribution: {summary.get('domain_distribution', {})}
- Data type distribution: {summary.get('data_type_distribution', {})}
- Scale distribution: {summary.get('scale_distribution', {})}

## Top Keywords
{', '.join(list(all_keywords.keys())[:50]) if all_keywords else 'Not available'}

## Key Insights
{chr(10).join(['- ' + insight for insight in summary.get('key_insights', [])[:10]])}

*Note: Detailed analysis could not be completed. Manual review recommended.*
"""

    def save_results(self, consolidated_results: Dict[str, Any], summary: str, output_file: str):
        """Save results to files"""
        console.print(f"[blue]Saving results to {output_file}...")
        
        # Clean summary to remove markdown code block wrappers
        cleaned_summary = summary
        if cleaned_summary.startswith('```markdown\n'):
            cleaned_summary = cleaned_summary[12:]  # Remove ```markdown\n
        if cleaned_summary.endswith('\n```'):
            cleaned_summary = cleaned_summary[:-4]  # Remove \n```
        elif cleaned_summary.endswith('```'):
            cleaned_summary = cleaned_summary[:-3]  # Remove ```
        
        # Save markdown report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_summary)
            
        # Save detailed JSON results
        json_file = output_file.replace('.md', '_detailed.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_results, f, indent=2, ensure_ascii=False)
            
        console.print(f"[green]Results saved to:")
        console.print(f"  - Summary report: {output_file}")
        console.print(f"  - Detailed data: {json_file}")
        
    def cleanup_debug_files(self):
        """Clean up debug files created during processing"""
        import glob
        debug_files = glob.glob("debug_chunk_*_response.txt")
        for file in debug_files:
            try:
                os.remove(file)
            except Exception:
                pass
        if debug_files:
            console.print(f"[blue]Cleaned up {len(debug_files)} debug files")


def main():
    parser = argparse.ArgumentParser(description='Analyze datasets using LLM with parallel processing')
    parser.add_argument('csv_file', nargs='?', default='datasets_summary.csv', 
                       help='CSV file to analyze (default: datasets_summary.csv)')
    parser.add_argument('--model', default='gpt-4.1', 
                       help='Model shortname from config (default: gpt-4.1)')
    parser.add_argument('--config', default=DEFAULT_MODEL_CONFIG_FILE,
                       help=f'Model configuration file (default: {DEFAULT_MODEL_CONFIG_FILE})')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                       help=f'Number of datasets per chunk (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                       help=f'Number of parallel workers (default: {DEFAULT_WORKERS})')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_FILE,
                       help=f'Output file (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--keep-debug', action='store_true',
                       help='Keep debug files for failed JSON parsing (default: cleanup)')
    
    args = parser.parse_args()
    
    # Initialize analyzer first to get model info
    analyzer = DatasetAnalyzer(args.model, args.config)
    
    # Display startup info with detailed model information
    console.print(Panel.fit(
        "[bold blue]Dataset Analysis Tool - Parallel Processing[/bold blue]\n"
        f"CSV File: {args.csv_file}\n"
        f"Model: {analyzer.get_model_display_name()}\n"
        f"Chunk Size: {args.chunk_size}\n"
        f"Workers: {args.workers}\n"
        f"Output: {args.output}",
        title="Configuration"
    ))
    
    # Load data
    data = analyzer.load_csv_data(args.csv_file)
    
    # Split into chunks
    chunks = analyzer.chunk_data(data, args.chunk_size)
    console.print(f"[blue]Processing {len(chunks)} chunks with {args.workers} workers...")
    
    # Process chunks in parallel
    chunk_results = analyzer.process_chunks_parallel(chunks, args.workers)
    
    # Consolidate results using divide-and-conquer approach
    consolidated_results = analyzer.consolidate_summaries(chunk_results)
    
    # Generate final summary
    summary = analyzer.generate_final_summary(consolidated_results)
    
    # Save results
    analyzer.save_results(consolidated_results, summary, args.output)
    
    # Display summary statistics
    stats_table = Table(title="Analysis Summary")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    
    summary_data = consolidated_results.get('summary', {})
    
    stats_table.add_row("Total Datasets", str(consolidated_results.get('total_datasets', 0)))
    
    domain_dist = summary_data.get('domain_distribution', {})
    if domain_dist:
        top_domain = max(domain_dist, key=domain_dist.get)
        stats_table.add_row("Top Domain", f"{top_domain} ({domain_dist[top_domain]})")
    
    data_type_dist = summary_data.get('data_type_distribution', {})
    if data_type_dist:
        top_data_type = max(data_type_dist, key=data_type_dist.get)
        stats_table.add_row("Top Data Type", f"{top_data_type} ({data_type_dist[top_data_type]})")
    
    scale_dist = summary_data.get('scale_distribution', {})
    if scale_dist:
        top_scale = max(scale_dist, key=scale_dist.get)
        stats_table.add_row("Most Common Scale", f"{top_scale} ({scale_dist[top_scale]})")
    
    console.print(stats_table)
    
    # Cleanup debug files unless requested to keep them
    if not args.keep_debug:
        analyzer.cleanup_debug_files()
    
    console.print(f"\n[green]Analysis complete! Check {args.output} for the full report.")


if __name__ == "__main__":
    main()