#!/usr/bin/env python3
"""
Convert RDF knowledge graph to DOT format for Graphviz visualization
"""

import re
import sys

def parse_rdf_to_dot(rdf_file):
    """Parse RDF file and convert to DOT format"""
    
    dot_content = """digraph KnowledgeGraph {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor=lightblue];
    edge [color=darkblue, fontsize=10];
    
    // Graph attributes
    graph [fontsize=12, fontname="Arial"];
    node [fontsize=10, fontname="Arial"];
    edge [fontsize=8, fontname="Arial"];
    
"""
    
    # Read the RDF file
    with open(rdf_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract triples using regex
    triple_pattern = r':entity_triple_(\d+)'
    subject_pattern = r':entity_-_subject :hasProperty "\*\* ([^"]+)"'
    predicate_pattern = r':entity_-_predicate :hasProperty "\*\* ([^"]+)"'
    object_pattern = r':entity_-_object :hasProperty "\*\* ([^"]+)"'
    
    # Find all triples
    triples = re.findall(triple_pattern, content)
    
    # Process each triple
    for triple_num in triples:
        # Find the section for this triple
        section_start = content.find(f':entity_triple_{triple_num}')
        if triple_num != triples[-1]:
            next_triple = int(triple_num) + 1
            section_end = content.find(f':entity_triple_{next_triple}')
        else:
            section_end = len(content)
        
        section = content[section_start:section_end]
        
        # Extract subject, predicate, object
        subject_match = re.search(subject_pattern, section)
        predicate_match = re.search(predicate_pattern, section)
        object_match = re.search(object_pattern, section)
        
        if subject_match and predicate_match and object_match:
            subject = subject_match.group(1).split(' (Type:')[0]
            predicate = predicate_match.group(1)
            obj = object_match.group(1).split(' (Type:')[0]
            
            # Clean up names for DOT format
            subject_id = f"s_{triple_num}"
            object_id = f"o_{triple_num}"
            
            # Add nodes
            dot_content += f'    {subject_id} [label="{subject}"];\n'
            dot_content += f'    {object_id} [label="{obj}"];\n'
            
            # Add edge
            dot_content += f'    {subject_id} -> {object_id} [label="{predicate}"];\n'
            dot_content += '\n'
    
    dot_content += "}\n"
    return dot_content

def main():
    if len(sys.argv) != 2:
        print("Usage: python rdf_to_dot.py <rdf_file>")
        sys.exit(1)
    
    rdf_file = sys.argv[1]
    dot_content = parse_rdf_to_dot(rdf_file)
    
    # Write DOT file
    dot_file = rdf_file.replace('.rdf', '.dot')
    with open(dot_file, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    
    print(f"DOT file created: {dot_file}")

if __name__ == "__main__":
    main()