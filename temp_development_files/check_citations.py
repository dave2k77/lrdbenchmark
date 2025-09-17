#!/usr/bin/env python3
"""
Check which citations are used in the manuscript but missing from references.bib
"""

import re

# Read the manuscript
with open('manuscript.tex', 'r') as f:
    manuscript = f.read()

# Extract all citations
citation_pattern = r'\\cite[a-z]*\{([^}]+)\}'
citations = re.findall(citation_pattern, manuscript)

# Flatten and clean citations
all_citations = []
for citation in citations:
    # Split by comma and clean
    for ref in citation.split(','):
        ref = ref.strip()
        if ref:
            all_citations.append(ref)

# Remove duplicates and sort
used_citations = sorted(set(all_citations))

# Read the bibliography
with open('references.bib', 'r') as f:
    bib_content = f.read()

# Extract bibliography keys
bib_pattern = r'@\w+\{([^,]+),'
bib_keys = re.findall(bib_pattern, bib_content)

# Find missing citations
missing_citations = []
for citation in used_citations:
    if citation not in bib_keys:
        missing_citations.append(citation)

# Find unused citations
unused_citations = []
for key in bib_keys:
    if key not in used_citations:
        unused_citations.append(key)

print("=== CITATION ANALYSIS ===")
print(f"Total citations used in manuscript: {len(used_citations)}")
print(f"Total entries in references.bib: {len(bib_keys)}")
print()

if missing_citations:
    print("=== MISSING CITATIONS ===")
    for citation in missing_citations:
        print(f"  - {citation}")
    print()

if unused_citations:
    print("=== UNUSED CITATIONS (first 20) ===")
    for citation in unused_citations[:20]:
        print(f"  - {citation}")
    if len(unused_citations) > 20:
        print(f"  ... and {len(unused_citations) - 20} more")
    print()

print("=== USED CITATIONS ===")
for citation in used_citations:
    status = "✓" if citation in bib_keys else "✗"
    print(f"  {status} {citation}")
