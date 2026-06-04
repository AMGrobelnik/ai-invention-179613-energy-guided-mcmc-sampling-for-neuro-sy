#!/usr/bin/env python3
"""Fix syntax errors in method.py"""
from pathlib import Path

file_path = Path("method.py")
content = file_path.read_text()

# Fix missing comma in dictionary
old = "'empty_fragments': 0, 'llm_refusals': 0"
new = "'empty_fragments': 0, 'llm_refusals': 0"
content = content.replace(old, new)

file_path.write_text(content)
print("Fixed syntax error")
