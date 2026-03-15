#!/usr/bin/env python3
"""Analyze data directory contents."""

import os
import json
from pathlib import Path
from collections import defaultdict

def get_file_info(folder_path):
    """Get file count, total size, and extension counts for a folder."""
    file_count = 0
    total_size = 0
    extension_counts = defaultdict(int)
    structured_files = []
    
    structured_exts = {'.csv', '.tsv', '.json', '.xml', '.musicxml', '.mxl', '.mei'}
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_count += 1
            file_path = Path(root) / file
            try:
                total_size += file_path.stat().st_size
            except OSError:
                pass
            
            ext = file_path.suffix.lower()
            extension_counts[ext] += 1
            
            if ext in structured_exts and len(structured_files) < 3:
                structured_files.append(str(file_path))
    
    return file_count, total_size, dict(extension_counts), structured_files

def format_size(bytes_size):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"

def get_file_header(file_path):
    """Get header/schema from structured files."""
    ext = Path(file_path).suffix.lower()
    
    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return f"Keys: {list(data.keys())[:5]}"
                elif isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict):
                        return f"Keys: {list(data[0].keys())[:5]}"
                return "List of items"
        
        elif ext in ['.csv', '.tsv']:
            delimiter = '\t' if ext == '.tsv' else ','
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip()
                return f"Cols: {header[:80]}"
        
        elif ext == '.xml':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
                preview = (first_line + " " + second_line)[:80]
                return f"Root: {preview}"
        
        elif ext == '.mxl':
            return "ZIP archive (music XML)"
        
        elif ext == '.musicxml':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                return f"Header: {first_line[:80]}"
        
        elif ext == '.mei':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                return f"Header: {first_line[:80]}"
        
        return "N/A"
    
    except Exception as e:
        return f"Error: {str(e)[:40]}"

# Main analysis
base_path = Path(r"C:\Users\clq\Documents\GitHub\omr\data")
folders = sorted([d for d in base_path.iterdir() if d.is_dir()])

print("=" * 100)
print("DATA DIRECTORY ANALYSIS")
print("=" * 100)

for folder in folders:
    folder_name = folder.name
    file_count, total_size, ext_counts, struct_files = get_file_info(str(folder))
    
    print(f"\n📁 {folder_name}")
    print(f"   Files: {file_count:,} | Size: {format_size(total_size)}")
    
    if ext_counts:
        ext_str = ", ".join([f"{ext or 'no-ext'}: {count}" for ext, count in sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:5]])
        print(f"   Extensions: {ext_str}")
    
    if struct_files:
        print(f"   Structured files ({len(struct_files)}):")
        for file_path in struct_files:
            header = get_file_header(file_path)
            filename = Path(file_path).name
            print(f"      - {filename}: {header}")

print("\n" + "=" * 100)
