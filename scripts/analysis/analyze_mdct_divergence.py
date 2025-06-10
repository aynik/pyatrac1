#!/usr/bin/env python3
"""
Analyze specific MDCT_INPUT divergences between PyATRAC1 and atracdenc.
Extract exact numerical values for detailed comparison.
"""

import re
import numpy as np

def parse_log_line(line):
    """Parse a log line and extract key information."""
    # Pattern for the MDCT_INPUT entries that are being compared
    pattern = r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\[CH(\d+)\]\[FR(\d+)\](?:\[BAND_([^\]]+)\])?\s*([^:]+):\s*([^=]+)=([^\|]+)\s*\|META:\s*([^\|]+)'
    
    match = re.match(pattern, line.strip())
    if not match:
        return None
    
    timestamp, impl, file_line, function, channel, frame, band, stage, data_type, values_str, metadata = match.groups()
    
    # Parse values array
    values_str = values_str.strip()
    if values_str.startswith('[') and values_str.endswith(']'):
        values_str = values_str[1:-1]  # Remove brackets
        
        # Handle truncated arrays with '...'
        if '...' in values_str:
            parts = values_str.split('...')
            if len(parts) == 2:
                left_vals = [float(x.strip()) for x in parts[0].split(',') if x.strip()]
                right_vals = [float(x.strip()) for x in parts[1].split(',') if x.strip()]
                values = left_vals + ['...'] + right_vals
            else:
                values = []
        else:
            values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
    
    return {
        'impl': impl,
        'channel': int(channel),
        'frame': int(frame),
        'band': band or "",
        'stage': stage.strip(),
        'data_type': data_type.strip(),
        'values': values,
        'metadata': metadata,
        'values_str': values_str
    }

def extract_full_array(values_str):
    """Extract the full array from a truncated representation."""
    # Parse metadata to get the full size
    if '...' not in values_str:
        return [float(x.strip()) for x in values_str.split(',') if x.strip()]
    
    # For truncated arrays, we only have the start and end values
    parts = values_str.split('...')
    if len(parts) == 2:
        left_vals = [float(x.strip()) for x in parts[0].split(',') if x.strip()]
        right_vals = [float(x.strip()) for x in parts[1].split(',') if x.strip()]
        return left_vals, right_vals
    return [], []

def main():
    print("🔍 MDCT_INPUT Divergence Analysis")
    print("="*50)
    
    # Read PyATRAC1 log
    pytrac_entries = []
    with open('pytrac_debug.log', 'r') as f:
        for line in f:
            if 'MDCT_INPUT:' in line and '[CH0]' in line:
                entry = parse_log_line(line)
                if entry:
                    pytrac_entries.append(entry)
    
    # Read atracdenc log  
    atracdenc_entries = []
    with open('atracdenc_debug.log', 'r') as f:
        for line in f:
            if 'MDCT_INPUT:' in line and '[CH0]' in line:
                entry = parse_log_line(line)
                if entry:
                    atracdenc_entries.append(entry)
    
    print(f"Found {len(pytrac_entries)} PyATRAC1 MDCT_INPUT entries")
    print(f"Found {len(atracdenc_entries)} atracdenc MDCT_INPUT entries")
    
    # Group by frame and band for comparison
    pytrac_by_key = {}
    for entry in pytrac_entries:
        key = f"FR{entry['frame']:03d}_{entry['band']}"
        pytrac_by_key[key] = entry
    
    atracdenc_by_key = {}
    for entry in atracdenc_entries:
        key = f"FR{entry['frame']:03d}_{entry['band']}"
        atracdenc_by_key[key] = entry
    
    # Find matching entries and compare
    common_keys = set(pytrac_by_key.keys()) & set(atracdenc_by_key.keys())
    print(f"\\nCommon keys: {sorted(common_keys)}")
    
    for key in sorted(common_keys):
        pytrac_entry = pytrac_by_key[key]
        atracdenc_entry = atracdenc_by_key[key]
        
        print(f"\\n📊 Comparing {key}")
        print(f"   PyATRAC1 metadata: {pytrac_entry['metadata']}")
        print(f"   atracdenc metadata: {atracdenc_entry['metadata']}")
        
        # Extract the visible values for comparison
        if '...' in pytrac_entry['values_str']:
            pytrac_left, pytrac_right = extract_full_array(pytrac_entry['values_str'])
            atracdenc_left, atracdenc_right = extract_full_array(atracdenc_entry['values_str'])
            
            print(f"\\n   PyATRAC1 values (first): {pytrac_left}")
            print(f"   atracdenc values (first): {atracdenc_left}")
            
            print(f"\\n   PyATRAC1 values (last):  {pytrac_right}")
            print(f"   atracdenc values (last):  {atracdenc_right}")
            
            # Calculate differences for visible values
            if len(pytrac_left) == len(atracdenc_left):
                left_diffs = [abs(a - b) for a, b in zip(pytrac_left, atracdenc_left)]
                print(f"   Differences (first): {left_diffs}")
                print(f"   Max diff (first): {max(left_diffs):.6e}")
            
            if len(pytrac_right) == len(atracdenc_right):
                right_diffs = [abs(a - b) for a, b in zip(pytrac_right, atracdenc_right)]
                print(f"   Differences (last):  {right_diffs}")
                print(f"   Max diff (last): {max(right_diffs):.6e}")
        else:
            # Full arrays visible
            pytrac_vals = pytrac_entry['values']
            atracdenc_vals = atracdenc_entry['values']
            
            print(f"\\n   PyATRAC1 values: {pytrac_vals}")
            print(f"   atracdenc values: {atracdenc_vals}")
            
            if len(pytrac_vals) == len(atracdenc_vals):
                diffs = [abs(a - b) for a, b in zip(pytrac_vals, atracdenc_vals)]
                print(f"   Differences: {diffs}")
                print(f"   Max diff: {max(diffs):.6e}")

if __name__ == "__main__":
    main()