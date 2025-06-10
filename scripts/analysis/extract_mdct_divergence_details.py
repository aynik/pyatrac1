#!/usr/bin/env python3
"""
Extract detailed numerical values for the specific MDCT_INPUT divergences.
Focus on the plain "MDCT_INPUT" stage that shows large differences.
"""

import re
import numpy as np
from typing import Dict, List, Tuple

def extract_mdct_input_entries(log_file: str, implementation: str) -> Dict[str, List[float]]:
    """Extract MDCT_INPUT entries (not MDCT_INPUT_LOW/MID/HIGH)."""
    entries = {}
    
    # Pattern for plain MDCT_INPUT entries (not the _LOW/_MID/_HIGH variants)
    if implementation == "PYTRAC":
        pattern = re.compile(
            r'\[([^\]]+)\]\[PYTRAC\]\[([^\]]+)\]\[([^\]]+)\]\[CH(\d+)\]\[FR(\d+)\]\[BAND_([^\]]+)\]\s*MDCT_INPUT:\s*samples=\[([^\]]+)\]'
        )
    else:  # ATRACDENC
        pattern = re.compile(
            r'\[([^\]]+)\]\[ATRACDENC\]\[([^\]]+)\]\[([^\]]+)\]\[CH(\d+)\]\[FR(\d+)\]\[BAND_([^\]]+)\]\s*MDCT_INPUT:\s*samples=\[([^\]]+)\]'
        )
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                timestamp, file_line, function, channel, frame, band, values_str = match.groups()
                key = f"CH{channel}_FR{int(frame):03d}_{band}"
                
                # Parse values from truncated format
                values = parse_truncated_values(values_str)
                entries[key] = {
                    'values': values,
                    'line': line.strip(),
                    'function': function,
                    'file_line': file_line
                }
    
    return entries

def parse_truncated_values(values_str: str) -> List[float]:
    """Parse values from truncated array format like '1.0,2.0,...,-1.0,-2.0'."""
    if '...' in values_str:
        parts = values_str.split('...')
        if len(parts) == 2:
            left_vals = [float(x.strip()) for x in parts[0].split(',') if x.strip()]
            right_vals = [float(x.strip()) for x in parts[1].split(',') if x.strip()]
            return left_vals + right_vals
    else:
        return [float(x.strip()) for x in values_str.split(',') if x.strip()]
    return []

def main():
    print("🔍 Extracting detailed MDCT_INPUT divergence information...")
    print("="*60)
    
    # Extract entries from both logs
    pytrac_entries = extract_mdct_input_entries('pytrac_debug.log', 'PYTRAC')
    atracdenc_entries = extract_mdct_input_entries('atracdenc_debug.log', 'ATRACDENC')
    
    print(f"PyATRAC1 MDCT_INPUT entries: {len(pytrac_entries)}")
    print(f"atracdenc MDCT_INPUT entries: {len(atracdenc_entries)}")
    print()
    
    # Find common keys
    common_keys = set(pytrac_entries.keys()) & set(atracdenc_entries.keys())
    print(f"Common keys: {len(common_keys)}")
    
    if not common_keys:
        print("❌ No common keys found!")
        print("PyATRAC1 keys:", list(pytrac_entries.keys()))
        print("atracdenc keys:", list(atracdenc_entries.keys()))
        return
    
    # Sort by frame and band
    sorted_keys = sorted(common_keys, key=lambda k: (
        int(k.split('_')[1][2:]),  # Frame
        k.split('_')[2]            # Band
    ))
    
    for key in sorted_keys:
        pytrac_data = pytrac_entries[key]
        atracdenc_data = atracdenc_entries[key]
        
        pytrac_vals = pytrac_data['values']
        atracdenc_vals = atracdenc_data['values']
        
        print(f"=== {key} ===")
        print(f"PyATRAC1 function: {pytrac_data['function']} ({pytrac_data['file_line']})")
        print(f"atracdenc function: {atracdenc_data['function']} ({atracdenc_data['file_line']})")
        
        if len(pytrac_vals) != len(atracdenc_vals):
            print(f"❌ Length mismatch: PyATRAC1={len(pytrac_vals)}, atracdenc={len(atracdenc_vals)}")
        else:
            # Calculate differences 
            arr1 = np.array(pytrac_vals)
            arr2 = np.array(atracdenc_vals)
            abs_diff = np.abs(arr1 - arr2)
            max_diff = np.max(abs_diff)
            
            print(f"Array length: {len(pytrac_vals)}")
            print(f"Max difference: {max_diff:.6f}")
            
            # Show all values side by side
            print("Index | PyATRAC1      | atracdenc     | Difference")
            print("------|---------------|---------------|-------------")
            for i in range(len(pytrac_vals)):
                diff = abs_diff[i]
                marker = " ⚠️ " if diff > 1e-3 else "   "
                print(f"{i:5d} | {arr1[i]:13.6f} | {arr2[i]:13.6f} | {diff:11.6f}{marker}")
            
            # Show which indices have the largest differences
            if max_diff > 1e-6:
                worst_indices = np.argsort(abs_diff)[-5:]
                print(f"\\nWorst differences at indices: {worst_indices}")
                for idx in worst_indices:
                    if abs_diff[idx] > 1e-6:
                        print(f"  [{idx:2d}]: PyATRAC1={arr1[idx]:10.6f}, atracdenc={arr2[idx]:10.6f}, diff={abs_diff[idx]:10.6f}")
        
        print()
    
    print("=== SUMMARY ===")
    print("The large differences in MDCT_INPUT are due to different input buffers")
    print("being passed to the MDCT function in PyATRAC1 vs atracdenc.")
    print("This suggests the issue is in the QMF output or buffer preparation stage,")
    print("not in the MDCT algorithm itself.")

if __name__ == "__main__":
    main()