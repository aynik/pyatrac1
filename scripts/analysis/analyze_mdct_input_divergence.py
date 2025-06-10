#!/usr/bin/env python3
"""
Analyze MDCT_INPUT divergences between PyATRAC1 and atracdenc.
Extract exact numerical values and show specific differences.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional

class MDCTInputAnalyzer:
    """Analyzes MDCT_INPUT entries from both implementations."""
    
    def __init__(self):
        # Pattern for PyATRAC1 MDCT_INPUT entries 
        self.pytrac_pattern = re.compile(
            r'\[([^\]]+)\]\[PYTRAC\]\[([^\]]+)\]\[([^\]]+)\]\[CH(\d+)\]\[FR(\d+)\]\[BAND_([^\]]+)\]\s*MDCT_INPUT_([^:]+):\s*samples=\[([^\]]+)\]'
        )
        
        # Pattern for atracdenc MDCT_INPUT entries
        self.atracdenc_pattern = re.compile(
            r'\[([^\]]+)\]\[ATRACDENC\]\[([^\]]+)\]\[([^\]]+)\]\[CH(\d+)\]\[FR(\d+)\]\[BAND_([^\]]+)\]\s*MDCT_INPUT:\s*samples=\[([^\]]+)\]'
        )
    
    def parse_values(self, values_str: str) -> List[float]:
        """Parse values from the truncated array format."""
        # Handle format like "0.000000,0.000000,...2.104861,2.088111,1.940622"
        if '...' in values_str:
            parts = values_str.split('...')
            if len(parts) == 2:
                left_vals = [float(x.strip()) for x in parts[0].split(',') if x.strip()]
                right_vals = [float(x.strip()) for x in parts[1].split(',') if x.strip()]
                return left_vals + right_vals
        else:
            return [float(x.strip()) for x in values_str.split(',') if x.strip()]
        return []
    
    def parse_pytrac_entries(self, filename: str) -> Dict[str, List[float]]:
        """Parse PyATRAC1 MDCT_INPUT entries."""
        entries = {}
        
        with open(filename, 'r') as f:
            for line in f:
                match = self.pytrac_pattern.search(line)
                if match:
                    timestamp, file_line, function, channel, frame, band, stage, values_str = match.groups()
                    key = f"CH{channel}_FR{int(frame):03d}_{band}"
                    values = self.parse_values(values_str)
                    entries[key] = values
        
        return entries
    
    def parse_atracdenc_entries(self, filename: str) -> Dict[str, List[float]]:
        """Parse atracdenc MDCT_INPUT entries."""
        entries = {}
        
        with open(filename, 'r') as f:
            for line in f:
                match = self.atracdenc_pattern.search(line)
                if match:
                    timestamp, file_line, function, channel, frame, band, values_str = match.groups()
                    key = f"CH{channel}_FR{int(frame):03d}_{band}"
                    values = self.parse_values(values_str)
                    entries[key] = values
        
        return entries
    
    def compare_entries(self, pytrac_entries: Dict[str, List[float]], 
                       atracdenc_entries: Dict[str, List[float]]) -> None:
        """Compare MDCT_INPUT entries and show detailed differences."""
        
        print("=== MDCT_INPUT DIVERGENCE ANALYSIS ===")
        print(f"PyATRAC1 entries: {len(pytrac_entries)}")
        print(f"atracdenc entries: {len(atracdenc_entries)}")
        print()
        
        common_keys = set(pytrac_entries.keys()) & set(atracdenc_entries.keys())
        print(f"Common keys: {len(common_keys)}")
        
        if not common_keys:
            print("❌ No common keys found!")
            print("PyATRAC1 keys:", list(pytrac_entries.keys())[:5])
            print("atracdenc keys:", list(atracdenc_entries.keys())[:5])
            return
        
        print()
        
        # Sort keys logically
        sorted_keys = sorted(common_keys, key=lambda k: (
            int(k.split('_')[1][2:]),  # Frame number
            int(k.split('_')[0][2:]),  # Channel number  
            k.split('_')[2]            # Band
        ))
        
        total_divergences = 0
        
        for key in sorted_keys:
            pytrac_vals = pytrac_entries[key]
            atracdenc_vals = atracdenc_entries[key]
            
            print(f"--- {key} ---")
            
            if len(pytrac_vals) != len(atracdenc_vals):
                print(f"❌ Length mismatch: PyATRAC1={len(pytrac_vals)}, atracdenc={len(atracdenc_vals)}")
                print()
                continue
            
            if not pytrac_vals:
                print("⚠️  Both arrays are empty")
                print()
                continue
            
            # Calculate differences
            arr1 = np.array(pytrac_vals)
            arr2 = np.array(atracdenc_vals)
            abs_diff = np.abs(arr1 - arr2)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            
            is_match = np.all(abs_diff <= 1e-6)
            
            print(f"Length: {len(pytrac_vals)} samples")
            print(f"Max difference: {max_diff:.2e}")
            print(f"Mean difference: {mean_diff:.2e}")
            print(f"Match (tol=1e-6): {'✅' if is_match else '❌'}")
            
            if not is_match:
                total_divergences += 1
                
                # Show first few values for comparison
                print(f"PyATRAC1 first 10: {pytrac_vals[:10]}")
                print(f"atracdenc first 10: {atracdenc_vals[:10]}")
                
                # Show worst differences
                worst_indices = np.argsort(abs_diff)[-5:]  # Top 5 worst
                print(f"Worst differences at indices {worst_indices}:")
                for idx in worst_indices:
                    if idx < len(arr1):
                        print(f"  [{idx:3d}]: PyATRAC1={arr1[idx]:10.6f}, atracdenc={arr2[idx]:10.6f}, diff={abs_diff[idx]:10.6f}")
                
                # Show systematic pattern if it exists
                if len(pytrac_vals) >= 20:
                    print(f"PyATRAC1 last 10:  {pytrac_vals[-10:]}")
                    print(f"atracdenc last 10:  {atracdenc_vals[-10:]}")
            
            print()
        
        print(f"=== SUMMARY ===")
        print(f"Total entries analyzed: {len(sorted_keys)}")
        print(f"Divergent entries: {total_divergences}")
        print(f"Perfect matches: {len(sorted_keys) - total_divergences}")

def main():
    analyzer = MDCTInputAnalyzer()
    
    print("🔍 Parsing MDCT_INPUT entries...")
    pytrac_entries = analyzer.parse_pytrac_entries('pytrac_debug.log')
    atracdenc_entries = analyzer.parse_atracdenc_entries('atracdenc_debug.log')
    
    analyzer.compare_entries(pytrac_entries, atracdenc_entries)

if __name__ == "__main__":
    main()