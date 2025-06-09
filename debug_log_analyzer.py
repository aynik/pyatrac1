#!/usr/bin/env python3
"""
Debug log analyzer for comparing PyATRAC1 and atracdenc signal processing.
Identifies the first point where the implementations diverge significantly.
"""

import re
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LogEntry:
    """Represents a parsed log entry."""
    timestamp: str
    impl: str  # PYTRAC or ATRACDENC
    file_line: str
    function: str
    channel: int
    frame: int
    band: str
    stage: str
    data_type: str
    values: List[float]
    metadata: Dict[str, str]
    source_info: str

class LogParser:
    """Parses debug log entries from both PyATRAC1 and atracdenc."""
    
    def __init__(self):
        # Regex pattern to parse log entries
        self.log_pattern = re.compile(
            r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\[CH(\d+)\]\[FR(\d+)\](?:\[BAND_([^\]]+)\])?\s*([^:]+):\s*([^=]+)=([^\|]+)\s*\|META:\s*([^\|]+)\s*\|SRC:\s*(.+)'
        )
    
    def parse_values(self, values_str: str) -> List[float]:
        """Parse the values array from log entry."""
        values_str = values_str.strip()
        if values_str.startswith('[') and values_str.endswith(']'):
            values_str = values_str[1:-1]  # Remove brackets
            
            # Handle truncated arrays with '...'
            if '...' in values_str:
                parts = values_str.split('...')
                if len(parts) == 2:
                    left_vals = [float(x.strip()) for x in parts[0].split(',') if x.strip()]
                    right_vals = [float(x.strip()) for x in parts[1].split(',') if x.strip()]
                    return left_vals + right_vals
            else:
                return [float(x.strip()) for x in values_str.split(',') if x.strip()]
        else:
            # Single value
            try:
                return [float(values_str)]
            except ValueError:
                return []
    
    def parse_metadata(self, meta_str: str) -> Dict[str, str]:
        """Parse metadata from log entry."""
        metadata = {}
        for item in meta_str.split():
            if '=' in item:
                key, value = item.split('=', 1)
                metadata[key] = value
        return metadata
    
    def parse_entry(self, line: str) -> Optional[LogEntry]:
        """Parse a single log entry."""
        match = self.log_pattern.match(line.strip())
        if not match:
            return None
        
        timestamp, impl, file_line, function, channel, frame, band, stage, data_type, values, metadata, source = match.groups()
        
        return LogEntry(
            timestamp=timestamp,
            impl=impl,
            file_line=file_line,
            function=function,
            channel=int(channel),
            frame=int(frame),
            band=band or "",
            stage=stage.strip(),
            data_type=data_type.strip(),
            values=self.parse_values(values),
            metadata=self.parse_metadata(metadata),
            source_info=source.strip()
        )
    
    def parse_file(self, filename: str) -> List[LogEntry]:
        """Parse all log entries from a file."""
        entries = []
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue
                    
                entry = self.parse_entry(line)
                if entry:
                    entries.append(entry)
                elif line.strip():  # Only warn for non-empty lines
                    print(f"Warning: Could not parse line {line_num} in {filename}: {line.strip()[:100]}...")
        
        return entries

class LogComparer:
    """Compares log entries between PyATRAC1 and atracdenc."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def make_key(self, entry: LogEntry) -> str:
        """Create a unique key for matching entries between implementations."""
        return f"{entry.stage}|CH{entry.channel}|FR{entry.frame}|{entry.band}|{entry.data_type}"
    
    def compare_values(self, values1: List[float], values2: List[float]) -> Tuple[bool, float, str]:
        """Compare two value arrays and return (is_match, max_diff, details)."""
        if len(values1) != len(values2):
            return False, float('inf'), f"Length mismatch: {len(values1)} vs {len(values2)}"
        
        if not values1:  # Both empty
            return True, 0.0, "Both empty"
        
        # Convert to numpy for easier comparison
        arr1 = np.array(values1)
        arr2 = np.array(values2)
        
        # Calculate absolute differences
        abs_diff = np.abs(arr1 - arr2)
        max_diff = np.max(abs_diff)
        
        # Check if all differences are within tolerance
        is_match = np.all(abs_diff <= self.tolerance)
        
        if not is_match:
            # Find indices of largest differences
            worst_indices = np.argsort(abs_diff)[-3:]  # Top 3 worst
            details = f"Max diff: {max_diff:.2e}, worst at indices: {worst_indices}"
        else:
            details = f"Match within tolerance {self.tolerance:.2e}"
        
        return is_match, max_diff, details
    
    def find_first_divergence(self, pytrac_entries: List[LogEntry], 
                            atracdenc_entries: List[LogEntry]) -> Optional[Tuple[LogEntry, LogEntry, str]]:
        """Find the first point where implementations diverge significantly."""
        
        # Group entries by key
        pytrac_by_key = {self.make_key(entry): entry for entry in pytrac_entries}
        atracdenc_by_key = {self.make_key(entry): entry for entry in atracdenc_entries}
        
        # Find common keys and sort them logically
        common_keys = set(pytrac_by_key.keys()) & set(atracdenc_by_key.keys())
        sorted_keys = sorted(common_keys, key=lambda k: (
            int(k.split('|')[2][2:]),  # Frame number
            int(k.split('|')[1][2:]),  # Channel number
            k.split('|')[0],           # Stage name
            k.split('|')[3]            # Band
        ))
        
        for key in sorted_keys:
            pytrac_entry = pytrac_by_key[key]
            atracdenc_entry = atracdenc_by_key[key]
            
            is_match, max_diff, details = self.compare_values(
                pytrac_entry.values, atracdenc_entry.values
            )
            
            if not is_match:
                return pytrac_entry, atracdenc_entry, details
        
        return None
    
    def compare_stage(self, stage: str, pytrac_entries: List[LogEntry], 
                     atracdenc_entries: List[LogEntry]) -> Dict:
        """Compare a specific stage between implementations."""
        
        # Filter entries for this stage
        pytrac_stage = [e for e in pytrac_entries if e.stage == stage]
        atracdenc_stage = [e for e in atracdenc_entries if e.stage == stage]
        
        # Group by key
        pytrac_by_key = {self.make_key(entry): entry for entry in pytrac_stage}
        atracdenc_by_key = {self.make_key(entry): entry for entry in atracdenc_stage}
        
        results = {
            'stage': stage,
            'pytrac_entries': len(pytrac_stage),
            'atracdenc_entries': len(atracdenc_stage),
            'common_keys': 0,
            'matches': 0,
            'mismatches': 0,
            'max_difference': 0.0,
            'mismatched_keys': []
        }
        
        common_keys = set(pytrac_by_key.keys()) & set(atracdenc_by_key.keys())
        results['common_keys'] = len(common_keys)
        
        for key in common_keys:
            pytrac_entry = pytrac_by_key[key]
            atracdenc_entry = atracdenc_by_key[key]
            
            is_match, max_diff, details = self.compare_values(
                pytrac_entry.values, atracdenc_entry.values
            )
            
            results['max_difference'] = max(results['max_difference'], max_diff)
            
            if is_match:
                results['matches'] += 1
            else:
                results['mismatches'] += 1
                results['mismatched_keys'].append((key, max_diff, details))
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare PyATRAC1 and atracdenc debug logs')
    parser.add_argument('--pytrac-log', default='pytrac_debug.log', 
                       help='PyATRAC1 debug log file')
    parser.add_argument('--atracdenc-log', default='atracdenc_debug.log',
                       help='atracdenc debug log file') 
    parser.add_argument('--tolerance', type=float, default=1e-6,
                       help='Tolerance for numerical comparison')
    parser.add_argument('--stage', type=str,
                       help='Compare specific stage only')
    parser.add_argument('--find-first-divergence', action='store_true',
                       help='Find first point where implementations diverge')
    
    args = parser.parse_args()
    
    print("🔍 Debug Log Analyzer")
    print("="*50)
    
    # Parse log files
    parser_obj = LogParser()
    
    print(f"📂 Parsing {args.pytrac_log}...")
    pytrac_entries = parser_obj.parse_file(args.pytrac_log)
    print(f"   Found {len(pytrac_entries)} entries")
    
    print(f"📂 Parsing {args.atracdenc_log}...")
    atracdenc_entries = parser_obj.parse_file(args.atracdenc_log)
    print(f"   Found {len(atracdenc_entries)} entries")
    
    if not pytrac_entries or not atracdenc_entries:
        print("❌ No entries found in one or both log files")
        return 1
    
    # Compare logs
    comparer = LogComparer(tolerance=args.tolerance)
    
    if args.find_first_divergence:
        print(f"\\n🔍 Finding first divergence (tolerance: {args.tolerance:.2e})...")
        result = comparer.find_first_divergence(pytrac_entries, atracdenc_entries)
        
        if result:
            pytrac_entry, atracdenc_entry, details = result
            print(f"\\n❌ FIRST DIVERGENCE FOUND:")
            print(f"   Stage: {pytrac_entry.stage}")
            print(f"   Location: CH{pytrac_entry.channel} FR{pytrac_entry.frame:03d} {pytrac_entry.band}")
            print(f"   Data type: {pytrac_entry.data_type}")
            print(f"   Difference: {details}")
            print(f"\\n   PyATRAC1 values: {pytrac_entry.values[:10]}{'...' if len(pytrac_entry.values) > 10 else ''}")
            print(f"   atracdenc values: {atracdenc_entry.values[:10]}{'...' if len(atracdenc_entry.values) > 10 else ''}")
            print(f"\\n   PyATRAC1 source: {pytrac_entry.file_line} in {pytrac_entry.function}")
            print(f"   atracdenc source: {atracdenc_entry.file_line} in {atracdenc_entry.function}")
        else:
            print("✅ No significant divergences found within tolerance!")
    
    if args.stage:
        print(f"\\n📊 Comparing stage: {args.stage}")
        result = comparer.compare_stage(args.stage, pytrac_entries, atracdenc_entries)
        
        print(f"   PyATRAC1 entries: {result['pytrac_entries']}")
        print(f"   atracdenc entries: {result['atracdenc_entries']}")
        print(f"   Common keys: {result['common_keys']}")
        print(f"   Matches: {result['matches']}")
        print(f"   Mismatches: {result['mismatches']}")
        print(f"   Max difference: {result['max_difference']:.2e}")
        
        if result['mismatched_keys']:
            print(f"\\n   Top mismatches:")
            for key, diff, details in sorted(result['mismatched_keys'], key=lambda x: x[1], reverse=True)[:5]:
                print(f"     {key}: {diff:.2e} - {details}")
    
    else:
        # Overall comparison by stage
        print(f"\\n📊 Overall comparison by stage:")
        stages = set(e.stage for e in pytrac_entries + atracdenc_entries)
        
        for stage in sorted(stages):
            result = comparer.compare_stage(stage, pytrac_entries, atracdenc_entries)
            status = "✅" if result['mismatches'] == 0 else "❌"
            print(f"   {status} {stage:<20} {result['matches']:3d}/{result['common_keys']:3d} matches, max_diff: {result['max_difference']:.2e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())