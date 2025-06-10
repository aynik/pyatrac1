#!/usr/bin/env python3
"""
Focused analysis of MDCT_WINDOWED divergences between PyATRAC1 and atracdenc.
Identifies the pattern of which MDCT windowing operations show divergences.
"""

import re
import numpy as np

def parse_windowed_entry(line):
    """Parse a MDCT_WINDOWED log entry."""
    # Extract key info from log line
    match = re.search(r'\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\[([^\]]+)\]\[CH(\d+)\]\[FR(\d+)\](?:\[BAND_([^\]]+)\])?\s*([^:]+):\s*([^=]+)=([^\|]+)\s*\|META:\s*([^\|]+)', line)
    if not match:
        return None
    
    timestamp, impl, file_line, function, channel, frame, band, stage, data_type, values_str, metadata = match.groups()
    
    # Parse metadata
    meta_dict = {}
    for item in metadata.split():
        if '=' in item:
            key, value = item.split('=', 1)
            meta_dict[key] = value
    
    # Parse values (truncated array)
    values_str = values_str.strip()
    if values_str.startswith('[') and values_str.endswith(']'):
        values_str = values_str[1:-1]  # Remove brackets
        
        if '...' in values_str:
            # Handle truncated arrays - just get first and last few values
            parts = values_str.split('...')
            if len(parts) == 2:
                left_vals = [float(x.strip()) for x in parts[0].split(',') if x.strip()]
                right_vals = [float(x.strip()) for x in parts[1].split(',') if x.strip()]
                values = left_vals + right_vals
            else:
                values = []
        else:
            values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
    else:
        values = []
    
    return {
        'impl': impl,
        'channel': int(channel),
        'frame': int(frame),
        'band': band,
        'stage': stage.strip(),
        'data_type': data_type.strip(),
        'values': values,
        'metadata': meta_dict,
        'size': int(meta_dict.get('size', 0))
    }

def analyze_mdct_windowing():
    """Analyze the MDCT windowing divergences."""
    
    print("🔍 MDCT Windowing Divergence Analysis")
    print("=" * 60)
    
    # Parse PyATRAC1 entries
    pytrac_entries = []
    with open('pytrac_debug.log', 'r') as f:
        for line in f:
            if 'MDCT_WINDOWED' in line and 'samples=' in line:
                entry = parse_windowed_entry(line)
                if entry:
                    pytrac_entries.append(entry)
    
    # Parse atracdenc entries
    atracdenc_entries = []
    with open('atracdenc_debug.log', 'r') as f:
        for line in f:
            if 'MDCT_WINDOWED' in line and 'samples=' in line and 'MDCT_WINDOWED_FINAL' not in line:
                entry = parse_windowed_entry(line)
                if entry:
                    atracdenc_entries.append(entry)
    
    print(f"📂 Found {len(pytrac_entries)} PyATRAC1 MDCT_WINDOWED entries")
    print(f"📂 Found {len(atracdenc_entries)} atracdenc MDCT_WINDOWED entries")
    
    # Group by key for comparison
    pytrac_by_key = {}
    for entry in pytrac_entries:
        key = f"CH{entry['channel']}_FR{entry['frame']:03d}_{entry['band']}"
        pytrac_by_key[key] = entry
    
    atracdenc_by_key = {}
    for entry in atracdenc_entries:
        key = f"CH{entry['channel']}_FR{entry['frame']:03d}_{entry['band']}"
        atracdenc_by_key[key] = entry
    
    # Find matches and divergences
    common_keys = set(pytrac_by_key.keys()) & set(atracdenc_by_key.keys())
    print(f"🔗 Found {len(common_keys)} common windowing operations")
    
    print("\n📊 Windowing Size Analysis:")
    print("-" * 60)
    
    matches = 0
    divergences = []
    
    for key in sorted(common_keys):
        pytrac_entry = pytrac_by_key[key]
        atracdenc_entry = atracdenc_by_key[key]
        
        # Check if array sizes match
        pytrac_size = pytrac_entry['size']
        atracdenc_size = atracdenc_entry['size']
        
        # Check if values match (for truncated arrays, compare what we have)
        max_diff = 0.0
        value_match = True
        
        if len(pytrac_entry['values']) > 0 and len(atracdenc_entry['values']) > 0:
            min_len = min(len(pytrac_entry['values']), len(atracdenc_entry['values']))
            pytrac_vals = np.array(pytrac_entry['values'][:min_len])
            atracdenc_vals = np.array(atracdenc_entry['values'][:min_len])
            
            abs_diff = np.abs(pytrac_vals - atracdenc_vals)
            max_diff = np.max(abs_diff)
            value_match = max_diff < 1e-5  # Tolerance for numerical precision
        
        # Determine overall match
        size_match = pytrac_size == atracdenc_size
        overall_match = size_match and value_match
        
        status = "✅" if overall_match else "❌"
        
        print(f"{status} {key:<20} PyATRAC1: {pytrac_size:3d}, atracdenc: {atracdenc_size:3d}, max_diff: {max_diff:.2e}")
        
        if overall_match:
            matches += 1
        else:
            divergences.append({
                'key': key,
                'pytrac_size': pytrac_size,
                'atracdenc_size': atracdenc_size,
                'max_diff': max_diff,
                'pytrac_entry': pytrac_entry,
                'atracdenc_entry': atracdenc_entry
            })
    
    print(f"\n📈 Summary: {matches}/{len(common_keys)} matches, {len(divergences)} divergences")
    
    if divergences:
        print(f"\n❌ Divergent Cases Analysis:")
        print("-" * 60)
        
        for div in divergences:
            print(f"\n🔍 {div['key']}:")
            print(f"   PyATRAC1 size: {div['pytrac_size']}")
            print(f"   atracdenc size: {div['atracdenc_size']}")
            print(f"   Max difference: {div['max_diff']:.2e}")
            
            # Show first few values for comparison
            pytrac_vals = div['pytrac_entry']['values'][:10]
            atracdenc_vals = div['atracdenc_entry']['values'][:10]
            
            print(f"   PyATRAC1  values: {pytrac_vals}")
            print(f"   atracdenc values: {atracdenc_vals}")
            
            # Check if this is a windowing state issue
            if div['pytrac_size'] != div['atracdenc_size']:
                print(f"   ⚠️  SIZE MISMATCH: Different MDCT transform sizes!")
                if 'FR001' in div['key']:
                    print(f"   💡 Frame 1 windowing state difference detected")
        
        print(f"\n🎯 Root Cause Analysis:")
        print("-" * 40)
        
        # Analyze patterns
        frame_1_divergences = [d for d in divergences if 'FR001' in d['key']]
        other_divergences = [d for d in divergences if 'FR001' not in d['key']]
        
        if frame_1_divergences:
            print(f"📌 Frame 1 has {len(frame_1_divergences)}/3 bands with divergences")
            print(f"   This suggests frame-to-frame windowing state differences")
            print(f"   PyATRAC1 and atracdenc are using different MDCT sizes for frame 1")
            
            for div in frame_1_divergences:
                band = div['key'].split('_')[-1]
                print(f"   {band} band: PyATRAC1={div['pytrac_size']}, atracdenc={div['atracdenc_size']}")
        
        if other_divergences:
            print(f"📌 Other frames have {len(other_divergences)} divergences")
            for div in other_divergences:
                if div['max_diff'] > 1e-3:
                    print(f"   {div['key']}: Large difference {div['max_diff']:.2e}")

if __name__ == "__main__":
    analyze_mdct_windowing()