#!/usr/bin/env python3
"""
Debug script to analyze IMDCT buffer extraction logic and compare with atracdenc.
Focus on the exact indexing and what gets stored where.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyatrac1.core.mdct import *
from pyatrac1.common.debug_logger import debug_logger

def debug_imdct_buffer_extraction():
    """
    Debug the IMDCT buffer extraction logic to understand why middle sections are zeros.
    """
    
    # Create a test IMDCT instance
    mdct = Atrac1MDCT()
    
    # Create test spectral coefficients - use a simple pattern that should produce non-zero output
    test_coeffs = np.zeros(256, dtype=np.float32)
    test_coeffs[0] = 1.0  # DC component
    test_coeffs[1] = 0.5  # First harmonic
    test_coeffs[2] = 0.25 # Second harmonic
    
    print("=== IMDCT Buffer Extraction Debug ===")
    print(f"Input coefficients: {test_coeffs[:10]}")
    
    # Test with IMDCT256 (most common case)
    raw_imdct = mdct.imdct256(test_coeffs)
    print(f"\nRaw IMDCT output length: {len(raw_imdct)}")
    print(f"Raw IMDCT output range: [{np.min(raw_imdct):.6f}, {np.max(raw_imdct):.6f}]")
    print(f"Raw IMDCT first 16: {raw_imdct[:16]}")
    print(f"Raw IMDCT last 16: {raw_imdct[-16:]}")
    
    # atracdenc extraction logic analysis
    inv_len = len(raw_imdct)  # Should be 256
    print(f"\nIMDCT length: {inv_len}")
    
    # Key extraction indices from atracdenc
    # for (size_t i = 0; i < (inv.size()/2); i++) {
    #     invBuf[start+i] = inv[i + inv.size()/4];
    # }
    middle_start = inv_len // 4      # inv.size()/4 = 64
    middle_length = inv_len // 2     # inv.size()/2 = 128
    middle_end = middle_start + middle_length  # 64 + 128 = 192
    
    print(f"Middle extraction:")
    print(f"  - Start index: {middle_start}")
    print(f"  - Length: {middle_length}")
    print(f"  - End index: {middle_end}")
    print(f"  - Range: [{middle_start}:{middle_end}]")
    
    # Extract middle section
    middle_section = raw_imdct[middle_start:middle_end]
    print(f"Middle section range: [{np.min(middle_section):.6f}, {np.max(middle_section):.6f}]")
    print(f"Middle section first 16: {middle_section[:16]}")
    print(f"Middle section non-zero count: {np.count_nonzero(middle_section)}")
    
    # Check other sections for comparison
    first_quarter = raw_imdct[:middle_start]  # [0:64]
    last_quarter = raw_imdct[middle_end:]     # [192:256]
    
    print(f"\nFirst quarter [0:{middle_start}]:")
    print(f"  - Range: [{np.min(first_quarter):.6f}, {np.max(first_quarter):.6f}]")
    print(f"  - Non-zero count: {np.count_nonzero(first_quarter)}")
    print(f"  - First 8: {first_quarter[:8]}")
    
    print(f"\nLast quarter [{middle_end}:{inv_len}]:")
    print(f"  - Range: [{np.min(last_quarter):.6f}, {np.max(last_quarter):.6f}]")
    print(f"  - Non-zero count: {np.count_nonzero(last_quarter)}")
    print(f"  - First 8: {last_quarter[:8]}")
    
    # Test the atracdenc long block copy logic
    # memcpy(dstBuf + 32, &invBuf[16], ((band == 2) ? 240 : 112) * sizeof(float));
    print(f"\n=== Long Block Copy Analysis ===")
    print("For 256-sample band (mid band), copy length = 112")
    
    # Simulate the invBuf as it would be in atracdenc
    inv_buf = np.zeros(256, dtype=np.float32)  # Size matches buffer size
    inv_buf[:middle_length] = middle_section  # Store middle section starting at index 0
    
    print(f"invBuf (simulated) first 16: {inv_buf[:16]}")
    print(f"invBuf (simulated) [16:32]: {inv_buf[16:32]}")
    
    # The long block copy extracts from invBuf[16:16+112] 
    long_copy_start = 16
    long_copy_length = 112
    long_copy_data = inv_buf[long_copy_start:long_copy_start + long_copy_length]
    
    print(f"Long copy data [16:128]:")
    print(f"  - Range: [{np.min(long_copy_data):.6f}, {np.max(long_copy_data):.6f}]")
    print(f"  - Non-zero count: {np.count_nonzero(long_copy_data)}")
    print(f"  - First 8: {long_copy_data[:8]}")
    
    # Check if the issue is in the IMDCT implementation itself
    print(f"\n=== IMDCT Implementation Check ===")
    
    # Test with different input patterns
    dc_only = np.zeros(256, dtype=np.float32)
    dc_only[0] = 1.0
    dc_result = mdct.imdct256(dc_only)
    print(f"DC-only IMDCT middle section: {dc_result[64:80]}")
    
    # Test with impulse at different positions
    impulse_64 = np.zeros(256, dtype=np.float32)
    impulse_64[64] = 1.0
    impulse_result = mdct.imdct256(impulse_64)
    print(f"Impulse@64 IMDCT middle section: {impulse_result[64:80]}")
    
    # Test with simple sine wave coefficients
    sine_coeffs = np.zeros(256, dtype=np.float32)
    for i in range(10):
        sine_coeffs[i] = np.sin(i * 0.1) * 0.5
    sine_result = mdct.imdct256(sine_coeffs)
    print(f"Sine coeffs IMDCT middle section: {sine_result[64:80]}")
    
    print(f"\n=== Summary ===")
    print(f"Middle section extraction: [{middle_start}:{middle_end}] from {inv_len} samples")
    print(f"Long block copy: [16:128] from invBuf (which contains the middle section)")
    print(f"Key insight: If middle_section is zeros, then long_copy_data will also be zeros")
    
    return {
        'raw_imdct': raw_imdct,
        'middle_section': middle_section,
        'long_copy_data': long_copy_data,
        'middle_start': middle_start,
        'middle_length': middle_length
    }

if __name__ == "__main__":
    debug_imdct_buffer_extraction()