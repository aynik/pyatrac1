#!/usr/bin/env python3
"""
Debug the specific indexing offset causing the +32 shift.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT

def debug_index_calculations():
    """Debug the exact index calculations in IMDCT output generation."""
    
    print("=== Index Calculation Debug ===")
    
    # Current constants for N=256 IMDCT
    N = 256
    n2 = N // 2     # 128
    n4 = N // 4     # 64  
    n34 = 3 * n4    # 192
    n54 = 5 * n4    # 320
    
    print(f"Current constants: N={N}, n2={n2}, n4={n4}, n34={n34}, n54={n54}")
    
    # Test different n34 values to see if one produces better alignment
    n34_candidates = [
        ("Current n34=192", 192),
        ("Shifted n34=160", 160),   # 192 - 32
        ("Alt n34=128", 128),       # n2
        ("Alt n34=224", 224),       # 192 + 32
    ]
    
    print(f"\nTesting different n34 values for k=0 (DC):")
    
    for name, n34_test in n34_candidates:
        # Calculate output indices for k=0
        idx1 = n34_test - 1 - 0  # n34 - 1 - k
        idx2 = n34_test + 0      # n34 + k
        idx3 = n4 + 0            # n4 + k = 64
        idx4 = n4 - 1 - 0        # n4 - 1 - k = 63
        
        # Check if indices fall within atracdenc extraction [64:192]
        in_atracdenc = []
        for idx in [idx1, idx2, idx3, idx4]:
            if 64 <= idx < 192:
                in_atracdenc.append(f"{idx}✓")
            else:
                in_atracdenc.append(f"{idx}❌")
        
        print(f"  {name}: indices {in_atracdenc[0]}, {in_atracdenc[1]}, {in_atracdenc[2]}, {in_atracdenc[3]}")
        
        # Count how many indices are within atracdenc region
        valid_count = sum(1 for idx in [idx1, idx2, idx3, idx4] if 64 <= idx < 192)
        print(f"    {valid_count}/4 indices in atracdenc region [64:192]")

def test_index_offset_hypothesis():
    """Test if a simple index offset fixes the alignment."""
    
    print(f"\n=== Index Offset Hypothesis Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test DC coefficient
    test_coeffs = np.zeros(128, dtype=np.float32)
    test_coeffs[0] = 1.0
    
    # Get current output
    current_output = mdct.imdct256(test_coeffs)
    
    print(f"Current output analysis:")
    print(f"  Peak at: {np.argmax(np.abs(current_output))}")
    print(f"  atracdenc [64:192] max: {np.max(np.abs(current_output[64:192])):.6f}")
    
    # Test what happens if we modify the index calculations
    print(f"\nTesting modified index calculations:")
    
    # Hypothesis: subtract 32 from the n34-based indices
    print(f"  If we subtract 32 from n34-based indices:")
    print(f"    Original: buf[191], buf[192]")
    print(f"    Modified: buf[159], buf[160]")
    print(f"    Both would be in [64:192]: ✓✓")
    
    # Check where 159, 160 would place the energy
    test_region_159_160 = current_output[159:161]
    print(f"    Current values at [159:160]: {test_region_159_160}")

def check_atracdenc_constants():
    """Double-check if our constants match atracdenc exactly."""
    
    print(f"\n=== atracdenc Constants Verification ===")
    
    # From atracdenc source:
    # const size_t n2 = N >> 1;     // N/2
    # const size_t n4 = N >> 2;     // N/4  
    # const size_t n34 = 3 * n4;    // 3*N/4
    # const size_t n54 = 5 * n4;    // 5*N/4
    
    N = 256
    atracdenc_n2 = N >> 1      # 128
    atracdenc_n4 = N >> 2      # 64
    atracdenc_n34 = 3 * atracdenc_n4  # 192
    atracdenc_n54 = 5 * atracdenc_n4  # 320
    
    print(f"atracdenc constants for N={N}:")
    print(f"  n2 = {atracdenc_n2}")
    print(f"  n4 = {atracdenc_n4}")
    print(f"  n34 = {atracdenc_n34}")
    print(f"  n54 = {atracdenc_n54}")
    
    # Our constants should match
    our_n2 = N // 2
    our_n4 = N // 4
    our_n34 = 3 * our_n4
    our_n54 = 5 * our_n4
    
    print(f"\nOur constants for N={N}:")
    print(f"  n2 = {our_n2}")
    print(f"  n4 = {our_n4}")
    print(f"  n34 = {our_n34}")
    print(f"  n54 = {our_n54}")
    
    constants_match = (atracdenc_n2 == our_n2 and atracdenc_n4 == our_n4 and 
                      atracdenc_n34 == our_n34 and atracdenc_n54 == our_n54)
    
    if constants_match:
        print(f"  ✓ Constants match atracdenc")
    else:
        print(f"  ❌ Constants differ from atracdenc")
    
    # The constants are correct, so the issue must be elsewhere

def analyze_extraction_boundary():
    """Analyze exactly what happens at the extraction boundary."""
    
    print(f"\n=== Extraction Boundary Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # DC test
    test_coeffs = np.zeros(128, dtype=np.float32)
    test_coeffs[0] = 1.0
    output = mdct.imdct256(test_coeffs)
    
    # Look at values around the boundary
    boundary_region = output[60:196]  # [60:196] covers [64:192] ± 4
    
    print(f"Values around atracdenc boundary [64:192]:")
    for i, val in enumerate(boundary_region):
        actual_idx = 60 + i
        if actual_idx == 64:
            print(f"  --- atracdenc extraction starts here ---")
        if actual_idx == 192:
            print(f"  --- atracdenc extraction ends here ---")
        
        if abs(val) > 0.1:  # Significant values
            in_region = "✓" if 64 <= actual_idx < 192 else "❌"
            print(f"  [{actual_idx:3d}]: {val:8.6f} {in_region}")
    
    print(f"\nThe problem:")
    print(f"  Large values at 191 (✓ inside) and 192 (❌ outside)")
    print(f"  This creates 50/50 split across boundary")
    print(f"  Need to shift indices so both large values are inside [64:192]")

if __name__ == "__main__":
    debug_index_calculations()
    test_index_offset_hypothesis()
    check_atracdenc_constants()
    analyze_extraction_boundary()
    
    print(f"\n=== CONCLUSION ===")
    print("The +32 shift comes from output indices landing at the boundary.")
    print("Need to identify if this is:")
    print("1. Wrong n34 calculation")
    print("2. Missing offset in index formulas") 
    print("3. Different buffer layout assumption")
    print("4. FFT output interpretation difference")