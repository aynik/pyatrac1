#!/usr/bin/env python3
"""
Debug why vector_fmul_window produces tiny values in IMDCT context.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode, vector_fmul_window

def debug_vector_fmul_tiny_values():
    """Debug the tiny values from vector_fmul_window in IMDCT."""
    
    print("=== vector_fmul_window Tiny Values Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Create a simple test case with known values
    low_input = np.ones(128, dtype=np.float32) * 0.5  # DC signal
    
    print("Test input: DC signal (all 0.5)")
    print(f"Input energy: {np.sum(low_input**2):.6f}")
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    # Get low band coefficients
    low_coeffs = specs[:128]
    print(f"MDCT coeffs energy: {np.sum(low_coeffs**2):.6f}")
    print(f"MDCT coeffs[0:4]: {low_coeffs[:4]}")
    
    # Manual IMDCT to trace the exact process
    print(f"\n=== Manual IMDCT Process ===")
    
    # Raw IMDCT
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"Raw IMDCT energy: {np.sum(raw_imdct**2):.6f}")
    
    # Extract middle half (atracdenc approach)
    inv_len = len(raw_imdct)  # 256
    middle_start = inv_len // 4  # 64
    middle_length = inv_len // 2  # 128
    inv_buf = raw_imdct[middle_start:middle_start + middle_length]  # [64:192]
    
    print(f"Extracted inv_buf:")
    print(f"  Length: {len(inv_buf)}")
    print(f"  Energy: {np.sum(inv_buf**2):.6f}")
    print(f"  First 8: {inv_buf[:8]}")
    print(f"  Max absolute: {np.max(np.abs(inv_buf)):.6f}")
    
    # Check if the extraction contains meaningful values
    if np.max(np.abs(inv_buf)) < 1e-6:
        print("  ❌ inv_buf contains only tiny values!")
    else:
        print("  ✅ inv_buf contains meaningful values")
    
    # Now test vector_fmul_window with this data
    print(f"\n=== vector_fmul_window Test ===")
    
    # For first frame, prev_buf should be zeros
    prev_buf = np.zeros(16, dtype=np.float32)
    temp_src1 = inv_buf[:16]  # First 16 samples of extracted buffer
    
    print(f"Inputs to vector_fmul_window:")
    print(f"  prev_buf (zeros): {prev_buf}")
    print(f"  temp_src1: {temp_src1}")
    print(f"  temp_src1 max: {np.max(np.abs(temp_src1)):.6f}")
    print(f"  sine_window[0:4]: {mdct.SINE_WINDOW[:4]}")
    print(f"  sine_window[-4:]: {mdct.SINE_WINDOW[-4:]}")
    
    # Apply vector_fmul_window
    temp_dst = np.zeros(32, dtype=np.float32)
    vector_fmul_window(temp_dst, prev_buf, temp_src1, np.array(mdct.SINE_WINDOW, dtype=np.float32), 16)
    
    print(f"\nvector_fmul_window output:")
    print(f"  temp_dst: {temp_dst}")
    print(f"  temp_dst max: {np.max(np.abs(temp_dst)):.6f}")
    print(f"  temp_dst energy: {np.sum(temp_dst**2):.6f}")
    
    # Since prev_buf is zeros, the result should be:
    # dst[i] = 0 * win[j] - temp_src1[j] * win[i] = -temp_src1[j] * win[i]
    # dst[16+j] = 0 * win[i] + temp_src1[j] * win[j] = temp_src1[j] * win[j]
    
    print(f"\nExpected calculation (since prev_buf is zeros):")
    expected_dst = np.zeros(32, dtype=np.float32)
    for i in range(16):
        j = 15 - i  # j = length - 1 - i
        wi = mdct.SINE_WINDOW[i]
        wj = mdct.SINE_WINDOW[j]
        s1 = temp_src1[j]
        
        expected_dst[i] = -s1 * wi  # 0 * wj - s1 * wi
        expected_dst[16 + j] = s1 * wj  # 0 * wi + s1 * wj
        
        if i < 4:
            print(f"  i={i}, j={j}: s1={s1:.6e}, wi={wi:.4f}, wj={wj:.4f}")
            print(f"    expected_dst[{i}] = -{s1:.6e} * {wi:.4f} = {expected_dst[i]:.6e}")
            print(f"    expected_dst[{16+j}] = {s1:.6e} * {wj:.4f} = {expected_dst[16+j]:.6e}")
    
    print(f"\nExpected vs actual:")
    print(f"  Expected max: {np.max(np.abs(expected_dst)):.6e}")
    print(f"  Actual max: {np.max(np.abs(temp_dst)):.6e}")
    print(f"  Close match: {np.allclose(expected_dst, temp_dst)}")
    
    if np.max(np.abs(expected_dst)) < 1e-6:
        print("  ❌ Expected result is also tiny - issue is in temp_src1!")
    else:
        print("  ✅ Expected result should be meaningful")

def debug_extraction_alternatives():
    """Test different extraction strategies to find meaningful data."""
    
    print(f"\n=== Alternative Extraction Strategies ===")
    
    mdct = Atrac1MDCT()
    
    # Use same test case
    low_input = np.ones(128, dtype=np.float32) * 0.5
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    raw_imdct = mdct.imdct256(low_coeffs)
    
    print(f"Testing different extractions for vector_fmul_window:")
    
    extractions = [
        ("atracdenc [64:80]", raw_imdct[64:80]),     # First 16 of middle half
        ("alt1 [80:96]", raw_imdct[80:96]),          # Next 16
        ("alt2 [96:112]", raw_imdct[96:112]),        # Next 16  
        ("alt3 [128:144]", raw_imdct[128:144]),      # From third quarter
        ("alt4 [0:16]", raw_imdct[0:16]),            # From first quarter
        ("alt5 [48:64]", raw_imdct[48:64]),          # Just before middle half
    ]
    
    for name, extraction in extractions:
        print(f"\n{name}:")
        print(f"  Values: {extraction[:4]} ...")
        print(f"  Max abs: {np.max(np.abs(extraction)):.6e}")
        print(f"  Energy: {np.sum(extraction**2):.6f}")
        
        # Test with vector_fmul_window
        temp_dst = np.zeros(32, dtype=np.float32)
        prev_buf = np.zeros(16, dtype=np.float32)
        vector_fmul_window(temp_dst, prev_buf, extraction, np.array(mdct.SINE_WINDOW, dtype=np.float32), 16)
        
        result_max = np.max(np.abs(temp_dst))
        print(f"  vector_fmul result max: {result_max:.6e}")
        
        if result_max > 1e-3:
            print(f"  ✅ This extraction produces meaningful overlap values!")
        elif result_max > 1e-6:
            print(f"  ⚠️  This extraction produces small but non-trivial values")
        else:
            print(f"  ❌ This extraction produces tiny values")

if __name__ == "__main__":
    debug_vector_fmul_tiny_values()
    debug_extraction_alternatives()