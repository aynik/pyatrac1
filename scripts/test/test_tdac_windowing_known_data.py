#!/usr/bin/env python3
"""
TDAC windowing test with known data to find min/max values and verify implementation.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_tdac_windowing_known_data():
    """Test TDAC windowing with known data to find implementation issues."""
    
    print("=== TDAC Windowing Test with Known Data ===")
    
    mdct = Atrac1MDCT()
    
    # Use precisely known test data
    print("Test 1: DC signal analysis")
    
    # DC test case - should be perfectly reconstructible
    dc_value = 1.0
    test_input = np.full(128, dc_value, dtype=np.float32)
    
    print(f"Input: DC signal, value={dc_value}")
    print(f"Input dtype: {test_input.dtype}")
    print(f"Input energy: {np.sum(test_input**2):.6f}")
    
    # Forward MDCT 
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"MDCT coefficients:")
    print(f"  DC coeff: {low_coeffs[0]:.6f}")
    print(f"  Max AC coeff: {np.max(np.abs(low_coeffs[1:])):.6f}")
    print(f"  Total energy: {np.sum(low_coeffs**2):.6f}")
    
    # Raw IMDCT analysis
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"\nRaw IMDCT analysis (dtype: {raw_imdct.dtype}):")
    print(f"  Total energy: {np.sum(raw_imdct**2):.6f}")
    print(f"  Min value: {np.min(raw_imdct):.6f}")
    print(f"  Max value: {np.max(raw_imdct):.6f}")
    print(f"  Mean value: {np.mean(raw_imdct):.6f}")
    
    # Analyze each quarter of raw IMDCT
    quarters = [
        raw_imdct[:64],      # Q1 [0:64]
        raw_imdct[64:128],   # Q2 [64:128] - atracdenc extraction start
        raw_imdct[128:192],  # Q3 [128:192]
        raw_imdct[192:256]   # Q4 [192:256]
    ]
    
    for i, quarter in enumerate(quarters):
        print(f"  Q{i+1} [{i*64}:{(i+1)*64}]: min={np.min(quarter):.6f}, max={np.max(quarter):.6f}, mean={np.mean(quarter):.6f}")
    
    # Key insight: For DC, we expect all quarters to have similar structure
    # If Q2 has tiny values but Q3 has meaningful values, our IMDCT is wrong
    
    print(f"\n=== atracdenc Extraction Analysis ===")
    
    # Test exactly what atracdenc extracts
    atracdenc_extraction = raw_imdct[64:192]  # inv[i + inv.size()/4] for i=0..127
    
    print(f"atracdenc extraction [64:192]:")
    print(f"  Energy: {np.sum(atracdenc_extraction**2):.6f}")
    print(f"  Min: {np.min(atracdenc_extraction):.6f}")
    print(f"  Max: {np.max(atracdenc_extraction):.6f}")
    print(f"  Mean: {np.mean(atracdenc_extraction):.6f}")
    print(f"  First 8: {atracdenc_extraction[:8]}")
    print(f"  Last 8: {atracdenc_extraction[-8:]}")
    
    # For DC input, this should contain meaningful reconstruction data
    # If it doesn't, our IMDCT implementation is fundamentally wrong
    
    if np.max(np.abs(atracdenc_extraction)) < 1e-6:
        print("  ❌ CRITICAL: atracdenc extraction contains only noise!")
        print("     This means our IMDCT implementation is wrong")
    else:
        print("  ✅ atracdenc extraction contains meaningful data")
    
    # Test our current extraction for comparison
    our_extraction = raw_imdct[96:224]
    print(f"\nOur extraction [96:224]:")
    print(f"  Energy: {np.sum(our_extraction**2):.6f}")
    print(f"  Min: {np.min(our_extraction):.6f}")
    print(f"  Max: {np.max(our_extraction):.6f}")
    print(f"  Mean: {np.mean(our_extraction):.6f}")
    
    # Complete IMDCT pipeline test
    print(f"\n=== Complete IMDCT Pipeline ===")
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    print(f"Complete pipeline output:")
    print(f"  Total energy: {np.sum(low_out**2):.6f}")
    print(f"  Min: {np.min(low_out):.6f}")
    print(f"  Max: {np.max(low_out):.6f}")
    
    # Critical: middle region analysis
    middle_region = low_out[32:144]  # What should be constant for DC
    print(f"  Middle region mean: {np.mean(middle_region):.6f}")
    print(f"  Middle region std: {np.std(middle_region):.6f}")
    print(f"  Expected for DC: {dc_value * 0.25:.6f} (25% scaling)")
    
    # SNR calculation
    useful_output = low_out[32:160]  # 128 samples
    if len(useful_output) == len(test_input):
        error = useful_output - test_input
        error_energy = np.sum(error**2)
        signal_energy = np.sum(test_input**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
            print(f"  DC reconstruction SNR: {snr_db:.2f} dB")
        else:
            print(f"  Perfect reconstruction")
    
    return raw_imdct, atracdenc_extraction

def debug_imdct_implementation():
    """Debug our IMDCT implementation against theoretical expectations."""
    
    print(f"\n=== IMDCT Implementation Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Test with pure DC coefficient
    print("Test: Pure DC coefficient")
    
    dc_coeffs = np.zeros(128, dtype=np.float32)
    dc_coeffs[0] = 1.0  # Pure DC
    
    print(f"Input coeffs: DC=1.0, all others=0")
    
    # Raw IMDCT of pure DC
    raw_imdct = mdct.imdct256(dc_coeffs)
    
    print(f"Raw IMDCT of pure DC:")
    print(f"  Shape: {raw_imdct.shape}")
    print(f"  Dtype: {raw_imdct.dtype}")
    print(f"  Energy: {np.sum(raw_imdct**2):.6f}")
    print(f"  Min: {np.min(raw_imdct):.6f}")
    print(f"  Max: {np.max(raw_imdct):.6f}")
    
    # For pure DC, IMDCT should produce a specific pattern
    # Check if it's symmetric or has expected structure
    
    print(f"Symmetry analysis:")
    first_half = raw_imdct[:128]
    second_half = raw_imdct[128:]
    
    print(f"  First half mean: {np.mean(first_half):.6f}")
    print(f"  Second half mean: {np.mean(second_half):.6f}")
    print(f"  Symmetric: {np.allclose(first_half, second_half)}")
    print(f"  Anti-symmetric: {np.allclose(first_half, -second_half)}")
    
    # Check quarters
    q1 = raw_imdct[:64]
    q2 = raw_imdct[64:128] 
    q3 = raw_imdct[128:192]
    q4 = raw_imdct[192:256]
    
    print(f"Quarter analysis:")
    print(f"  Q1 mean: {np.mean(q1):.6f}, std: {np.std(q1):.6f}")
    print(f"  Q2 mean: {np.mean(q2):.6f}, std: {np.std(q2):.6f}")
    print(f"  Q3 mean: {np.mean(q3):.6f}, std: {np.std(q3):.6f}")
    print(f"  Q4 mean: {np.mean(q4):.6f}, std: {np.std(q4):.6f}")
    
    # The pattern should tell us where the meaningful reconstruction data is
    
    # Test different frequency components
    print(f"\n=== Frequency Component Tests ===")
    
    freq_tests = [1, 4, 8, 16, 32, 64]  # Different frequency bins
    
    for freq_bin in freq_tests:
        if freq_bin < 128:
            coeffs = np.zeros(128, dtype=np.float32)
            coeffs[freq_bin] = 1.0
            
            raw_imdct = mdct.imdct256(coeffs)
            
            # Find where the peak response is
            peak_idx = np.argmax(np.abs(raw_imdct))
            peak_val = raw_imdct[peak_idx]
            
            print(f"  Freq bin {freq_bin}: peak at idx {peak_idx}, val={peak_val:.6f}")
            
            # Check which quarter contains the peak
            quarter = peak_idx // 64
            print(f"    Peak in quarter {quarter+1}")
    
    # This will tell us if our IMDCT output structure is correct

def test_float32_consistency():
    """Verify all operations maintain float32 consistency."""
    
    print(f"\n=== Float32 Consistency Test ===")
    
    mdct = Atrac1MDCT()
    
    # Check all internal buffer dtypes
    print("Internal buffer dtypes:")
    print(f"  pcm_buf_low[0]: {mdct.pcm_buf_low[0].dtype}")
    print(f"  pcm_buf_mid[0]: {mdct.pcm_buf_mid[0].dtype}")
    print(f"  pcm_buf_hi[0]: {mdct.pcm_buf_hi[0].dtype}")
    print(f"  tmp_buffers[0][0]: {mdct.tmp_buffers[0][0].dtype}")
    
    # Check MDCT/IMDCT engine internal arrays
    print(f"  mdct256.buf: {mdct.mdct256.buf.dtype}")
    print(f"  imdct256.buf: {mdct.imdct256.buf.dtype}")
    print(f"  mdct256.SinCos: {mdct.mdct256.SinCos.dtype}")
    print(f"  imdct256.SinCos: {mdct.imdct256.SinCos.dtype}")
    
    # Test a round trip and check dtypes at each step
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    print(f"After MDCT:")
    print(f"  specs dtype: {specs.dtype}")
    print(f"  pcm_buf_low[0] dtype: {mdct.pcm_buf_low[0].dtype}")
    
    # Check raw IMDCT dtype
    raw_imdct = mdct.imdct256(specs[:128])
    print(f"  raw_imdct dtype: {raw_imdct.dtype}")
    
    # Check final output dtypes
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    print(f"After IMDCT:")
    print(f"  low_out dtype: {low_out.dtype}")
    print(f"  mid_out dtype: {mid_out.dtype}")
    print(f"  hi_out dtype: {hi_out.dtype}")
    
    # Verify no dtype conversions occurred
    all_float32 = all([
        mdct.pcm_buf_low[0].dtype == np.float32,
        specs.dtype == np.float32,
        raw_imdct.dtype == np.float32,
        low_out.dtype == np.float32
    ])
    
    if all_float32:
        print("  ✅ All dtypes are float32")
    else:
        print("  ❌ Some dtypes are not float32")

if __name__ == "__main__":
    raw_imdct, atracdenc_extraction = test_tdac_windowing_known_data()
    debug_imdct_implementation()
    test_float32_consistency()
    
    print(f"\n=== CRITICAL ANALYSIS ===")
    if np.max(np.abs(atracdenc_extraction)) < 1e-6:
        print("❌ MAJOR ISSUE: atracdenc extraction yields only noise!")
        print("   Our IMDCT implementation is fundamentally wrong")
        print("   The meaningful data should be in [64:192] but isn't")
    else:
        print("✅ atracdenc extraction contains meaningful data")
        print("   Our +32 offset change was wrong - we should use [64:192]")