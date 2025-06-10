#!/usr/bin/env python3
"""
Test IMDCT output placement to identify exactly where meaningful data appears.
Following evaluation.txt analysis steps.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_python_good_data_region():
    """Step 1: Confirm the Python "Good Data" Region for DC 0.5 test."""
    
    print("=== Step 1: Python Good Data Region Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # DC 0.5 test case
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    # MDCT to get coefficients
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"MDCT DC coefficient: {low_coeffs[0]:.6f}")
    
    # Direct call to IMDCT.__call__ to get raw self.buf output
    raw_imdct_output = mdct.imdct256(low_coeffs)
    
    print(f"Raw IMDCT output analysis:")
    print(f"  Total length: {len(raw_imdct_output)}")
    print(f"  Dtype: {raw_imdct_output.dtype}")
    
    # Analyze each 32-sample segment to find where meaningful data is
    print(f"\nSegment analysis (32-sample chunks):")
    for i in range(0, 256, 32):
        segment = raw_imdct_output[i:i+32]
        segment_mean = np.mean(np.abs(segment))
        segment_max = np.max(np.abs(segment))
        print(f"  [{i:3d}:{i+32:3d}]: mean_abs={segment_mean:.6f}, max_abs={segment_max:.6f}")
        if segment_mean > 0.01:  # Meaningful data threshold
            print(f"    ✅ MEANINGFUL DATA: {segment[:4]} ... {segment[-4:]}")
    
    # Specifically check the regions mentioned in previous tests
    regions_to_check = [
        ("atracdenc target [64:192]", raw_imdct_output[64:192]),
        ("Our observed [96:224]", raw_imdct_output[96:224] if len(raw_imdct_output) >= 224 else raw_imdct_output[96:]),
        ("First half [0:128]", raw_imdct_output[0:128]),
        ("Second half [128:256]", raw_imdct_output[128:256]),
    ]
    
    print(f"\nSpecific region analysis:")
    for name, region in regions_to_check:
        if len(region) > 0:
            region_energy = np.sum(region**2)
            region_mean = np.mean(region)
            region_std = np.std(region)
            print(f"  {name}:")
            print(f"    Energy: {region_energy:.6f}")
            print(f"    Mean: {region_mean:.6f}")
            print(f"    Std: {region_std:.6f}")
            print(f"    Sample: {region[:4]} ... {region[-4:] if len(region) > 4 else region}")
    
    return raw_imdct_output

def inspect_imdct_implementation():
    """Step 2: Inspect Python IMDCT.__call__ implementation details."""
    
    print(f"\n=== Step 2: IMDCT Implementation Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # Test with simple coefficient pattern
    test_coeffs = np.zeros(128, dtype=np.float32)
    test_coeffs[0] = 1.0  # Pure DC
    
    print(f"Testing with pure DC coefficient=1.0")
    
    # Get raw output and analyze the indexing pattern
    raw_output = mdct.imdct256(test_coeffs)
    
    print(f"Raw IMDCT output for pure DC:")
    
    # Check the key indexing regions used in C++ TMIDCT
    n = 256  # Transform size
    n2 = n // 2  # 128
    n4 = n // 4  # 64
    n34 = 3 * n4  # 192
    n54 = 5 * n4  # 320 (but this exceeds our 256 buffer)
    
    print(f"  Transform size: {n}")
    print(f"  n2={n2}, n4={n4}, n34={n34}, n54={n54}")
    
    # Check C++ indexing regions:
    # First loop: Buf[n34 - 1 - n] = r1; Buf[n34 + n] = r1; (for n in [0, n4))
    # Second loop: Buf[n34 - 1 - n] = r1; Buf[n - n4] = -r1; (for n in [n4, n2))
    
    print(f"\nC++ indexing analysis:")
    print(f"  First loop indices (n in [0, {n4})):")
    for n_val in [0, 2, 32, 62]:  # Sample values
        if n_val < n4:
            idx1 = n34 - 1 - n_val  # Should be around 192-1-n = 191, 189, etc.
            idx2 = n34 + n_val      # Should be around 192+n = 192, 194, etc.
            print(f"    n={n_val}: Buf[{idx1}], Buf[{idx2}]")
    
    print(f"  Second loop indices (n in [{n4}, {n2})):")
    for n_val in [64, 66, 96, 126]:  # Sample values
        if n4 <= n_val < n2:
            idx1 = n34 - 1 - n_val  # Should be around 192-1-n = 127, 125, etc.
            idx2 = n_val - n4       # Should be around n-64 = 0, 2, etc.
            print(f"    n={n_val}: Buf[{idx1}], Buf[{idx2}]")
    
    # Check our Python output at these critical indices
    print(f"\nPython output at key indices:")
    critical_indices = [0, 32, 64, 96, 128, 160, 192, 224]
    for idx in critical_indices:
        if idx < len(raw_output):
            val = raw_output[idx]
            print(f"  raw_output[{idx}] = {val:.6f}")
    
    return raw_output

def test_single_coefficient():
    """Step 3: Test with a single non-DC MDCT coefficient."""
    
    print(f"\n=== Step 3: Single Coefficient Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test different frequency bins
    test_bins = [1, 4, 8, 16, 32, 64]
    
    for k_test in test_bins:
        if k_test < 128:
            print(f"\nTesting coefficient bin {k_test}:")
            
            # Create specs with only one non-zero coefficient
            test_coeffs = np.zeros(128, dtype=np.float32)
            test_coeffs[k_test] = 1.0
            
            # Get IMDCT output
            raw_output = mdct.imdct256(test_coeffs)
            
            # Find where the peak response occurs
            peak_idx = np.argmax(np.abs(raw_output))
            peak_val = raw_output[peak_idx]
            
            print(f"  Peak at index {peak_idx}, value {peak_val:.6f}")
            
            # Check which 64-sample quarter contains the peak
            quarter = peak_idx // 64
            quarter_names = ["Q1[0:64]", "Q2[64:128]", "Q3[128:192]", "Q4[192:256]"]
            print(f"  Peak in {quarter_names[quarter]}")
            
            # For frequency bin k, we expect a cosine/sine pattern
            # Check if the pattern looks like expected frequency content
            
            # Analyze the pattern around the peak
            start_idx = max(0, peak_idx - 8)
            end_idx = min(len(raw_output), peak_idx + 8)
            pattern = raw_output[start_idx:end_idx]
            
            print(f"  Pattern around peak: {pattern}")
            
            # Check if atracdenc extraction [64:192] captures meaningful content
            atracdenc_region = raw_output[64:192]
            atracdenc_energy = np.sum(atracdenc_region**2)
            total_energy = np.sum(raw_output**2)
            atracdenc_fraction = atracdenc_energy / total_energy if total_energy > 0 else 0
            
            print(f"  atracdenc region [64:192] contains {atracdenc_fraction*100:.1f}% of energy")
            
            if atracdenc_fraction > 0.8:
                print(f"  ✅ atracdenc region captures most energy")
            elif atracdenc_fraction > 0.3:
                print(f"  ⚠️  atracdenc region captures some energy")
            else:
                print(f"  ❌ atracdenc region misses most energy")

def verify_atracdenc_target_alignment():
    """Step 4: Verify atracdenc IMDCT target alignment for DC."""
    
    print(f"\n=== Step 4: atracdenc Target Alignment Verification ===")
    
    # This step would ideally compare with actual atracdenc output
    # For now, we'll analyze what we expect based on the MDCT/IMDCT theory
    
    print("Expected behavior for DC 0.5 -> MDCT -> IMDCT:")
    print("1. DC 0.5 input should produce MDCT DC coefficient")
    print("2. IMDCT of DC coefficient should produce constant amplitude")
    print("3. atracdenc extracts [64:192] from 256-sample IMDCT output")
    print("4. This extraction should contain the reconstruction data")
    
    # Test our understanding
    mdct = Atrac1MDCT()
    
    dc_input = np.ones(128, dtype=np.float32) * 0.5
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, dc_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    dc_coeff = specs[0]
    print(f"\nActual MDCT DC coefficient: {dc_coeff:.6f}")
    
    # Direct IMDCT of just the DC coefficient
    pure_dc_coeffs = np.zeros(128, dtype=np.float32)
    pure_dc_coeffs[0] = dc_coeff
    
    pure_dc_output = mdct.imdct256(pure_dc_coeffs)
    
    print(f"Pure DC IMDCT output analysis:")
    print(f"  Mean value: {np.mean(pure_dc_output):.6f}")
    print(f"  Target region [64:192] mean: {np.mean(pure_dc_output[64:192]):.6f}")
    print(f"  Target region [64:192] std: {np.std(pure_dc_output[64:192]):.6f}")
    
    # Expected: for DC input, IMDCT should produce roughly constant values
    # The question is whether these appear in [64:192] or elsewhere
    
    # Check if [64:192] contains the expected reconstruction amplitude
    expected_amplitude = 0.5 * 0.25  # Input * expected MDCT/IMDCT scaling
    actual_mean = np.mean(pure_dc_output[64:192])
    
    print(f"  Expected amplitude: ~{expected_amplitude:.6f}")
    print(f"  Actual mean in [64:192]: {actual_mean:.6f}")
    print(f"  Ratio: {actual_mean/expected_amplitude:.3f}")
    
    if abs(actual_mean - expected_amplitude) < 0.01:
        print("  ✅ atracdenc target region contains expected amplitude")
    else:
        print("  ❌ atracdenc target region does not match expected amplitude")
        
        # Find where the expected amplitude actually appears
        for i in range(0, 256, 32):
            segment_mean = np.mean(pure_dc_output[i:i+32])
            if abs(segment_mean - expected_amplitude) < 0.01:
                print(f"    Expected amplitude found in [{i}:{i+32}]")

if __name__ == "__main__":
    print("Following evaluation.txt analysis steps...\n")
    
    raw_output = test_python_good_data_region()
    inspect_imdct_implementation()
    test_single_coefficient()
    verify_atracdenc_target_alignment()
    
    print(f"\n=== SUMMARY ===")
    print("This analysis will help identify:")
    print("1. Exactly where Python IMDCT places meaningful data")
    print("2. Whether our indexing matches C++ TMIDCT indexing")
    print("3. Whether frequency components appear in expected positions")
    print("4. Whether atracdenc's [64:192] extraction target is correct")