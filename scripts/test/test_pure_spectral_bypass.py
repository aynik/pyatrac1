#!/usr/bin/env python3
"""
Test pure spectral coefficients bypassing quantization to isolate MDCT/IMDCT/TDAC.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_pure_dc_coefficient():
    """Test with pure DC coefficient, all others zero."""
    
    print("=== Pure DC Coefficient Test ===")
    
    mdct = Atrac1MDCT()
    
    # Create specs with only DC coefficient
    specs = np.zeros(512, dtype=np.float32)
    specs[0] = 1.0  # Pure DC coefficient
    
    print(f"Input: Pure DC coefficient = 1.0, all others zero")
    
    # IMDCT only
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    # Analyze result
    final_output = low_out[32:160]  # Account for QMF delay
    final_mean = np.mean(final_output)
    final_std = np.std(final_output)
    final_energy = np.sum(final_output**2)
    
    print(f"Results:")
    print(f"  Mean: {final_mean:.6f}")
    print(f"  Std: {final_std:.6f}")
    print(f"  Energy: {final_energy:.6f}")
    
    # For perfect reconstruction, DC=1.0 should give constant output
    if final_std < 0.01:
        print(f"  ✅ Good: Nearly constant output (std < 0.01)")
    else:
        print(f"  ❌ Poor: Variable output (std = {final_std:.6f})")
    
    return final_mean, final_std

def test_frequency_coefficients():
    """Test with different frequency coefficients."""
    
    print(f"\n=== Frequency Coefficient Tests ===")
    
    mdct = Atrac1MDCT()
    
    # Test different frequency bins
    test_frequencies = [0, 1, 2, 4, 8, 16, 32, 64]
    
    for freq_bin in test_frequencies:
        if freq_bin < 128:  # Within LOW band
            specs = np.zeros(512, dtype=np.float32)
            specs[freq_bin] = 1.0
            
            low_out = np.zeros(256, dtype=np.float32)
            mid_out = np.zeros(256, dtype=np.float32)
            hi_out = np.zeros(512, dtype=np.float32)
            
            mdct.imdct(specs, BlockSizeMode(False, False, False),
                       low_out, mid_out, hi_out, channel=0, frame=0)
            
            final_output = low_out[32:160]
            final_mean = np.mean(final_output)
            final_std = np.std(final_output)
            final_energy = np.sum(final_output**2)
            
            print(f"  Freq bin {freq_bin:2d}: mean={final_mean:8.6f}, std={final_std:8.6f}, energy={final_energy:8.6f}")

def test_known_mdct_coefficients():
    """Test with coefficients we know should produce specific outputs."""
    
    print(f"\n=== Known MDCT Coefficient Tests ===")
    
    mdct = Atrac1MDCT()
    
    # Test 1: Use the coefficient we got from MDCT of DC 0.5
    known_dc_coeff = -0.082304
    
    specs = np.zeros(512, dtype=np.float32)
    specs[0] = known_dc_coeff
    
    print(f"Test 1: Known DC coefficient from MDCT({known_dc_coeff:.6f})")
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    final_output = low_out[32:160]
    final_mean = np.mean(final_output)
    
    print(f"  Result: mean={final_mean:.6f}")
    print(f"  Expected: ~0.5 (original DC input)")
    print(f"  Scaling factor: {final_mean/0.5:.6f} if expecting 0.5")
    
    # Test 2: What coefficient would give us 0.5 output?
    target_output = 0.5
    target_coeff = known_dc_coeff * (target_output / final_mean)
    
    print(f"\nTest 2: Coefficient to achieve 0.5 output = {target_coeff:.6f}")
    
    specs[0] = target_coeff
    
    low_out.fill(0)
    mid_out.fill(0)
    hi_out.fill(0)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    final_output = low_out[32:160]
    final_mean = np.mean(final_output)
    
    print(f"  Result: mean={final_mean:.6f}")
    print(f"  Target: 0.5")
    print(f"  Error: {abs(final_mean - 0.5):.6f}")

def test_multi_frame_consistency():
    """Test consistency across multiple frames."""
    
    print(f"\n=== Multi-Frame Consistency Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test with same DC coefficient across multiple frames
    dc_coeff = 1.0
    
    frame_results = []
    
    for frame in range(3):
        specs = np.zeros(512, dtype=np.float32)
        specs[0] = dc_coeff
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=frame)
        
        final_output = low_out[32:160]
        final_mean = np.mean(final_output)
        final_std = np.std(final_output)
        
        frame_results.append((final_mean, final_std))
        print(f"  Frame {frame}: mean={final_mean:.6f}, std={final_std:.6f}")
    
    # Check consistency
    means = [r[0] for r in frame_results]
    stds = [r[1] for r in frame_results]
    
    mean_variation = np.std(means)
    std_variation = np.std(stds)
    
    print(f"\nConsistency analysis:")
    print(f"  Mean variation across frames: {mean_variation:.6f}")
    print(f"  Std variation across frames: {std_variation:.6f}")
    
    if mean_variation < 0.01:
        print(f"  ✅ Good: Consistent means across frames")
    else:
        print(f"  ❌ Poor: Inconsistent means across frames")

def calculate_expected_dc_coefficient():
    """Calculate what DC coefficient should theoretically produce 0.5 output."""
    
    print(f"\n=== Theoretical DC Coefficient Calculation ===")
    
    # From our scaling analysis:
    # MDCT scale: 0.044194
    # IMDCT scale: 2.0
    # Combined: 0.088388
    
    theoretical_scaling = 0.088388
    
    # To get output 0.5, what input coefficient do we need?
    # output = coeff * scaling
    # coeff = output / scaling
    
    target_output = 0.5
    theoretical_coeff = target_output / theoretical_scaling
    
    print(f"Theoretical calculation:")
    print(f"  Target output: {target_output}")
    print(f"  Theoretical scaling: {theoretical_scaling:.6f}")
    print(f"  Required coefficient: {theoretical_coeff:.6f}")
    
    # Compare with our MDCT result
    mdct_result = -0.082304
    print(f"  Actual MDCT coefficient: {mdct_result:.6f}")
    print(f"  Ratio: {theoretical_coeff / abs(mdct_result):.6f}")

if __name__ == "__main__":
    dc_mean, dc_std = test_pure_dc_coefficient()
    test_frequency_coefficients()
    test_known_mdct_coefficients()
    test_multi_frame_consistency()
    calculate_expected_dc_coefficient()
    
    print(f"\n=== SPECTRAL BYPASS TEST SUMMARY ===")
    print(f"Key findings:")
    print(f"1. Pure DC coefficient (1.0) gives mean={dc_mean:.6f}, std={dc_std:.6f}")
    print(f"2. DC variation indicates TDAC/windowing issues")
    print(f"3. Need to compare with atracdenc reference output")
    print(f"4. Scaling factors are now well understood")
    print(f"\nNext: Get atracdenc reference output for same inputs")