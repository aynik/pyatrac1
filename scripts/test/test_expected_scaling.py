#!/usr/bin/env python3
"""
Test if our MDCT/IMDCT scaling matches atracdenc expected scaling factors.
"""

import numpy as np
from pyatrac1.core.mdct import MDCT, IMDCT

def test_expected_scaling():
    """Test if our MDCT/IMDCT matches atracdenc expected scaling."""
    
    print("=== Testing Expected atracdenc Scaling ===")
    
    # Create test input
    test_input = np.ones(64, dtype=np.float32)  # DC input for clear scaling test
    
    # Test 64-point (used for low/mid bands in short mode)
    mdct64 = MDCT(64, 0.5)
    imdct64 = IMDCT(64, 64 * 2)  # Match atracdenc scaling
    
    coeffs = mdct64(test_input)
    reconstructed = imdct64(coeffs)
    
    print(f"64-point MDCT/IMDCT:")
    print(f"Input amplitude: {test_input[0]}")
    print(f"Reconstructed amplitude: {reconstructed[32]}")  # Skip first 32 samples as per test
    print(f"Scaling factor: {reconstructed[32] / test_input[0]}")
    print(f"Expected for low/mid: 1/4 = {1/4}")
    print(f"Expected for high: 1/2 = {1/2}")
    
    # Test 256-point (used for low/mid bands in long mode)
    test_input_256 = np.ones(256, dtype=np.float32)
    mdct256 = MDCT(256, 0.5)
    imdct256 = IMDCT(256, 256 * 2)  # Match atracdenc scaling
    
    coeffs_256 = mdct256(test_input_256)
    reconstructed_256 = imdct256(coeffs_256)
    
    print(f"\n256-point MDCT/IMDCT:")
    print(f"Input amplitude: {test_input_256[0]}")
    print(f"Reconstructed amplitude: {reconstructed_256[32]}")  # Skip first 32 samples
    print(f"Scaling factor: {reconstructed_256[32] / test_input_256[0]}")
    print(f"Expected for low/mid: 1/4 = {1/4}")
    
    # Test 512-point (used for high band in long mode)
    test_input_512 = np.ones(512, dtype=np.float32)
    mdct512 = MDCT(512, 1)
    imdct512 = IMDCT(512, 512 * 2)  # Match atracdenc scaling
    
    coeffs_512 = mdct512(test_input_512)
    reconstructed_512 = imdct512(coeffs_512)
    
    print(f"\n512-point MDCT/IMDCT:")
    print(f"Input amplitude: {test_input_512[0]}")
    print(f"Reconstructed amplitude: {reconstructed_512[32]}")  # Skip first 32 samples
    print(f"Scaling factor: {reconstructed_512[32] / test_input_512[0]}")
    print(f"Expected for high: 1/2 = {1/2}")
    
def test_scaling_validation():
    """Validate if our scaling matches the expected factors."""
    
    print("\n=== Scaling Validation ===")
    
    # Test each size
    sizes_and_scales = [
        (64, 0.5, "low/mid", 1/4),
        (256, 0.5, "low/mid", 1/4), 
        (512, 1, "high", 1/2)
    ]
    
    for size, mdct_scale, band_type, expected_factor in sizes_and_scales:
        test_input = np.ones(size, dtype=np.float32)
        
        mdct = MDCT(size, mdct_scale)
        imdct = IMDCT(size, size * 2)  # Match atracdenc TMIDCT(TN * 2)
        
        coeffs = mdct(test_input)
        reconstructed = imdct(coeffs)
        
        # Get scaling factor (skip first 32 samples per atracdenc test)
        actual_factor = reconstructed[32] / test_input[0]
        
        error = abs(actual_factor - expected_factor)
        tolerance = 0.01  # 1% tolerance
        
        print(f"{size}-point ({band_type}): actual={actual_factor:.4f}, expected={expected_factor:.4f}, error={error:.4f}")
        
        if error < tolerance:
            print(f"  ✅ PASS - Within tolerance")
        else:
            print(f"  ❌ FAIL - Outside tolerance ({tolerance})")

if __name__ == "__main__":
    test_expected_scaling()
    test_scaling_validation()