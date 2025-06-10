#!/usr/bin/env python3
"""
Debug MDCT coefficients to isolate if issue is in MDCT or IMDCT.
"""

import numpy as np
from pyatrac1.core.mdct import MDCT, IMDCT

def debug_mdct_coefficients():
    """Debug MDCT coefficients for different sizes."""
    
    print("=== MDCT Coefficients Debug ===")
    
    # Test each size with DC input (should give strong first coefficient)
    test_cases = [
        (64, 0.5, "works"),
        (256, 0.5, "fails"),
        (512, 1, "fails")
    ]
    
    for size, scale, status in test_cases:
        print(f"\n{size}-point MDCT ({status}):")
        
        # DC input - should concentrate energy in first coefficient
        test_input = np.ones(size, dtype=np.float32)
        mdct = MDCT(size, scale)
        coeffs = mdct(test_input)
        
        print(f"  Input: all ones (DC)")
        print(f"  MDCT output length: {len(coeffs)}")
        print(f"  First 8 coefficients: {coeffs[:8]}")
        print(f"  Max coefficient: {np.max(np.abs(coeffs))}")
        print(f"  Energy: {np.sum(coeffs**2)}")
        
        # Check if coefficients are reasonable
        if np.max(np.abs(coeffs)) < 1e-6:
            print(f"  ❌ COEFFICIENTS ARE NEAR ZERO - MDCT FAILURE")
        else:
            print(f"  ✅ Coefficients look reasonable")

def debug_imdct_with_known_coeffs():
    """Test IMDCT with known good coefficients."""
    
    print("\n=== IMDCT Debug with Known Coefficients ===")
    
    # Use coefficients from working 64-point case
    test_input_64 = np.ones(64, dtype=np.float32)
    mdct64 = MDCT(64, 0.5)
    good_coeffs = mdct64(test_input_64)
    
    print(f"Known good 64-point coefficients: {good_coeffs[:8]}")
    
    # Test IMDCT at different sizes
    sizes_and_scales = [(64, 128), (256, 512), (512, 1024)]
    
    for size, imdct_scale in sizes_and_scales:
        print(f"\n{size}-point IMDCT (scale={imdct_scale}):")
        
        if size == 64:
            # Use actual good coefficients for 64-point
            test_coeffs = good_coeffs
        else:
            # Create synthetic coefficients for larger sizes
            test_coeffs = np.zeros(size // 2, dtype=np.float32)
            test_coeffs[0] = good_coeffs[0]  # Copy DC component
            test_coeffs[1] = good_coeffs[1] if len(good_coeffs) > 1 else 0
        
        imdct = IMDCT(size, imdct_scale)
        
        print(f"  Input coeffs length: {len(test_coeffs)}")
        print(f"  Input coeffs[0:4]: {test_coeffs[:4]}")
        
        reconstructed = imdct(test_coeffs)
        
        print(f"  Output length: {len(reconstructed)}")
        print(f"  Output[32:36]: {reconstructed[32:36]}")
        print(f"  Max output: {np.max(np.abs(reconstructed))}")
        
        if np.max(np.abs(reconstructed)) < 1e-6:
            print(f"  ❌ IMDCT OUTPUT IS NEAR ZERO - IMDCT FAILURE")
        else:
            print(f"  ✅ IMDCT output looks reasonable")

if __name__ == "__main__":
    debug_mdct_coefficients()
    debug_imdct_with_known_coeffs()