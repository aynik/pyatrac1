#!/usr/bin/env python3
"""
Debug MDCT/IMDCT scale factors to understand normalization.
"""

import numpy as np
import math
from pyatrac1.core.mdct import MDCT, IMDCT, calc_sin_cos

def debug_scale_factors():
    """Debug the actual scale factors being used."""
    
    print("=== MDCT Scale Debug ===")
    
    # Test 64-point
    mdct64 = MDCT(64, 0.5)
    imdct64 = IMDCT(64, 64)
    
    print(f"MDCT64 - N:{mdct64.N}, Scale:{mdct64.Scale}")
    print(f"IMDCT64 - N:{imdct64.N}, Scale:{imdct64.Scale}")
    
    # Check SinCos computation
    mdct_sincos = mdct64.SinCos
    imdct_sincos = imdct64.SinCos
    
    print(f"MDCT SinCos first 8: {mdct_sincos[:8]}")
    print(f"IMDCT SinCos first 8: {imdct_sincos[:8]}")
    
    # Calculate expected values
    mdct_expected_scale = np.sqrt(0.5 / 64)
    imdct_expected_scale = np.sqrt(32 / 64)  # Should be sqrt(0.5) = 0.7071
    
    print(f"MDCT expected scale: {mdct_expected_scale}")
    print(f"IMDCT expected scale: {imdct_expected_scale}")
    
    # Check if our SinCos values match expected
    mdct_actual_scale = mdct_sincos[0] / np.cos(2.0 * math.pi * 0 / 64 + 2.0 * math.pi / (8.0 * 64))
    imdct_actual_scale = imdct_sincos[0] / np.cos(2.0 * math.pi * 0 / 64 + 2.0 * math.pi / (8.0 * 64))
    
    print(f"MDCT actual scale (from SinCos[0]): {mdct_actual_scale}")
    print(f"IMDCT actual scale (from SinCos[0]): {imdct_actual_scale}")
    
def test_minimal_reconstruction():
    """Test minimal MDCT->IMDCT with simple input."""
    
    print("\n=== Minimal Reconstruction Test ===")
    
    # Create simple test: single impulse
    test_input = np.zeros(64, dtype=np.float32)
    test_input[0] = 1.0
    
    mdct64 = MDCT(64, 0.5)
    imdct64 = IMDCT(64, 64)
    
    # Forward transform
    coeffs = mdct64(test_input)
    print(f"MDCT coefficients: {coeffs[:8]}")
    
    # Inverse transform
    reconstructed = imdct64(coeffs)
    print(f"IMDCT output: {reconstructed[:8]}")
    print(f"Original input: {test_input[:8]}")
    
    # The first sample should be close to 1.0 for perfect reconstruction
    print(f"First sample error: {abs(reconstructed[0] - 1.0)}")
    
    # Check total energy conservation
    input_energy = np.sum(test_input**2)
    output_energy = np.sum(reconstructed**2)
    coeff_energy = np.sum(coeffs**2)
    
    print(f"Input energy: {input_energy}")
    print(f"Coefficient energy: {coeff_energy}")  
    print(f"Output energy: {output_energy}")
    print(f"Energy ratios - Coeffs/Input: {coeff_energy/input_energy}")
    print(f"Energy ratios - Output/Input: {output_energy/input_energy}")

def test_dc_input():
    """Test with DC (all ones) input to check normalization."""
    
    print("\n=== DC Input Test ===")
    
    test_input = np.ones(64, dtype=np.float32)
    
    mdct64 = MDCT(64, 0.5)
    imdct64 = IMDCT(64, 64)
    
    coeffs = mdct64(test_input)
    reconstructed = imdct64(coeffs)
    
    print(f"DC input: {test_input[0]} (all same)")
    print(f"DC coefficients [0]: {coeffs[0]}")
    print(f"DC reconstructed [0]: {reconstructed[0]}")
    print(f"DC reconstruction error: {abs(reconstructed[0] - 1.0)}")

if __name__ == "__main__":
    debug_scale_factors()
    test_minimal_reconstruction()
    test_dc_input()