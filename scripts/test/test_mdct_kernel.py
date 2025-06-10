#!/usr/bin/env python3
"""
Test MDCT/IMDCT kernels directly to compare with atracdenc.
"""

import numpy as np
from pyatrac1.core.mdct import MDCT, IMDCT

def test_mdct_kernel():
    """Test MDCT kernel with simple known inputs."""
    
    # Test with impulse input
    print("Testing MDCT Kernel...")
    
    # Test 64-point MDCT
    mdct64 = MDCT(64, 0.5)
    
    # Impulse at beginning
    input_impulse = np.zeros(64, dtype=np.float32)
    input_impulse[0] = 1.0
    
    coeffs_impulse = mdct64(input_impulse)
    print(f"MDCT64 impulse output (first 8): {coeffs_impulse[:8]}")
    
    # Test with DC input
    input_dc = np.ones(64, dtype=np.float32)
    coeffs_dc = mdct64(input_dc)
    print(f"MDCT64 DC output (first 8): {coeffs_dc[:8]}")
    
    # Test IMDCT
    print("\nTesting IMDCT Kernel...")
    imdct64 = IMDCT(64, 64*2)
    
    # Test with impulse coefficients
    coeffs_impulse_test = np.zeros(32, dtype=np.float32)
    coeffs_impulse_test[0] = 1.0
    
    recon_impulse = imdct64(coeffs_impulse_test)
    print(f"IMDCT64 impulse reconstruction (first 8): {recon_impulse[:8]}")
    
    # Test perfect reconstruction
    print("\nTesting Perfect Reconstruction...")
    
    # Create random input
    np.random.seed(42)  # For reproducible results
    test_input = np.random.randn(64).astype(np.float32) * 0.1
    
    # MDCT -> IMDCT
    coeffs = mdct64(test_input)
    reconstructed = imdct64(coeffs)
    
    print(f"Original input (first 8): {test_input[:8]}")
    print(f"Reconstructed (first 8): {reconstructed[:8]}")
    
    # Check reconstruction error (should be very small for pure MDCT/IMDCT)
    error = np.mean(np.abs(test_input - reconstructed[:64]))
    print(f"Mean reconstruction error: {error}")
    
    if error < 1e-6:
        print("✅ MDCT/IMDCT kernels appear to be working correctly")
    else:
        print("❌ MDCT/IMDCT kernels have reconstruction errors")
    
    # Test different sizes
    print("\nTesting Different MDCT Sizes...")
    
    # Test 256-point
    mdct256 = MDCT(256, 0.5)
    imdct256 = IMDCT(256, 256*2)
    
    test_input_256 = np.random.randn(256).astype(np.float32) * 0.1
    coeffs_256 = mdct256(test_input_256)
    recon_256 = imdct256(coeffs_256)
    error_256 = np.mean(np.abs(test_input_256 - recon_256[:256]))
    print(f"MDCT256 reconstruction error: {error_256}")
    
    # Test 512-point
    mdct512 = MDCT(512, 1)
    imdct512 = IMDCT(512, 512*2)
    
    test_input_512 = np.random.randn(512).astype(np.float32) * 0.1
    coeffs_512 = mdct512(test_input_512)
    recon_512 = imdct512(coeffs_512)
    error_512 = np.mean(np.abs(test_input_512 - recon_512[:512]))
    print(f"MDCT512 reconstruction error: {error_512}")
    
    return error < 1e-6 and error_256 < 1e-6 and error_512 < 1e-6

if __name__ == "__main__":
    success = test_mdct_kernel()
    print(f"\nMDCT Kernel Test Result: {'PASS' if success else 'FAIL'}")