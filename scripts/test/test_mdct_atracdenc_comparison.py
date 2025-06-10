#!/usr/bin/env python3
"""
Direct comparison of MDCT/IMDCT kernels with atracdenc scaling and FFT approach.
"""

import numpy as np
import math
from pyatrac1.core.mdct import MDCT, IMDCT, calc_sin_cos

def atracdenc_fft_approach():
    """Test MDCT using atracdenc's exact approach for comparison."""
    
    print("Testing atracdenc-style MDCT/IMDCT...")
    
    # Test with 64-point MDCT (simplest case)
    N = 64
    scale = 0.5
    
    # Create test input (impulse at beginning)
    test_input = np.zeros(N, dtype=np.float32)
    test_input[0] = 1.0
    
    print(f"Input impulse: {test_input[:8]}")
    
    # Manual MDCT following atracdenc exactly
    n2 = N // 2  # 32
    n4 = N // 4  # 16
    n34 = 3 * n4  # 48
    n54 = 5 * n4  # 80
    
    # Calculate SinCos exactly like atracdenc
    sin_cos = calc_sin_cos(N, scale)
    cos_values = sin_cos[0::2]
    sin_values = sin_cos[1::2]
    
    print(f"SinCos first 8: {sin_cos[:8]}")
    print(f"Scale factor: {scale}, sqrt(scale/N): {np.sqrt(scale/N)}")
    
    # Pre-rotation stage (first loop)
    real = np.zeros(n2 // 2, dtype=np.float32)  # n2/2 = 16 for 64-point
    imag = np.zeros(n2 // 2, dtype=np.float32)
    
    pre_rotation_debug = []
    for idx, k in enumerate(range(0, n4, 2)):
        r0 = test_input[n34 - 1 - k] + test_input[n34 + k]
        i0 = test_input[n4 + k] - test_input[n4 - 1 - k]
        c = cos_values[idx]
        s = sin_values[idx]
        real[idx] = r0 * c + i0 * s
        imag[idx] = i0 * c - r0 * s
        
        if k < 8:
            pre_rotation_debug.extend([r0, i0, c, s, real[idx], imag[idx]])
            
    print(f"Pre-rotation debug: {pre_rotation_debug[:12]}")
    
    # Second loop
    for idx, k in enumerate(range(n4, n2, 2), start=n2 // 4):
        r0 = test_input[n34 - 1 - k] - test_input[k - n4]
        i0 = test_input[n4 + k] + test_input[n54 - 1 - k]
        c = cos_values[idx]
        s = sin_values[idx]
        real[idx] = r0 * c + i0 * s
        imag[idx] = i0 * c - r0 * s
    
    print(f"FFT input real: {real}")
    print(f"FFT input imag: {imag}")
    
    # Perform FFT
    complex_input = real + 1j * imag
    fft_result = np.fft.fft(complex_input)
    real_fft = fft_result.real.astype(np.float32)
    imag_fft = fft_result.imag.astype(np.float32)
    
    print(f"FFT output real: {real_fft}")
    print(f"FFT output imag: {imag_fft}")
    
    # Post-rotation stage
    mdct_output = np.zeros(n2, dtype=np.float32)
    post_rotation_debug = []
    
    for idx, k in enumerate(range(0, n2, 2)):
        r0 = real_fft[idx]
        i0 = imag_fft[idx]
        c = cos_values[idx]
        s = sin_values[idx]
        mdct_output[k] = -r0 * c - i0 * s
        mdct_output[n2 - 1 - k] = -r0 * s + i0 * c
        
        if k < 8:
            post_rotation_debug.extend([r0, i0, c, s, mdct_output[k], mdct_output[n2 - 1 - k]])
    
    print(f"Post-rotation debug: {post_rotation_debug[:12]}")
    print(f"MDCT output: {mdct_output[:8]}")
    
    return mdct_output

def test_scaling_differences():
    """Test different scaling approaches to match atracdenc."""
    
    print("\n=== Testing Scaling Differences ===")
    
    # Test atracdenc constructor scaling
    # TMDCT(float scale = 1.0) : TMDCTBase(TN, scale)
    # calc_sin_cos: scale = sqrt(scale/n)
    
    for N, scale_param in [(64, 0.5), (256, 0.5), (512, 1.0)]:
        print(f"\nTesting N={N}, scale_param={scale_param}")
        
        # atracdenc scaling
        final_scale = np.sqrt(scale_param / N)
        print(f"atracdenc final scale: sqrt({scale_param}/{N}) = {final_scale}")
        
        # Our current scaling
        our_mdct = MDCT(N, scale_param)
        our_scale = np.sqrt(our_mdct.Scale / N) 
        print(f"Our final scale: sqrt({our_mdct.Scale}/{N}) = {our_scale}")
        
        scale_diff = abs(final_scale - our_scale)
        print(f"Scale difference: {scale_diff}")
        
        if scale_diff > 1e-7:
            print(f"⚠️ Scale mismatch detected!")

def compare_with_our_implementation():
    """Compare manual implementation with our MDCT class."""
    
    print("\n=== Comparing with Our Implementation ===")
    
    test_input = np.zeros(64, dtype=np.float32)
    test_input[0] = 1.0
    
    # Our implementation
    our_mdct = MDCT(64, 0.5)
    our_output = our_mdct(test_input)
    
    print(f"Our MDCT output: {our_output[:8]}")
    
    # Manual implementation
    manual_output = atracdenc_fft_approach()
    
    # Compare
    diff = np.mean(np.abs(our_output - manual_output))
    print(f"\nMean difference: {diff}")
    
    if diff < 1e-6:
        print("✅ Our implementation matches manual approach")
    else:
        print("❌ Our implementation differs from manual approach")
        print(f"Max difference: {np.max(np.abs(our_output - manual_output))}")
        print(f"First 8 differences: {(our_output - manual_output)[:8]}")

if __name__ == "__main__":
    atracdenc_fft_approach()
    test_scaling_differences()
    compare_with_our_implementation()