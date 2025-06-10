#!/usr/bin/env python3
"""
Test IMDCT scaling fix to match atracdenc exactly.
"""

import numpy as np
import math

def calc_sin_cos(n, scale):
    tmp = np.zeros(n // 2, dtype=np.float32)
    alpha = 2.0 * math.pi / (8.0 * n)
    omega = 2.0 * math.pi / n
    scale = np.sqrt(scale / n)

    for i in range(n // 4):
        tmp[2 * i] = scale * np.cos(omega * i + alpha)
        tmp[2 * i + 1] = scale * np.sin(omega * i + alpha)

    return tmp

def test_imdct_scaling():
    """Test IMDCT with corrected atracdenc scaling."""
    
    print("Testing IMDCT Scaling Fix...")
    
    # Test with 64-point IMDCT
    N = 64
    
    # atracdenc: TMIDCT(float scale = TN) : TMDCTBase(TN, scale/2)
    # For 64-point: scale = 64, TMDCTBase gets scale/2 = 32
    # calc_sin_cos gets: sqrt(32/64) = sqrt(0.5) = 0.7071
    
    scale_param = N  # TMIDCT constructor default: scale = TN
    base_scale = scale_param / 2  # TMDCTBase gets scale/2
    final_scale = np.sqrt(base_scale / N)  # calc_sin_cos: sqrt(scale/n)
    
    print(f"N={N}")
    print(f"TMIDCT scale param: {scale_param}")
    print(f"TMDCTBase scale: {base_scale}")
    print(f"calc_sin_cos final scale: {final_scale}")
    print(f"Expected: sqrt({base_scale}/{N}) = {np.sqrt(base_scale/N)}")
    
    # Our current approach
    our_scale = N * 2  # We pass N*2 to IMDCT constructor
    our_base_scale = our_scale / 2  # Our IMDCT does scale/2
    our_final_scale = np.sqrt(our_base_scale / N)
    
    print(f"\nOur current approach:")
    print(f"Our scale param: {our_scale}")
    print(f"Our base scale: {our_base_scale}")
    print(f"Our final scale: {our_final_scale}")
    
    scale_diff = abs(final_scale - our_final_scale)
    print(f"Scale difference: {scale_diff}")
    
    if scale_diff > 1e-7:
        print("⚠️ Scale mismatch! Need to fix IMDCT scaling")
        print(f"atracdenc uses: sqrt({base_scale}/{N}) = {final_scale}")
        print(f"We use: sqrt({our_base_scale}/{N}) = {our_final_scale}")
    else:
        print("✅ Scaling matches")

def test_perfect_reconstruction():
    """Test perfect reconstruction with corrected scaling."""
    
    print("\n=== Testing Perfect Reconstruction ===")
    
    # Test with simple input
    test_input = np.zeros(64, dtype=np.float32)
    test_input[0] = 1.0
    
    # Forward MDCT (scale=0.5 for 64-point)
    mdct_scale = 0.5
    sin_cos_mdct = calc_sin_cos(64, mdct_scale)
    
    # Manual MDCT
    N = 64
    n2, n4, n34, n54 = N//2, N//4, 3*N//4, 5*N//4
    cos_values = sin_cos_mdct[0::2]
    sin_values = sin_cos_mdct[1::2]
    
    # MDCT processing
    real = np.zeros(n2 // 2, dtype=np.float32)
    imag = np.zeros(n2 // 2, dtype=np.float32)
    
    for idx, k in enumerate(range(0, n4, 2)):
        r0 = test_input[n34 - 1 - k] + test_input[n34 + k]
        i0 = test_input[n4 + k] - test_input[n4 - 1 - k]
        c, s = cos_values[idx], sin_values[idx]
        real[idx] = r0 * c + i0 * s
        imag[idx] = i0 * c - r0 * s
    
    for idx, k in enumerate(range(n4, n2, 2), start=n2 // 4):
        r0 = test_input[n34 - 1 - k] - test_input[k - n4]
        i0 = test_input[n4 + k] + test_input[n54 - 1 - k]
        c, s = cos_values[idx], sin_values[idx]
        real[idx] = r0 * c + i0 * s
        imag[idx] = i0 * c - r0 * s
    
    complex_input = real + 1j * imag
    fft_result = np.fft.fft(complex_input)
    real_fft, imag_fft = fft_result.real.astype(np.float32), fft_result.imag.astype(np.float32)
    
    mdct_coeffs = np.zeros(n2, dtype=np.float32)
    for idx, k in enumerate(range(0, n2, 2)):
        r0, i0 = real_fft[idx], imag_fft[idx]
        c, s = cos_values[idx], sin_values[idx]
        mdct_coeffs[k] = -r0 * c - i0 * s
        mdct_coeffs[n2 - 1 - k] = -r0 * s + i0 * c
    
    print(f"MDCT coefficients: {mdct_coeffs[:8]}")
    
    # IMDCT with corrected scaling (atracdenc approach)
    imdct_scale = N  # atracdenc TMIDCT default scale = TN = 64
    base_scale = imdct_scale / 2  # TMDCTBase gets scale/2 = 32
    sin_cos_imdct = calc_sin_cos(N, base_scale)
    cos_values_imdct = sin_cos_imdct[0::2]
    sin_values_imdct = sin_cos_imdct[1::2]
    
    print(f"IMDCT scale: {imdct_scale} -> base: {base_scale} -> final: {np.sqrt(base_scale/N)}")
    
    # IMDCT processing
    real_imdct = np.zeros(n2 // 2, dtype=np.float32)
    imag_imdct = np.zeros(n2 // 2, dtype=np.float32)
    
    for idx, k in enumerate(range(0, n2, 2)):
        r0 = mdct_coeffs[k]
        i0 = mdct_coeffs[n2 - 1 - k]
        c, s = cos_values_imdct[idx], sin_values_imdct[idx]
        real_imdct[idx] = -2.0 * (i0 * s + r0 * c)
        imag_imdct[idx] = -2.0 * (i0 * c - r0 * s)
    
    complex_imdct = real_imdct + 1j * imag_imdct
    fft_imdct = np.fft.fft(complex_imdct)
    real_fft_imdct, imag_fft_imdct = fft_imdct.real.astype(np.float32), fft_imdct.imag.astype(np.float32)
    
    # Output reconstruction
    imdct_output = np.zeros(N, dtype=np.float32)
    
    for idx, k in enumerate(range(0, n4, 2)):
        r0, i0 = real_fft_imdct[idx], imag_fft_imdct[idx]
        c, s = cos_values_imdct[idx], sin_values_imdct[idx]
        r1 = r0 * c + i0 * s
        i1 = r0 * s - i0 * c
        imdct_output[n34 - 1 - k] = r1
        imdct_output[n34 + k] = r1
        imdct_output[n4 + k] = i1
        imdct_output[n4 - 1 - k] = -i1
    
    for idx, k in enumerate(range(n4, n2, 2), start=n2 // 4):
        r0, i0 = real_fft_imdct[idx], imag_fft_imdct[idx]
        c, s = cos_values_imdct[idx], sin_values_imdct[idx]
        r1 = r0 * c + i0 * s
        i1 = r0 * s - i0 * c
        imdct_output[n34 - 1 - k] = r1
        imdct_output[k - n4] = -r1
        imdct_output[n4 + k] = i1
        imdct_output[n54 - 1 - k] = i1
    
    print(f"IMDCT output: {imdct_output[:8]}")
    print(f"Original input: {test_input[:8]}")
    
    # Check reconstruction
    error = np.mean(np.abs(test_input - imdct_output))
    print(f"Reconstruction error: {error}")
    
    if error < 1e-6:
        print("✅ Perfect reconstruction achieved!")
    else:
        print("❌ Reconstruction still has errors")
        max_error = np.max(np.abs(test_input - imdct_output))
        print(f"Max error: {max_error}")

if __name__ == "__main__":
    test_imdct_scaling()
    test_perfect_reconstruction()