#!/usr/bin/env python3
"""
Debug IMDCT with full vs synthetic coefficients to find destructive interference.
"""

import numpy as np
from pyatrac1.core.mdct import MDCT, IMDCT

def debug_imdct_full_vs_synthetic():
    """Compare IMDCT behavior with full MDCT coefficients vs synthetic ones."""
    
    print("=== IMDCT Full vs Synthetic Coefficients Debug ===")
    
    # Test 256-point case
    size = 256
    
    # Get full coefficient spectrum from MDCT
    test_input = np.ones(size, dtype=np.float32)
    mdct256 = MDCT(size, 0.5)
    coeffs_full = mdct256(test_input)
    
    print(f"Full coefficients from MDCT(ones({size})):")
    print(f"  Length: {len(coeffs_full)}")
    print(f"  First 8: {coeffs_full[:8]}")
    print(f"  Last 8: {coeffs_full[-8:]}")
    print(f"  Energy: {np.sum(coeffs_full**2)}")
    print(f"  Max: {np.max(np.abs(coeffs_full))}")
    
    # Create synthetic coefficients (mostly zeros)
    coeffs_synthetic = np.zeros(len(coeffs_full), dtype=np.float32)
    coeffs_synthetic[0] = coeffs_full[0]  # Copy DC component
    coeffs_synthetic[1] = coeffs_full[1]  # Copy first AC component
    
    print(f"\nSynthetic coefficients (DC + first AC only):")
    print(f"  First 8: {coeffs_synthetic[:8]}")
    print(f"  Energy: {np.sum(coeffs_synthetic**2)}")
    
    # Test IMDCT with both
    imdct256 = IMDCT(size, size * 2)
    
    print(f"\nIMDCT with full coefficients:")
    reconstructed_full = imdct256(coeffs_full.copy())
    print(f"  Output[32:36]: {reconstructed_full[32:36]}")
    print(f"  Max output: {np.max(np.abs(reconstructed_full))}")
    print(f"  Energy: {np.sum(reconstructed_full**2)}")
    
    print(f"\nIMDCT with synthetic coefficients:")
    reconstructed_synthetic = imdct256(coeffs_synthetic.copy())
    print(f"  Output[32:36]: {reconstructed_synthetic[32:36]}")
    print(f"  Max output: {np.max(np.abs(reconstructed_synthetic))}")
    print(f"  Energy: {np.sum(reconstructed_synthetic**2)}")

def debug_imdct_internals():
    """Step through IMDCT internals to find where cancellation occurs."""
    
    print("\n=== IMDCT Internals Debug ===")
    
    # Get failing case
    size = 256
    test_input = np.ones(size, dtype=np.float32)
    mdct256 = MDCT(size, 0.5)
    coeffs = mdct256(test_input)
    
    print(f"Debugging IMDCT({size}) with full coefficient spectrum...")
    
    # Manual IMDCT step-by-step
    imdct = IMDCT(size, size * 2)
    n2 = imdct.N // 2  # 128
    n4 = imdct.N // 4  # 64
    n34 = 3 * n4      # 192
    n54 = 5 * n4      # 320
    
    cos_values = imdct.SinCos[0::2]  # Length 64
    sin_values = imdct.SinCos[1::2]  # Length 64
    
    print(f"IMDCT parameters: N={imdct.N}, n2={n2}, n4={n4}")
    print(f"SinCos length: {len(imdct.SinCos)}, cos/sin length: {len(cos_values)}")
    print(f"Input coeffs length: {len(coeffs)}")
    
    # Pre-FFT stage
    size_fft = n2 // 2  # 64
    real = np.zeros(size_fft, dtype=np.float32)
    imag = np.zeros(size_fft, dtype=np.float32)
    
    print(f"\nPre-FFT stage:")
    print(f"FFT size: {size_fft}")
    
    # Check a few iterations of the pre-FFT loop
    for idx, k in enumerate(range(0, n2, 2)):
        r0 = coeffs[k]
        i0 = coeffs[n2 - 1 - k]
        c = cos_values[idx]
        s = sin_values[idx]
        
        real_val = -2.0 * (i0 * s + r0 * c)
        imag_val = -2.0 * (i0 * c - r0 * s)
        
        real[idx] = real_val
        imag[idx] = imag_val
        
        if idx < 8:  # Print first few iterations
            print(f"  idx={idx}, k={k}: r0={r0:.6f}, i0={i0:.6f}, c={c:.6f}, s={s:.6f}")
            print(f"    real[{idx}] = -2*({i0:.6f}*{s:.6f} + {r0:.6f}*{c:.6f}) = {real_val:.6f}")
            print(f"    imag[{idx}] = -2*({i0:.6f}*{c:.6f} - {r0:.6f}*{s:.6f}) = {imag_val:.6f}")
    
    print(f"\nPre-FFT results:")
    print(f"  real array: {real[:8]} ... (max: {np.max(np.abs(real))})")
    print(f"  imag array: {imag[:8]} ... (max: {np.max(np.abs(imag))})")
    print(f"  real energy: {np.sum(real**2)}")
    print(f"  imag energy: {np.sum(imag**2)}")
    
    # Check if pre-FFT arrays are near zero
    if np.max(np.abs(real)) < 1e-6 and np.max(np.abs(imag)) < 1e-6:
        print("  ❌ PRE-FFT ARRAYS ARE NEAR ZERO - FOUND THE BUG!")
        return
    
    # FFT stage
    complex_input = real + 1j * imag
    fft_result = np.fft.fft(complex_input)
    real_fft = fft_result.real.astype(np.float32)
    imag_fft = fft_result.imag.astype(np.float32)
    
    print(f"\nFFT results:")
    print(f"  real_fft: {real_fft[:8]} ... (max: {np.max(np.abs(real_fft))})")
    print(f"  imag_fft: {imag_fft[:8]} ... (max: {np.max(np.abs(imag_fft))})")

def debug_coefficient_symmetry():
    """Check if MDCT coefficients have symmetry that causes cancellation."""
    
    print("\n=== Coefficient Symmetry Analysis ===")
    
    # Test 256-point
    size = 256
    test_input = np.ones(size, dtype=np.float32)
    mdct256 = MDCT(size, 0.5)
    coeffs = mdct256(test_input)
    
    n2 = len(coeffs)  # 128
    
    print(f"Analyzing {size}-point MDCT coefficients...")
    print(f"Coefficient array length: {n2}")
    
    # Check symmetry/anti-symmetry
    print(f"\nChecking for symmetry patterns:")
    
    symmetric_pairs = []
    antisymmetric_pairs = []
    
    for k in range(0, n2, 2):
        mirror_k = n2 - 1 - k
        r0 = coeffs[k]
        i0 = coeffs[mirror_k]
        
        if k < mirror_k:  # Avoid double-checking
            ratio = i0 / r0 if abs(r0) > 1e-10 else float('inf')
            
            if abs(ratio - 1.0) < 0.1:  # Nearly symmetric
                symmetric_pairs.append((k, mirror_k, r0, i0, ratio))
            elif abs(ratio + 1.0) < 0.1:  # Nearly anti-symmetric
                antisymmetric_pairs.append((k, mirror_k, r0, i0, ratio))
            
            if k < 8:  # Print first few pairs
                print(f"  k={k}, mirror_k={mirror_k}: coeffs[{k}]={r0:.6f}, coeffs[{mirror_k}]={i0:.6f}, ratio={ratio:.3f}")
    
    print(f"\nSymmetric pairs (ratio ≈ 1): {len(symmetric_pairs)}")
    print(f"Anti-symmetric pairs (ratio ≈ -1): {len(antisymmetric_pairs)}")
    
    if len(symmetric_pairs) > len(coeffs) // 4:
        print("  ⚠️  High symmetry detected - may cause cancellation in pre-FFT twiddle")
    if len(antisymmetric_pairs) > len(coeffs) // 4:
        print("  ⚠️  High anti-symmetry detected - may cause cancellation in pre-FFT twiddle")

if __name__ == "__main__":
    debug_imdct_full_vs_synthetic()
    debug_imdct_internals()
    debug_coefficient_symmetry()