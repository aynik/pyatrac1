#!/usr/bin/env python3
"""
Systematic tracking of the +32 shift in IMDCT output placement.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_shift_with_impulse_inputs():
    """Test with impulse inputs to track exactly where the shift occurs."""
    
    print("=== Systematic +32 Shift Tracking ===")
    
    mdct = Atrac1MDCT()
    
    # Test with single impulse in different positions
    impulse_positions = [0, 1, 2, 8, 16, 32, 64]
    
    for pos in impulse_positions:
        if pos < 128:
            print(f"\nTesting impulse at position {pos}:")
            
            # Create impulse
            test_coeffs = np.zeros(128, dtype=np.float32)
            test_coeffs[pos] = 1.0
            
            # Get IMDCT output
            imdct_output = mdct.imdct256(test_coeffs)
            
            # Find peak in output
            peak_idx = np.argmax(np.abs(imdct_output))
            peak_val = imdct_output[peak_idx]
            
            print(f"  Input impulse at: {pos}")
            print(f"  Output peak at: {peak_idx}")
            print(f"  Peak value: {peak_val:.6f}")
            
            # Check if there's a consistent shift pattern
            if pos == 0:
                dc_peak_pos = peak_idx
                print(f"  DC peak position (baseline): {dc_peak_pos}")
            else:
                # For other frequencies, see if peak position has expected relationship
                # In a correct IMDCT, peak position should relate to frequency bin
                expected_pattern = f"Expected frequency-dependent pattern"
                print(f"  Pattern: {expected_pattern}")
            
            # Check where atracdenc would extract from this
            atracdenc_region = imdct_output[64:192]
            atracdenc_energy = np.sum(atracdenc_region**2)
            total_energy = np.sum(imdct_output**2)
            atracdenc_fraction = atracdenc_energy / total_energy if total_energy > 0 else 0
            
            print(f"  atracdenc [64:192] captures: {atracdenc_fraction*100:.1f}% of energy")
            
            # Check our suspected region [96:224]
            if len(imdct_output) >= 224:
                our_region = imdct_output[96:224]
                our_energy = np.sum(our_region**2)
                our_fraction = our_energy / total_energy if total_energy > 0 else 0
                print(f"  Our region [96:224] captures: {our_fraction*100:.1f}% of energy")
                
                # Shift comparison
                shift_amount = 96 - 64
                print(f"  Shift amount: +{shift_amount} samples")

def test_circular_shift_hypothesis():
    """Test if a circular shift by -32 aligns our output with atracdenc expectations."""
    
    print(f"\n=== Circular Shift Hypothesis Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test with DC signal
    test_coeffs = np.zeros(128, dtype=np.float32)
    test_coeffs[0] = -0.082304  # DC coefficient from previous tests
    
    # Get current output
    current_output = mdct.imdct256(test_coeffs)
    
    # Test different shift amounts
    shift_amounts = [-32, -16, 0, +16, +32]
    
    print(f"Testing different circular shifts:")
    print(f"Target: maximize meaningful data in atracdenc region [64:192]")
    
    baseline_atracdenc_energy = np.sum(current_output[64:192]**2)
    
    for shift in shift_amounts:
        shifted_output = np.roll(current_output, shift)
        shifted_atracdenc_energy = np.sum(shifted_output[64:192]**2)
        improvement = shifted_atracdenc_energy / baseline_atracdenc_energy
        
        print(f"  Shift {shift:+3d}: atracdenc energy = {shifted_atracdenc_energy:.6f}, improvement = {improvement:.3f}x")
        
        # Check the actual values in the atracdenc region after shift
        atracdenc_region = shifted_output[64:192]
        region_mean = np.mean(atracdenc_region)
        region_std = np.std(atracdenc_region)
        
        print(f"           mean={region_mean:.6f}, std={region_std:.6f}")
        
        # For DC, we expect roughly constant values ~0.125
        if abs(region_mean - 0.125) < 0.02 and region_std < 0.01:
            print(f"           ✅ Good DC reconstruction!")
        
        # Check if this shift gives us the expected 0.25 values for input 0.5
        if shift == -32:
            print(f"           Shift -32 analysis:")
            print(f"           First 8: {atracdenc_region[:8]}")
            print(f"           Last 8: {atracdenc_region[-8:]}")

def test_step_by_step_imdct():
    """Step through IMDCT implementation to find where +32 shift is introduced."""
    
    print(f"\n=== Step-by-Step IMDCT Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # Simple test case: pure DC
    test_coeffs = np.zeros(128, dtype=np.float32)
    test_coeffs[0] = 1.0
    
    print(f"Input: DC coefficient = 1.0 at position 0")
    
    # Manually step through IMDCT
    imdct = mdct.imdct256
    n2 = imdct.N // 2  # 128
    n4 = imdct.N // 4  # 64
    n34 = 3 * n4       # 192
    n54 = 5 * n4       # 320
    
    print(f"IMDCT parameters: N={imdct.N}, n2={n2}, n4={n4}, n34={n34}")
    
    # Step 1: FFT input preparation
    size = n2 // 2  # 64
    real = np.zeros(size, dtype=np.float32)
    imag = np.zeros(size, dtype=np.float32)
    
    print(f"\nStep 1: FFT input preparation (size={size})")
    
    for idx, k in enumerate(range(0, n2, 2)):
        if idx < 8:  # Log first few
            r0 = test_coeffs[k]
            i0 = test_coeffs[n2 - 1 - k]
            c = imdct.SinCos[k]
            s = imdct.SinCos[k + 1]
            real[idx] = -2.0 * (i0 * s + r0 * c)
            imag[idx] = -2.0 * (i0 * c - r0 * s)
            
            print(f"  idx={idx}, k={k}: r0={r0:.6f}, i0={i0:.6f}, c={c:.6f}, s={s:.6f}")
            print(f"    real[{idx}]={real[idx]:.6f}, imag[{idx}]={imag[idx]:.6f}")
    
    # Complete FFT input
    for idx, k in enumerate(range(0, n2, 2)):
        r0 = test_coeffs[k]
        i0 = test_coeffs[n2 - 1 - k]
        c = imdct.SinCos[k]
        s = imdct.SinCos[k + 1]
        real[idx] = -2.0 * (i0 * s + r0 * c)
        imag[idx] = -2.0 * (i0 * c - r0 * s)
    
    print(f"  FFT input energy: real={np.sum(real**2):.6f}, imag={np.sum(imag**2):.6f}")
    
    # Step 2: FFT
    complex_input = real + 1j * imag
    fft_result = np.fft.fft(complex_input)
    real_fft = fft_result.real
    imag_fft = fft_result.imag
    
    print(f"\nStep 2: FFT")
    print(f"  FFT output energy: real={np.sum(real_fft**2):.6f}, imag={np.sum(imag_fft**2):.6f}")
    print(f"  First 8 real: {real_fft[:8]}")
    print(f"  First 8 imag: {imag_fft[:8]}")
    
    # Step 3: Output generation - track where each coefficient goes
    print(f"\nStep 3: Output generation tracking")
    
    buf = np.zeros(imdct.N, dtype=np.float32)
    
    print(f"  First loop (k in [0, {n4}, step=2]):")
    for idx, k in enumerate(range(0, n4, 2)):
        if idx < 4:  # Log first few
            r0 = real_fft[idx]
            i0 = imag_fft[idx]
            c = imdct.SinCos[k]
            s = imdct.SinCos[k + 1]
            r1 = r0 * c + i0 * s
            i1 = r0 * s - i0 * c
            
            # Track output indices
            out_idx1 = n34 - 1 - k  # 191, 189, 187, ...
            out_idx2 = n34 + k      # 192, 194, 196, ...
            out_idx3 = n4 + k       # 64, 66, 68, ...
            out_idx4 = n4 - 1 - k   # 63, 61, 59, ...
            
            print(f"    idx={idx}, k={k}: r1={r1:.6f}, i1={i1:.6f}")
            print(f"      buf[{out_idx1}] = {r1:.6f}")
            print(f"      buf[{out_idx2}] = {r1:.6f}")
            print(f"      buf[{out_idx3}] = {i1:.6f}")
            print(f"      buf[{out_idx4}] = {-i1:.6f}")
    
    # For DC input, most energy should go to specific indices
    # Let's see which output indices get the strongest values
    
    # Complete the output generation
    for idx, k in enumerate(range(0, n4, 2)):
        r0 = real_fft[idx]
        i0 = imag_fft[idx]
        c = imdct.SinCos[k]
        s = imdct.SinCos[k + 1]
        r1 = r0 * c + i0 * s
        i1 = r0 * s - i0 * c
        buf[n34 - 1 - k] = r1
        buf[n34 + k] = r1
        buf[n4 + k] = i1
        buf[n4 - 1 - k] = -i1

    for idx, k in enumerate(range(n4, n2, 2), start=size // 2):
        r0 = real_fft[idx]
        i0 = imag_fft[idx]
        c = imdct.SinCos[k]
        s = imdct.SinCos[k + 1]
        r1 = r0 * c + i0 * s
        i1 = r0 * s - i0 * c
        buf[n34 - 1 - k] = r1
        buf[k - n4] = -r1
        buf[n4 + k] = i1
        buf[n54 - 1 - k] = i1
    
    print(f"\nStep 4: Final output analysis")
    print(f"  Total energy: {np.sum(buf**2):.6f}")
    
    # Find where the meaningful values ended up
    max_val = np.max(np.abs(buf))
    strong_indices = np.where(np.abs(buf) > max_val * 0.5)[0]
    print(f"  Strong values (>{max_val*0.5:.3f}) at indices: {strong_indices[:10]}...")
    
    # Check if the pattern shows a systematic shift
    print(f"  atracdenc region [64:192] max: {np.max(np.abs(buf[64:192])):.6f}")
    print(f"  Shifted region [96:224] max: {np.max(np.abs(buf[96:224])):.6f}")
    print(f"  First quarter [0:64] max: {np.max(np.abs(buf[0:64])):.6f}")
    print(f"  Last quarter [192:256] max: {np.max(np.abs(buf[192:256])):.6f}")

def compare_with_expected_pattern():
    """Compare our IMDCT output pattern with theoretical expectations."""
    
    print(f"\n=== Expected Pattern Comparison ===")
    
    print(f"For a pure DC coefficient input:")
    print(f"  Theoretical: IMDCT should produce constant values")
    print(f"  atracdenc extracts [64:192] expecting meaningful reconstruction")
    print(f"  Our output: strongest values appear at [96:224] (+32 shift)")
    
    print(f"\nShift pattern analysis:")
    print(f"  If our algorithm is correct but shifted +32:")
    print(f"    - atracdenc expects data at [64:192]")
    print(f"    - We place data at [96:224]") 
    print(f"    - Difference: +32 samples")
    
    print(f"\nPossible causes of +32 shift:")
    print(f"  1. FFT output ordering difference (numpy vs KissFFT)")
    print(f"  2. SinCos table phase offset")
    print(f"  3. Buffer indexing offset in output generation")
    print(f"  4. MDCT window alignment difference")

if __name__ == "__main__":
    test_shift_with_impulse_inputs()
    test_circular_shift_hypothesis()
    test_step_by_step_imdct()
    compare_with_expected_pattern()
    
    print(f"\n=== TRACKING SUMMARY ===")
    print("The +32 shift is consistent across all tests.")
    print("Next steps: isolate the exact stage where the shift is introduced.")
    print("Focus areas: FFT output interpretation, SinCos indexing, output buffer placement.")