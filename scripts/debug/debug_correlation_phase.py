#!/usr/bin/env python3
"""
Debug correlation and phase issues in impulse reconstruction.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def debug_impulse_correlation():
    """Debug why impulse reconstruction has negative correlation."""
    
    print("=== Impulse Correlation Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Test impulse at different positions
    positions = [32, 64, 96]
    
    for pos in positions:
        print(f"\nImpulse at position {pos}:")
        
        # Create impulse
        low_input = np.zeros(128, dtype=np.float32)
        low_input[pos] = 1.0
        
        # MDCT->IMDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Analyze different output regions
        full_output = low_out
        useful_output = low_out[32:224]  # Middle region, avoid overlap
        
        print(f"  Input peak at: {pos}")
        print(f"  Input energy: {np.sum(low_input**2):.6f}")
        
        # Find peaks in output
        full_peak_idx = np.argmax(np.abs(full_output))
        full_peak_val = full_output[full_peak_idx]
        
        useful_peak_idx = np.argmax(np.abs(useful_output))
        useful_peak_val = useful_output[useful_peak_idx]
        useful_peak_abs_pos = useful_peak_idx + 32  # Adjust for offset
        
        print(f"  Full output peak: idx={full_peak_idx}, val={full_peak_val:.6f}")
        print(f"  Useful output peak: idx={useful_peak_abs_pos}, val={useful_peak_val:.6f}")
        
        # Expected position in useful output
        expected_pos_in_useful = pos - 32 if pos >= 32 else 0
        if expected_pos_in_useful < len(useful_output):
            response_at_expected = useful_output[expected_pos_in_useful]
            print(f"  Response at expected pos {pos}: {response_at_expected:.6f}")
        
        # Position error
        position_error = abs(useful_peak_abs_pos - pos)
        print(f"  Position error: {position_error} samples")
        
        if position_error <= 1:
            print("  ✅ Peak position is accurate")
        else:
            print(f"  ⚠️  Peak position shifted by {position_error} samples")
        
        # Amplitude analysis
        expected_amplitude = 1.0 * 0.269526  # Impulse * scaling factor
        amplitude_error = abs(useful_peak_val - expected_amplitude)
        
        print(f"  Expected amplitude: {expected_amplitude:.6f}")
        print(f"  Actual amplitude: {useful_peak_val:.6f}")
        print(f"  Amplitude error: {amplitude_error:.6f}")
        
        if amplitude_error < 0.05:
            print("  ✅ Amplitude is accurate")
        else:
            print("  ⚠️  Amplitude error")
        
        # Correlation analysis with proper alignment
        # Create expected output with impulse at correct position
        expected_output = np.zeros_like(useful_output)
        if expected_pos_in_useful < len(expected_output):
            expected_output[expected_pos_in_useful] = expected_amplitude
        
        # Calculate correlation
        if np.std(useful_output) > 0 and np.std(expected_output) > 0:
            correlation = np.corrcoef(useful_output, expected_output)[0, 1]
            print(f"  Correlation with ideal impulse: {correlation:.6f}")
            
            if correlation > 0.8:
                print("  ✅ High correlation")
            elif correlation > 0.3:
                print("  ⚠️  Moderate correlation")
            else:
                print("  ❌ Poor correlation")
        
        # Phase analysis - check if output is shifted version of input
        print(f"  Phase analysis:")
        
        # Look for the impulse response pattern
        # MDCT should spread impulse across frequency, IMDCT should reconstruct
        output_energy = np.sum(useful_output**2)
        input_energy = np.sum(low_input**2)
        energy_ratio = output_energy / input_energy
        
        print(f"    Energy ratio: {energy_ratio:.6f}")
        print(f"    Expected energy ratio: ~{0.269526**2:.6f}")
        
        # Check for sinc-like response (expected from MDCT->IMDCT)
        # The response should be concentrated but not exactly an impulse
        response_width = np.sum(np.abs(useful_output) > 0.01 * np.max(np.abs(useful_output)))
        print(f"    Response width (>1% peak): {response_width} samples")
        
        if response_width < 5:
            print("    ✅ Sharp impulse response")
        elif response_width < 20:
            print("    ⚠️  Spread impulse response")
        else:
            print("    ❌ Very spread response")

def debug_frequency_response():
    """Debug frequency response characteristics."""
    
    print(f"\n=== Frequency Response Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Test with different frequency sinusoids
    frequencies = [1, 4, 8, 16]  # Cycles per 128 samples
    
    for freq in frequencies:
        print(f"\nSinusoid at {freq} cycles/frame:")
        
        # Create sinusoid
        n_samples = 128
        t = np.arange(n_samples)
        low_input = np.sin(2 * np.pi * freq * t / n_samples).astype(np.float32)
        
        # MDCT->IMDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Analyze useful output region
        useful_output = low_out[32:160]  # 128 samples to match input
        
        # Energy analysis
        input_energy = np.sum(low_input**2)
        output_energy = np.sum(useful_output**2)
        energy_ratio = output_energy / input_energy
        
        print(f"  Input energy: {input_energy:.6f}")
        print(f"  Output energy: {output_energy:.6f}")
        print(f"  Energy ratio: {energy_ratio:.6f}")
        
        # Correlation analysis
        if len(useful_output) == len(low_input):
            correlation = np.corrcoef(low_input, useful_output)[0, 1]
            print(f"  Correlation: {correlation:.6f}")
            
            if correlation > 0.8:
                print("  ✅ High correlation")
            elif correlation > 0.3:
                print("  ⚠️  Moderate correlation")
            else:
                print("  ❌ Poor correlation")
        
        # Phase analysis
        # Calculate cross-correlation to find phase shift
        cross_corr = np.correlate(useful_output, low_input, mode='full')
        max_corr_idx = np.argmax(np.abs(cross_corr))
        phase_shift = max_corr_idx - (len(low_input) - 1)
        
        print(f"  Phase shift: {phase_shift} samples")
        
        if abs(phase_shift) <= 2:
            print("  ✅ Good phase alignment")
        else:
            print("  ⚠️  Phase shift detected")
        
        # Frequency domain analysis
        input_fft = np.fft.fft(low_input)
        output_fft = np.fft.fft(useful_output)
        
        # Find dominant frequency
        input_freq_idx = np.argmax(np.abs(input_fft[1:len(input_fft)//2])) + 1
        output_freq_idx = np.argmax(np.abs(output_fft[1:len(output_fft)//2])) + 1
        
        print(f"  Input dominant freq bin: {input_freq_idx}")
        print(f"  Output dominant freq bin: {output_freq_idx}")
        
        if input_freq_idx == output_freq_idx:
            print("  ✅ Frequency preserved")
        else:
            print("  ⚠️  Frequency shift")

def test_dc_perfect_reconstruction():
    """Test if DC reconstruction can be made perfect."""
    
    print(f"\n=== DC Perfect Reconstruction Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test DC reconstruction in detail
    dc_level = 1.0
    low_input = np.ones(128, dtype=np.float32) * dc_level
    
    print(f"DC level: {dc_level}")
    
    # MDCT analysis
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    dc_coeff = low_coeffs[0]
    ac_energy = np.sum(low_coeffs[1:]**2)
    
    print(f"MDCT analysis:")
    print(f"  DC coefficient: {dc_coeff:.6f}")
    print(f"  AC energy: {ac_energy:.6f}")
    print(f"  AC/DC ratio: {ac_energy / (dc_coeff**2) if dc_coeff != 0 else float('inf'):.6f}")
    
    # If DC is perfect, we should only have DC coefficient
    if ac_energy / (dc_coeff**2) < 0.01:
        print("  ✅ Mostly DC component")
    else:
        print("  ⚠️  Significant AC components")
    
    # IMDCT analysis
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    # Check different output regions
    middle_region = low_out[32:144]  # Core middle region
    
    mean_val = np.mean(middle_region)
    std_val = np.std(middle_region)
    
    print(f"IMDCT reconstruction:")
    print(f"  Middle region mean: {mean_val:.6f}")
    print(f"  Middle region std: {std_val:.6f}")
    print(f"  Relative variation: {std_val / abs(mean_val):.4f}")
    
    # The variation might be inherent to the MDCT window function
    print(f"  This suggests the variation is fundamental to MDCT windowing")
    
    # Test with DC-only coefficients
    print(f"\nTesting with pure DC coefficient:")
    
    pure_dc_specs = np.zeros(128, dtype=np.float32)
    pure_dc_specs[0] = dc_coeff  # Only DC, no AC
    
    pure_dc_imdct = mdct.imdct256(pure_dc_specs)
    
    # Our extraction
    pure_dc_extraction = pure_dc_imdct[96:224]
    pure_dc_middle = pure_dc_extraction[16:128]
    
    print(f"  Pure DC IMDCT mean: {np.mean(pure_dc_middle):.6f}")
    print(f"  Pure DC IMDCT std: {np.std(pure_dc_middle):.6f}")
    print(f"  Pure DC variation: {np.std(pure_dc_middle) / abs(np.mean(pure_dc_middle)):.4f}")
    
    if np.std(pure_dc_middle) < 1e-6:
        print("  ✅ Pure DC gives constant output")
    else:
        print("  ❌ Even pure DC has variation - inherent to IMDCT")

if __name__ == "__main__":
    debug_impulse_correlation()
    debug_frequency_response()
    test_dc_perfect_reconstruction()
    
    print(f"\n=== Correlation & Phase Summary ===")
    print("Analysis complete. Check individual test results for specific issues.")