#!/usr/bin/env python3
"""
Debug the exact scaling factors in MDCT/IMDCT pipeline to fix energy mismatch.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_scaling_factors():
    """Test scaling factors at each stage of MDCT/IMDCT pipeline."""
    
    print("=== MDCT/IMDCT Scaling Factor Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # Test with known DC input
    test_input = np.ones(128, dtype=np.float32) * 0.5
    input_energy = np.sum(test_input**2)
    print(f"Input: DC=0.5, energy={input_energy:.6f}")
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    coeffs_energy = np.sum(low_coeffs**2)
    print(f"MDCT coeffs: DC={low_coeffs[0]:.6f}, energy={coeffs_energy:.6f}")
    print(f"MDCT energy ratio: {coeffs_energy/input_energy:.6f}")
    
    # Test MDCT scaling directly
    print(f"\n=== Direct MDCT Engine Test ===")
    mdct256_engine = mdct.mdct256
    print(f"MDCT256 scale parameter: {mdct256_engine.Scale}")
    
    # Test with windowed input (what MDCT actually sees)
    windowed_input = np.zeros(256, dtype=np.float32)
    windowed_input[:128] = test_input  # First half is new data
    windowed_input[128:] = 0.0  # Second half is previous frame overlap (zeros for first frame)
    
    direct_mdct_output = mdct256_engine(windowed_input)
    direct_mdct_energy = np.sum(direct_mdct_output**2)
    print(f"Direct MDCT output: DC={direct_mdct_output[0]:.6f}, energy={direct_mdct_energy:.6f}")
    print(f"Direct MDCT energy ratio: {direct_mdct_energy/input_energy:.6f}")
    
    # Test IMDCT scaling
    print(f"\n=== Direct IMDCT Engine Test ===")
    imdct256_engine = mdct.imdct256
    print(f"IMDCT256 scale parameter: {imdct256_engine.Scale}")
    
    direct_imdct_output = imdct256_engine(low_coeffs)
    direct_imdct_energy = np.sum(direct_imdct_output**2)
    print(f"Direct IMDCT output energy: {direct_imdct_energy:.6f}")
    print(f"Direct IMDCT energy ratio: {direct_imdct_energy/coeffs_energy:.6f}")
    
    # Extract meaningful region from raw IMDCT (what atracdenc would see)
    meaningful_region = direct_imdct_output[64:192]  # atracdenc extraction
    meaningful_energy = np.sum(meaningful_region**2)
    meaningful_mean = np.mean(meaningful_region)
    print(f"Meaningful region [64:192]: energy={meaningful_energy:.6f}, mean={meaningful_mean:.6f}")
    print(f"Meaningful energy ratio to input: {meaningful_energy/input_energy:.6f}")
    
    # Full pipeline test
    print(f"\n=== Full Pipeline Test ===")
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    final_output = low_out[32:160]  # atracdenc-style extraction
    final_energy = np.sum(final_output**2)
    final_mean = np.mean(final_output)
    print(f"Final output [32:160]: energy={final_energy:.6f}, mean={final_mean:.6f}")
    print(f"Final energy ratio to input: {final_energy/input_energy:.6f}")
    
    # Calculate total energy loss/gain through pipeline
    total_ratio = final_energy / input_energy
    print(f"\nTotal pipeline energy ratio: {total_ratio:.6f} ({total_ratio:.2f}x)")
    
    # Expected scaling analysis
    print(f"\n=== Expected Scaling Analysis ===")
    print(f"Perfect reconstruction should give:")
    print(f"  Input energy: {input_energy:.6f}")
    print(f"  Output energy: {input_energy:.6f} (same)")
    print(f"  Output mean: 0.5 (same as input)")
    
    print(f"\nActual results:")
    print(f"  Output energy: {final_energy:.6f} ({final_energy/input_energy:.3f}x)")
    print(f"  Output mean: {final_mean:.6f} ({final_mean/0.5:.3f}x)")
    
    # Check if the scaling is a power of 2 (common in DSP)
    scaling_factor = final_mean / 0.5
    print(f"\nScaling factor analysis:")
    print(f"  Amplitude scaling: {scaling_factor:.6f}")
    print(f"  Is it 1/2? {abs(scaling_factor - 0.5) < 0.01}")
    print(f"  Is it 1/4? {abs(scaling_factor - 0.25) < 0.01}")
    print(f"  Is it 1/8? {abs(scaling_factor - 0.125) < 0.01}")

def test_individual_scaling_components():
    """Test each scaling component individually."""
    
    print(f"\n=== Individual Component Scaling ===")
    
    mdct = Atrac1MDCT()
    
    # Test calc_sin_cos scaling
    print(f"SinCos scaling:")
    print(f"  MDCT256 scale parameter: {mdct.mdct256.Scale}")
    print(f"  IMDCT256 scale parameter: {mdct.imdct256.Scale}")
    
    # Check if the scaling from calc_sin_cos is correct
    n = 256
    scale = mdct.mdct256.Scale
    expected_scale = np.sqrt(scale / n)
    print(f"  calc_sin_cos scale factor: √({scale}/{n}) = {expected_scale:.6f}")
    
    # Test FFT scaling (numpy FFT has no scaling by default)
    print(f"\nFFT scaling:")
    test_signal = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    fft_output = np.fft.fft(test_signal)
    print(f"  Input: {test_signal}")
    print(f"  FFT output: {fft_output}")
    print(f"  FFT preserves energy: {np.sum(np.abs(test_signal)**2):.6f} → {np.sum(np.abs(fft_output)**2):.6f}")
    
    # Test IMDCT internal scaling
    print(f"\nIMDCT internal scaling:")
    print(f"  IMDCT constructor: scale/2 = {mdct.imdct256.Scale}")
    print(f"  Original scale: {mdct.imdct256.Scale * 2}")
    
    # Check vector_fmul_window scaling
    print(f"\nWindowing scaling:")
    print(f"  SINE_WINDOW values range: [{np.min(mdct.SINE_WINDOW):.6f}, {np.max(mdct.SINE_WINDOW):.6f}]")
    print(f"  SINE_WINDOW energy: {np.sum(np.array(mdct.SINE_WINDOW)**2):.6f}")

def test_atracdenc_scaling_compatibility():
    """Test if our scaling matches atracdenc expectations."""
    
    print(f"\n=== atracdenc Scaling Compatibility ===")
    
    mdct = Atrac1MDCT()
    
    # atracdenc MDCT constructor analysis:
    # MDCT(256, 0.5) in atracdenc means scale=0.5, but calc_sin_cos uses √(scale/n)
    atracdenc_mdct_scale = 0.5
    atracdenc_n = 256
    atracdenc_sincos_scale = np.sqrt(atracdenc_mdct_scale / atracdenc_n)
    print(f"atracdenc MDCT:")
    print(f"  Constructor scale: {atracdenc_mdct_scale}")
    print(f"  SinCos scale: √({atracdenc_mdct_scale}/{atracdenc_n}) = {atracdenc_sincos_scale:.6f}")
    
    # Our implementation
    our_mdct_scale = mdct.mdct256.Scale  
    our_sincos_scale = np.sqrt(our_mdct_scale / 256)
    print(f"\nOur MDCT:")
    print(f"  Constructor scale: {our_mdct_scale}")
    print(f"  SinCos scale: √({our_mdct_scale}/256) = {our_sincos_scale:.6f}")
    
    scale_match = abs(atracdenc_sincos_scale - our_sincos_scale) < 1e-6
    print(f"  Scales match: {scale_match}")
    
    # atracdenc IMDCT: TMIDCT(256*2) → TMDCTBase(256, 256, (256*2)/2)
    atracdenc_imdct_input_scale = 256 * 2  # 512
    atracdenc_imdct_final_scale = atracdenc_imdct_input_scale / 2  # 256
    atracdenc_imdct_sincos_scale = np.sqrt(atracdenc_imdct_final_scale / 256)
    print(f"\natracdenc IMDCT:")
    print(f"  Constructor input scale: {atracdenc_imdct_input_scale}")
    print(f"  Final scale (after /2): {atracdenc_imdct_final_scale}")
    print(f"  SinCos scale: √({atracdenc_imdct_final_scale}/256) = {atracdenc_imdct_sincos_scale:.6f}")
    
    # Our IMDCT
    our_imdct_scale = mdct.imdct256.Scale
    our_imdct_sincos_scale = np.sqrt(our_imdct_scale / 256)
    print(f"\nOur IMDCT:")
    print(f"  Final scale: {our_imdct_scale}")
    print(f"  SinCos scale: √({our_imdct_scale}/256) = {our_imdct_sincos_scale:.6f}")
    
    imdct_scale_match = abs(atracdenc_imdct_sincos_scale - our_imdct_sincos_scale) < 1e-6
    print(f"  Scales match: {imdct_scale_match}")
    
    if scale_match and imdct_scale_match:
        print(f"\n✅ Our scaling parameters match atracdenc exactly!")
        print(f"The 4x amplitude issue must be elsewhere in the pipeline.")
    else:
        print(f"\n❌ Scaling mismatch found - this explains the amplitude issue!")

if __name__ == "__main__":
    test_scaling_factors()
    test_individual_scaling_components()
    test_atracdenc_scaling_compatibility()
    
    print(f"\n=== SCALING ANALYSIS SUMMARY ===")
    print("Need to identify:")
    print("1. Where the 4x amplitude reduction occurs")
    print("2. Whether it's in MDCT, IMDCT, or windowing")
    print("3. If atracdenc expects different scaling conventions")