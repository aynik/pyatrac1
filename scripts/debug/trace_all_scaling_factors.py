#!/usr/bin/env python3
"""
Comprehensive tracing of all scaling factors in the MDCT/IMDCT/QMF pipeline
to understand the 4x scaling issue.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def trace_mdct_scaling():
    """Trace all MDCT scaling factors step by step."""
    
    print("=== MDCT Scaling Factor Tracing ===")
    
    mdct = Atrac1MDCT()
    
    # Parameters
    print(f"MDCT256 constructor scale: {mdct.mdct256.Scale}")
    print(f"calc_sin_cos scale factor: √({mdct.mdct256.Scale}/256) = {np.sqrt(mdct.mdct256.Scale/256):.6f}")
    
    # MDCT multiple factor
    print(f"MDCT multiple for LOW band (long block): 1.0")
    print(f"MDCT multiple for HIGH band (short blocks): 2.0")
    
    # Test DC input
    test_input = np.ones(128, dtype=np.float32) * 0.5
    input_energy = np.sum(test_input**2)
    print(f"\nInput: DC=0.5, energy={input_energy:.6f}")
    
    # Trace through MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    coeffs_energy = np.sum(low_coeffs**2)
    dc_coeff = low_coeffs[0]
    
    print(f"MDCT output: DC={dc_coeff:.6f}, energy={coeffs_energy:.6f}")
    print(f"MDCT energy scaling factor: {coeffs_energy/input_energy:.6f}")
    print(f"MDCT amplitude scaling factor: {dc_coeff/(0.5*128):.6f}")

def trace_imdct_scaling():
    """Trace all IMDCT scaling factors step by step."""
    
    print(f"\n=== IMDCT Scaling Factor Tracing ===")
    
    mdct = Atrac1MDCT()
    
    # Parameters
    print(f"IMDCT256 constructor scale: {mdct.imdct256.Scale}")
    print(f"calc_sin_cos scale factor: √({mdct.imdct256.Scale}/256) = {np.sqrt(mdct.imdct256.Scale/256):.6f}")
    print(f"IMDCT -2.0 factor in pre-FFT: -2.0")
    
    # Test with known coefficients
    test_coeffs = np.zeros(128, dtype=np.float32)
    test_coeffs[0] = -0.082304  # Known DC coefficient from MDCT
    
    coeffs_energy = np.sum(test_coeffs**2)
    print(f"\nInput coeffs: DC={test_coeffs[0]:.6f}, energy={coeffs_energy:.6f}")
    
    # Direct IMDCT test
    imdct_output = mdct.imdct256(test_coeffs)
    imdct_energy = np.sum(imdct_output**2)
    meaningful_region = imdct_output[64:192]
    meaningful_energy = np.sum(meaningful_region**2)
    meaningful_mean = np.mean(meaningful_region)
    
    print(f"IMDCT raw output energy: {imdct_energy:.6f}")
    print(f"IMDCT meaningful region [64:192]: energy={meaningful_energy:.6f}, mean={meaningful_mean:.6f}")
    print(f"IMDCT energy scaling factor: {meaningful_energy/coeffs_energy:.6f}")

def trace_qmf_scaling():
    """Trace QMF analysis/synthesis scaling factors."""
    
    print(f"\n=== QMF Scaling Factor Analysis ===")
    
    # Note: This requires access to QMF implementation
    # For now, document expected QMF scaling behavior
    
    print("QMF Analysis/Synthesis expected behavior:")
    print("- QMF analysis typically divides energy by number of subbands")
    print("- QMF synthesis should restore energy if designed for perfect reconstruction")
    print("- Unity gain QMF: analysis gain × synthesis gain = 1.0")
    print("- Common designs: analysis /√2, synthesis ×√2 per subband")
    
    # Check if we can infer QMF scaling from our pipeline
    mdct = Atrac1MDCT()
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    # Full pipeline test
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    final_output = low_out[32:160]  # Account for +32 QMF delay
    final_mean = np.mean(final_output)
    
    print(f"\nFull pipeline result:")
    print(f"Input mean: 0.5")
    print(f"Output mean: {final_mean:.6f}")
    print(f"Total pipeline scaling factor: {final_mean/0.5:.6f}")

def calculate_theoretical_scaling():
    """Calculate theoretical scaling factors from parameters."""
    
    print(f"\n=== Theoretical Scaling Calculation ===")
    
    mdct = Atrac1MDCT()
    
    # MDCT scaling chain
    mdct_scale = mdct.mdct256.Scale  # 0.5
    mdct_sincos_scale = np.sqrt(mdct_scale / 256)  # √(0.5/256)
    mdct_multiple = 1.0  # For LOW band long block
    
    print(f"MDCT scaling chain:")
    print(f"  Constructor scale: {mdct_scale}")
    print(f"  SinCos scale: {mdct_sincos_scale:.6f}")
    print(f"  Multiple factor: {mdct_multiple}")
    print(f"  Total MDCT scaling: {mdct_sincos_scale * mdct_multiple:.6f}")
    
    # IMDCT scaling chain
    imdct_scale = mdct.imdct256.Scale  # 256
    imdct_sincos_scale = np.sqrt(imdct_scale / 256)  # √(256/256) = 1.0
    imdct_minus_two = -2.0
    
    print(f"\nIMDCT scaling chain:")
    print(f"  Constructor scale: {imdct_scale}")
    print(f"  SinCos scale: {imdct_sincos_scale:.6f}")
    print(f"  -2.0 factor: {imdct_minus_two}")
    print(f"  Total IMDCT scaling: {imdct_sincos_scale * abs(imdct_minus_two):.6f}")
    
    # Combined theoretical scaling
    theoretical_scaling = (mdct_sincos_scale * mdct_multiple) * (imdct_sincos_scale * abs(imdct_minus_two))
    print(f"\nTheoretical MDCT+IMDCT scaling: {theoretical_scaling:.6f}")
    
    # What should perfect reconstruction be?
    print(f"\nPerfect reconstruction expectation:")
    print(f"  For unity gain: scaling should be 1.0")
    print(f"  Our theoretical: {theoretical_scaling:.6f}")
    print(f"  Deviation from unity: {theoretical_scaling:.6f} ({theoretical_scaling:.1f}x)")

def test_dc_coefficient_bypass():
    """Test bypassing quantization with known DC coefficient."""
    
    print(f"\n=== DC Coefficient Bypass Test ===")
    
    mdct = Atrac1MDCT()
    
    # Known DC coefficient (from MDCT)
    dc_coeff = -0.082304
    
    # Create specs with only DC coefficient
    specs = np.zeros(512, dtype=np.float32)
    specs[0] = dc_coeff  # Set only DC, others zero
    
    print(f"Input specs: DC={specs[0]:.6f}, all others zero")
    
    # Direct IMDCT
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    # Analyze result
    final_output = low_out[32:160]
    final_mean = np.mean(final_output)
    final_std = np.std(final_output)
    
    print(f"IMDCT+QMF result:")
    print(f"  Mean: {final_mean:.6f}")
    print(f"  Std: {final_std:.6f}")
    print(f"  Expected mean for DC 0.5 input: 0.5")
    print(f"  Actual scaling factor: {final_mean * (0.5 / 0.125) if abs(final_mean) > 1e-6 else 0:.6f}")

if __name__ == "__main__":
    trace_mdct_scaling()
    trace_imdct_scaling()
    trace_qmf_scaling()
    calculate_theoretical_scaling()
    test_dc_coefficient_bypass()
    
    print(f"\n=== SCALING ANALYSIS SUMMARY ===")
    print("Key findings:")
    print("1. MDCT SinCos scale: very small (√(0.5/256) ≈ 0.044)")
    print("2. IMDCT SinCos scale: 1.0 (√(256/256))")
    print("3. IMDCT -2.0 factor contributes to scaling")
    print("4. Need to verify QMF analysis/synthesis scaling")
    print("5. The 4x scaling likely comes from cumulative factors")
    print("\nNext steps:")
    print("- Compare with atracdenc reference output")
    print("- Test with different frequency signals")
    print("- Verify QMF filter design assumptions")