#!/usr/bin/env python3
"""
Fix only the MDCT/IMDCT scaling parameters to achieve better energy transfer.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode, MDCT, IMDCT

def test_different_mdct_scales():
    """Test different MDCT scale parameters to find optimal energy transfer."""
    
    print("=== MDCT Scale Parameter Testing ===")
    
    # Test input
    test_input = np.ones(128, dtype=np.float32) * 0.5
    input_energy = np.sum(test_input**2)
    print(f"Input: DC=0.5, energy={input_energy:.6f}")
    
    # Test different scale parameters for MDCT256
    scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    
    print(f"\nTesting MDCT256 scale parameters:")
    
    best_scale = None
    best_dc_coeff = 0
    
    for scale in scale_candidates:
        # Create MDCT with this scale
        mdct_engine = MDCT(256, scale)
        
        # Test with proper windowing buffer
        mdct = Atrac1MDCT()
        mdct.mdct256 = mdct_engine  # Replace with our test engine
        
        # Run full pipeline test
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_coeffs = specs[:128]
        dc_coeff = low_coeffs[0]
        coeffs_energy = np.sum(low_coeffs**2)
        
        print(f"  Scale {scale:4.2f}: DC={dc_coeff:8.6f}, energy={coeffs_energy:.6f}")
        
        if abs(dc_coeff) > abs(best_dc_coeff):
            best_dc_coeff = dc_coeff
            best_scale = scale
    
    print(f"\nBest MDCT scale: {best_scale} with DC coefficient {best_dc_coeff:.6f}")
    return best_scale

def test_different_imdct_scales(mdct_scale):
    """Test different IMDCT scale parameters to find optimal reconstruction."""
    
    print(f"\n=== IMDCT Scale Parameter Testing ===")
    
    # Use the best MDCT scale
    mdct = Atrac1MDCT()
    mdct.mdct256 = MDCT(256, mdct_scale)
    
    # Get MDCT coefficients with best scale
    test_input = np.ones(128, dtype=np.float32) * 0.5
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"MDCT coefficients: DC={low_coeffs[0]:.6f}")
    
    # Test different IMDCT scales
    imdct_scale_candidates = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\nTesting IMDCT256 scale parameters:")
    
    best_imdct_scale = None
    best_snr = -100
    
    for imdct_scale in imdct_scale_candidates:
        # Create IMDCT with this scale
        imdct_engine = IMDCT(256, imdct_scale)
        
        # Test reconstruction
        mdct.imdct256 = imdct_engine  # Replace with our test engine
        
        # Run IMDCT pipeline
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Check reconstruction quality
        final_output = low_out[32:160]
        final_mean = np.mean(final_output)
        
        # Calculate SNR
        error = final_output - test_input
        error_energy = np.sum(error**2)
        signal_energy = np.sum(test_input**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
        else:
            snr_db = float('inf')
        
        print(f"  Scale {imdct_scale:4.0f}: mean={final_mean:8.6f}, SNR={snr_db:6.2f} dB")
        
        if snr_db > best_snr:
            best_snr = snr_db
            best_imdct_scale = imdct_scale
    
    print(f"\nBest IMDCT scale: {best_imdct_scale} with SNR {best_snr:.2f} dB")
    return best_imdct_scale

def test_optimal_scale_combination():
    """Test the optimal MDCT+IMDCT scale combination."""
    
    print(f"\n=== Optimal Scale Combination Test ===")
    
    # Based on atracdenc analysis, try to match their scaling exactly
    # atracdenc: MDCT(256, 0.5), IMDCT(256*2) with internal /2 = scale 256
    
    # But let's also try some variations that might work better
    combinations = [
        ("atracdenc style", 0.5, 256),
        ("Double MDCT", 1.0, 256),
        ("Double IMDCT", 0.5, 512),
        ("Both double", 1.0, 512),
        ("Quad IMDCT", 0.5, 1024),
        ("Conservative", 0.25, 128),
    ]
    
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    print(f"\nTesting scale combinations:")
    
    best_combo = None
    best_snr = -100
    
    for name, mdct_scale, imdct_scale in combinations:
        # Create engines with these scales
        mdct = Atrac1MDCT()
        mdct.mdct256 = MDCT(256, mdct_scale)
        mdct.imdct256 = IMDCT(256, imdct_scale)
        
        # Run full pipeline
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Evaluate results
        final_output = low_out[32:160]
        final_mean = np.mean(final_output)
        final_std = np.std(final_output)
        
        error = final_output - test_input
        error_energy = np.sum(error**2)
        signal_energy = np.sum(test_input**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
        else:
            snr_db = float('inf')
        
        dc_coeff = specs[0]
        
        print(f"  {name:15s}: MDCT={mdct_scale:4.1f}, IMDCT={imdct_scale:4.0f}")
        print(f"                    DC={dc_coeff:8.6f}, mean={final_mean:8.6f}, SNR={snr_db:6.2f} dB")
        
        if snr_db > best_snr:
            best_snr = snr_db
            best_combo = (name, mdct_scale, imdct_scale)
    
    if best_combo:
        print(f"\nBest combination: {best_combo[0]} (MDCT={best_combo[1]}, IMDCT={best_combo[2]}) with SNR {best_snr:.2f} dB")
        
        if best_snr > 10:
            print(f"✅ Significant improvement achieved!")
            return best_combo[1], best_combo[2]
        else:
            print(f"⚠️  Still below target SNR (>40 dB)")
    
    return None, None

if __name__ == "__main__":
    # Step 1: Find best MDCT scale
    best_mdct_scale = test_different_mdct_scales()
    
    # Step 2: Find best IMDCT scale for that MDCT scale
    best_imdct_scale = test_different_imdct_scales(best_mdct_scale)
    
    # Step 3: Test optimal combination
    optimal_mdct, optimal_imdct = test_optimal_scale_combination()
    
    print(f"\n=== SCALING FIX SUMMARY ===")
    if optimal_mdct and optimal_imdct:
        print(f"Apply these scale parameters:")
        print(f"  self.mdct256 = MDCT(256, {optimal_mdct})")
        print(f"  self.imdct256 = IMDCT(256, {optimal_imdct})")
        print(f"This should significantly improve SNR while maintaining TDAC.")
    else:
        print(f"Current scale parameters are already optimal.")
        print(f"The energy issue must be in windowing or buffer layout.")