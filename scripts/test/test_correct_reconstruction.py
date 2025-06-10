#!/usr/bin/env python3
"""
Test the corrected understanding of MDCT/IMDCT reconstruction.
"""

import numpy as np
from pyatrac1.core.mdct import MDCT, IMDCT

def test_correct_reconstruction():
    """Test MDCT/IMDCT reconstruction looking at the right metrics."""
    
    print("=== Corrected Reconstruction Test ===")
    
    sizes_and_scales = [
        (64, 0.5, "low/mid", 1/4),
        (256, 0.5, "low/mid", 1/4), 
        (512, 1, "high", 1/2)
    ]
    
    for size, mdct_scale, band_type, expected_factor in sizes_and_scales:
        print(f"\n{size}-point MDCT/IMDCT ({band_type}):")
        
        test_input = np.ones(size, dtype=np.float32)
        
        mdct = MDCT(size, mdct_scale)
        imdct = IMDCT(size, size * 2)
        
        coeffs = mdct(test_input)
        reconstructed = imdct(coeffs)
        
        # Check different metrics
        input_energy = np.sum(test_input**2)
        output_energy = np.sum(reconstructed**2)
        energy_ratio = output_energy / input_energy
        
        # Check amplitude at different positions (not just sample 32)
        sample_positions = [0, 16, 32, 48, 64]
        sample_positions = [pos for pos in sample_positions if pos < len(reconstructed)]
        
        print(f"  Input: DC signal (all ones)")
        print(f"  Input energy: {input_energy}")
        print(f"  Output energy: {output_energy}")
        print(f"  Energy ratio: {energy_ratio}")
        print(f"  Max output amplitude: {np.max(np.abs(reconstructed))}")
        
        print(f"  Samples at different positions:")
        for pos in sample_positions:
            print(f"    reconstructed[{pos}]: {reconstructed[pos]:.6f}")
        
        # Find where the peak amplitude occurs
        max_idx = np.argmax(np.abs(reconstructed))
        max_val = reconstructed[max_idx]
        scaling_factor = max_val / test_input[0]
        
        print(f"  Peak at index {max_idx}: {max_val:.6f}")
        print(f"  Scaling factor (peak/input): {scaling_factor:.6f}")
        print(f"  Expected factor: {expected_factor:.6f}")
        
        error = abs(abs(scaling_factor) - expected_factor)
        
        if error < 0.01:
            print(f"  ✅ PASS - Scaling within tolerance")
        else:
            print(f"  ❌ FAIL - Scaling error: {error:.6f}")
            
        # Check if output is roughly constant (for DC input)
        dc_samples = reconstructed[32:96] if len(reconstructed) > 96 else reconstructed[16:-16]
        if len(dc_samples) > 0:
            dc_variation = np.std(dc_samples)
            dc_mean = np.mean(dc_samples)
            print(f"  DC region mean: {dc_mean:.6f}, std: {dc_variation:.6f}")

def test_different_input_types():
    """Test with different input types to verify reconstruction."""
    
    print("\n=== Different Input Types Test ===")
    
    # Test 256-point with different inputs
    size = 256
    mdct = MDCT(size, 0.5)
    imdct = IMDCT(size, size * 2)
    
    test_cases = [
        ("DC (all ones)", np.ones(size, dtype=np.float32)),
        ("Impulse", np.concatenate([np.array([1.0]), np.zeros(size-1)])),
        ("Sine wave", np.sin(2 * np.pi * np.arange(size) / size)),
    ]
    
    for name, test_input in test_cases:
        coeffs = mdct(test_input)
        reconstructed = imdct(coeffs)
        
        input_energy = np.sum(test_input**2)
        output_energy = np.sum(reconstructed**2)
        energy_ratio = output_energy / input_energy
        
        max_input = np.max(np.abs(test_input))
        max_output = np.max(np.abs(reconstructed))
        
        print(f"\n{name}:")
        print(f"  Input max: {max_input:.6f}, energy: {input_energy:.6f}")
        print(f"  Output max: {max_output:.6f}, energy: {output_energy:.6f}")
        print(f"  Energy ratio: {energy_ratio:.6f}")
        print(f"  Amplitude ratio: {max_output/max_input:.6f}")

if __name__ == "__main__":
    test_correct_reconstruction()
    test_different_input_types()