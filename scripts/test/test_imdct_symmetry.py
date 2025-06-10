#!/usr/bin/env python3
"""
Test IMDCT symmetry properties for TDAC.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_imdct_tdac_property():
    """Test if IMDCT has proper TDAC symmetry properties."""
    
    print("=== IMDCT TDAC Symmetry Test ===")
    
    mdct = Atrac1MDCT()
    
    # Create impulse at DC (should reconstruct to constant signal)
    specs = np.zeros(128, dtype=np.float32)
    specs[0] = 1.0  # DC component
    
    print("Test: DC impulse (specs[0] = 1.0)")
    
    # Get raw IMDCT
    raw_imdct = mdct.imdct256(specs)
    
    print(f"Raw IMDCT length: {len(raw_imdct)}")
    print(f"Raw IMDCT energy: {np.sum(raw_imdct**2):.6f}")
    
    # For TDAC, the IMDCT should have specific symmetry properties
    # The output should be symmetric around the center with sign flip
    n = len(raw_imdct)
    
    print(f"\nTDAC symmetry analysis:")
    print(f"  N = {n}")
    print(f"  N/2 = {n//2}")
    print(f"  N/4 = {n//4}")
    
    # Check TDAC property: y[n] = -y[N-1-n] for overlap regions
    # And the middle part should contain the main signal
    
    # First quarter vs last quarter (should be anti-symmetric)
    first_quarter = raw_imdct[:n//4]
    last_quarter = raw_imdct[3*n//4:]
    last_quarter_reversed = last_quarter[::-1]
    
    print(f"\nFirst quarter: {first_quarter[:8]}")
    print(f"Last quarter (reversed): {last_quarter_reversed[:8]}")
    print(f"Anti-symmetric check: {np.allclose(first_quarter, -last_quarter_reversed)}")
    
    # Second quarter vs third quarter  
    second_quarter = raw_imdct[n//4:n//2]
    third_quarter = raw_imdct[n//2:3*n//4]
    
    print(f"\nSecond quarter: {second_quarter[:8]}")
    print(f"Third quarter: {third_quarter[:8]}")
    
    # For DC input, we expect the middle regions to contain the constant value
    # and the edges to contain the TDAC windowing
    
    print(f"\nEnergy distribution:")
    print(f"  First quarter [0:{n//4}]: {np.sum(first_quarter**2):.6f}")
    print(f"  Second quarter [{n//4}:{n//2}]: {np.sum(second_quarter**2):.6f}")
    print(f"  Third quarter [{n//2}:{3*n//4}]: {np.sum(third_quarter**2):.6f}")
    print(f"  Fourth quarter [{3*n//4}:{n}]: {np.sum(last_quarter**2):.6f}")
    
    # Check where the main signal energy is
    max_energy_quarter = np.argmax([
        np.sum(first_quarter**2),
        np.sum(second_quarter**2), 
        np.sum(third_quarter**2),
        np.sum(last_quarter**2)
    ])
    
    quarter_names = ["First", "Second", "Third", "Fourth"]
    print(f"  Maximum energy in: {quarter_names[max_energy_quarter]} quarter")
    
    # For atracdenc compatibility, check what happens with their extraction
    atracdenc_extraction = raw_imdct[n//4:3*n//4]  # Middle half [64:192]
    print(f"\natracdenc extraction [64:192]:")
    print(f"  First 8: {atracdenc_extraction[:8]}")
    print(f"  Energy: {np.sum(atracdenc_extraction**2):.6f}")
    
    # Check the offset extraction we found
    optimal_extraction = raw_imdct[80:208] if len(raw_imdct) >= 208 else raw_imdct[80:]
    print(f"\nOptimal extraction [80:208]:")
    print(f"  First 8: {optimal_extraction[:8]}")
    print(f"  Energy: {np.sum(optimal_extraction**2):.6f}")
    
    # Test different extraction strategies
    print(f"\nExtraction strategy comparison:")
    
    strategies = [
        ("atracdenc [64:192]", raw_imdct[64:192]),
        ("Optimal [80:208]", raw_imdct[80:208] if len(raw_imdct) >= 208 else raw_imdct[80:]),
        ("Alt center [48:176]", raw_imdct[48:176]),
        ("First half [0:128]", raw_imdct[:128]),
        ("Second half [128:256]", raw_imdct[128:]),
    ]
    
    for name, extraction in strategies:
        if len(extraction) >= 32:
            # Simulate overlap-add: first 16 for vector_fmul_window, middle for copy
            overlap_section = extraction[:16]
            middle_section = extraction[16:16+112] if len(extraction) >= 128 else extraction[16:]
            
            overlap_energy = np.sum(overlap_section**2)
            middle_energy = np.sum(middle_section**2)
            middle_mean = np.mean(middle_section)
            middle_std = np.std(middle_section)
            
            print(f"  {name}:")
            print(f"    Overlap energy: {overlap_energy:.6f}")
            print(f"    Middle energy: {middle_energy:.6f}, mean: {middle_mean:.6f}, std: {middle_std:.6f}")
            
            # For DC input, middle should be roughly constant
            if middle_std < 0.1 * abs(middle_mean):
                print(f"    ✅ Middle section is constant (good for DC)")
            else:
                print(f"    ❌ Middle section varies too much")

def test_imdct_with_sine():
    """Test IMDCT with sine input to see structure."""
    
    print(f"\n=== IMDCT with Sine Input ===")
    
    mdct = Atrac1MDCT()
    
    # Create sine wave input for MDCT
    sine_input = np.sin(2 * np.pi * np.arange(128) / 16).astype(np.float32)
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, sine_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"Sine MDCT coeffs energy: {np.sum(low_coeffs**2):.6f}")
    print(f"Non-zero coeffs: {np.sum(np.abs(low_coeffs) > 1e-6)}")
    
    # IMDCT 
    raw_imdct = mdct.imdct256(low_coeffs)
    
    print(f"Sine IMDCT energy: {np.sum(raw_imdct**2):.6f}")
    
    # Test extraction strategies with sine
    print(f"\nSine extraction comparison:")
    
    strategies = [
        ("atracdenc [64:192]", raw_imdct[64:192]),
        ("Optimal [80:208]", raw_imdct[80:208]),
    ]
    
    for name, extraction in strategies:
        middle_section = extraction[16:16+96]  # Avoid edges
        
        # Compare with original sine (middle portion)
        sine_middle = sine_input[16:16+len(middle_section)]
        
        if len(sine_middle) == len(middle_section):
            correlation = np.corrcoef(sine_middle, middle_section)[0, 1]
            print(f"  {name}: correlation with input = {correlation:.4f}")

if __name__ == "__main__":
    test_imdct_tdac_property()
    test_imdct_with_sine()