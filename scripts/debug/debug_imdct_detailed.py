#!/usr/bin/env python3
"""
Debug IMDCT implementation in detail to find why middle section is zeros.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def debug_imdct_buffer_extraction():
    """Debug what happens in IMDCT buffer extraction."""
    
    print("=== Detailed IMDCT Buffer Extraction Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Create simple DC test case
    low_input = np.ones(128, dtype=np.float32) * 0.5
    mid_input = np.zeros(128, dtype=np.float32)
    hi_input = np.zeros(256, dtype=np.float32)
    
    print("Test: DC signal on low band")
    print(f"Input: {low_input[:4]} ... (all 0.5)")
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, mid_input, hi_input, 
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    # Get low band coefficients
    low_coeffs = specs[:128]
    print(f"MDCT coeffs: {low_coeffs[:8]}")
    print(f"MDCT energy: {np.sum(low_coeffs**2):.6f}")
    
    # Manually call IMDCT256 to see raw output
    print(f"\n=== Raw IMDCT256 Analysis ===")
    
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"Raw IMDCT length: {len(raw_imdct)}")
    print(f"Raw IMDCT energy: {np.sum(raw_imdct**2):.6f}")
    print(f"Raw IMDCT max: {np.max(np.abs(raw_imdct)):.6f}")
    
    # Show different sections of raw IMDCT
    print(f"\nRaw IMDCT sections:")
    print(f"  First quarter [0:64]: {raw_imdct[:8]} ... energy={np.sum(raw_imdct[:64]**2):.6f}")
    print(f"  Second quarter [64:128]: {raw_imdct[64:72]} ... energy={np.sum(raw_imdct[64:128]**2):.6f}")
    print(f"  Third quarter [128:192]: {raw_imdct[128:136]} ... energy={np.sum(raw_imdct[128:192]**2):.6f}")
    print(f"  Fourth quarter [192:256]: {raw_imdct[192:200]} ... energy={np.sum(raw_imdct[192:256]**2):.6f}")
    
    # Test our extraction
    print(f"\n=== Buffer Extraction Analysis ===")
    
    inv_len = len(raw_imdct)  # 256
    middle_start = inv_len // 4    # 64
    middle_length = inv_len // 2   # 128
    
    print(f"Extraction parameters:")
    print(f"  inv_len: {inv_len}")
    print(f"  middle_start: {middle_start}")
    print(f"  middle_length: {middle_length}")
    print(f"  Extracting: inv[{middle_start}:{middle_start + middle_length}]")
    
    extracted = raw_imdct[middle_start:middle_start + middle_length]
    print(f"Extracted section: {extracted[:8]} ... energy={np.sum(extracted**2):.6f}")
    
    # Compare with other possible extractions
    print(f"\nAlternative extractions:")
    alt1 = raw_imdct[:128]  # First half
    print(f"  First half [0:128]: {alt1[:8]} ... energy={np.sum(alt1**2):.6f}")
    
    alt2 = raw_imdct[128:]  # Second half  
    print(f"  Second half [128:256]: {alt2[:8]} ... energy={np.sum(alt2**2):.6f}")
    
    alt3 = raw_imdct[32:160]  # Middle 128 samples (different center)
    print(f"  Alt middle [32:160]: {alt3[:8]} ... energy={np.sum(alt3**2):.6f}")
    
    # Test if there's a better extraction point
    print(f"\nSearching for best extraction window:")
    best_energy = 0
    best_start = 0
    for start in range(0, inv_len - 128, 8):
        window = raw_imdct[start:start + 128]
        energy = np.sum(window**2)
        if energy > best_energy:
            best_energy = energy
            best_start = start
        print(f"  [{start}:{start+128}]: energy={energy:.6f}")
    
    print(f"Best extraction: [{best_start}:{best_start+128}] with energy={best_energy:.6f}")
    
    if best_start != middle_start:
        print(f"❌ Current extraction is suboptimal!")
        best_window = raw_imdct[best_start:best_start + 128]
        print(f"Better extraction: {best_window[:8]} ...")
    else:
        print(f"✅ Current extraction is optimal")

def debug_mdct_roundtrip():
    """Debug full MDCT->IMDCT roundtrip with different extraction strategies."""
    
    print(f"\n=== MDCT Roundtrip with Different Extractions ===")
    
    mdct = Atrac1MDCT()
    
    # Test cases
    test_cases = [
        ("DC", np.ones(128, dtype=np.float32) * 0.5),
        ("Ramp", np.arange(128, dtype=np.float32) / 128.0),
        ("Sine", np.sin(2 * np.pi * np.arange(128) / 16).astype(np.float32)),
    ]
    
    for name, test_input in test_cases:
        print(f"\nTest case: {name}")
        print(f"Input energy: {np.sum(test_input**2):.6f}")
        
        # MDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                 BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_coeffs = specs[:128]
        print(f"MDCT energy: {np.sum(low_coeffs**2):.6f}")
        
        # Raw IMDCT
        raw_imdct = mdct.imdct256(low_coeffs)
        
        # Test different extractions
        extractions = [
            ("Current (middle half)", raw_imdct[64:192]),
            ("First half", raw_imdct[:128]),
            ("Second half", raw_imdct[128:]),
            ("Alt middle", raw_imdct[32:160]),
        ]
        
        for ext_name, extracted in extractions:
            # Simulate what happens in TDAC reconstruction
            # Use middle portion (skip overlap regions)
            useful_start = 16
            useful_end = useful_start + 96  # Skip overlap on both sides
            useful_section = extracted[useful_start:useful_end]
            
            if len(useful_section) > 0 and len(test_input) >= len(useful_section):
                input_section = test_input[:len(useful_section)]
                correlation = np.corrcoef(useful_section, input_section)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                energy = np.sum(useful_section**2)
                
                print(f"  {ext_name}: energy={energy:.6f}, correlation={correlation:.4f}")

if __name__ == "__main__":
    debug_imdct_buffer_extraction()
    debug_mdct_roundtrip()