#!/usr/bin/env python3
"""
Trace a single impulse through the MDCT->IMDCT pipeline to find the issue.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def trace_impulse_pipeline():
    """Trace an impulse through each step of the pipeline."""
    
    print("=== Impulse Pipeline Trace ===")
    
    mdct = Atrac1MDCT()
    
    # Create impulse at position 64 (middle of input)
    impulse_pos = 64
    low_input = np.zeros(128, dtype=np.float32)
    low_input[impulse_pos] = 1.0
    
    print(f"Input: impulse at position {impulse_pos}")
    print(f"Input[60:68]: {low_input[60:68]}")
    
    # Step 1: MDCT
    print(f"\n=== Step 1: MDCT ===")
    
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"MDCT coeffs energy: {np.sum(low_coeffs**2):.6f}")
    print(f"MDCT coeffs max: {np.max(np.abs(low_coeffs)):.6f}")
    print(f"Non-zero coeffs: {np.sum(np.abs(low_coeffs) > 1e-6)}")
    print(f"MDCT coeffs[0:8]: {low_coeffs[:8]}")
    print(f"MDCT coeffs[60:68]: {low_coeffs[60:68]}")
    
    # Step 2: Raw IMDCT
    print(f"\n=== Step 2: Raw IMDCT ===")
    
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"Raw IMDCT length: {len(raw_imdct)}")
    print(f"Raw IMDCT energy: {np.sum(raw_imdct**2):.6f}")
    print(f"Raw IMDCT max: {np.max(np.abs(raw_imdct)):.6f}")
    
    # Show structure of raw IMDCT
    quarters = [
        ("First quarter [0:64]", raw_imdct[:64]),
        ("Second quarter [64:128]", raw_imdct[64:128]),  
        ("Third quarter [128:192]", raw_imdct[128:192]),
        ("Fourth quarter [192:256]", raw_imdct[192:256])
    ]
    
    max_quarter_idx = 0
    max_quarter_energy = 0
    
    for i, (name, quarter) in enumerate(quarters):
        energy = np.sum(quarter**2)
        max_val = np.max(np.abs(quarter))
        print(f"{name}: energy={energy:.6f}, max={max_val:.6f}")
        
        if energy > max_quarter_energy:
            max_quarter_energy = energy
            max_quarter_idx = i
    
    print(f"Maximum energy in: {quarters[max_quarter_idx][0]}")
    
    # Step 3: Our extraction (shifted)
    print(f"\n=== Step 3: Our Shifted Extraction ===")
    
    # Our current extraction: [80:208]
    our_extraction = raw_imdct[80:208]
    print(f"Our extraction [80:208]:")
    print(f"  Length: {len(our_extraction)}")
    print(f"  Energy: {np.sum(our_extraction**2):.6f}")
    print(f"  Max: {np.max(np.abs(our_extraction)):.6f}")
    print(f"  First 8: {our_extraction[:8]}")
    print(f"  Position 64 (original impulse): {our_extraction[64-16] if len(our_extraction) > 48 else 'N/A'}")
    
    # Compare with atracdenc extraction [64:192]
    atracdenc_extraction = raw_imdct[64:192]
    print(f"\nOriginal atracdenc extraction [64:192]:")
    print(f"  Length: {len(atracdenc_extraction)}")
    print(f"  Energy: {np.sum(atracdenc_extraction**2):.6f}")
    print(f"  Max: {np.max(np.abs(atracdenc_extraction)):.6f}")
    print(f"  First 8: {atracdenc_extraction[:8]}")
    print(f"  Position 64 (original impulse): {atracdenc_extraction[64-64] if len(atracdenc_extraction) > 0 else 'N/A'}")
    
    # Step 4: Full IMDCT pipeline
    print(f"\n=== Step 4: Full IMDCT Pipeline ===")
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    print(f"Final IMDCT output:")
    print(f"  Total energy: {np.sum(low_out**2):.6f}")
    print(f"  Max: {np.max(np.abs(low_out)):.6f}")
    
    # Analyze different regions
    regions = [
        ("Overlap start [0:32]", low_out[:32]),
        ("Middle region [32:224]", low_out[32:224]),
        ("Overlap end [224:256]", low_out[224:])
    ]
    
    for name, region in regions:
        energy = np.sum(region**2)
        max_val = np.max(np.abs(region))
        mean_val = np.mean(region)
        print(f"  {name}: energy={energy:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
    
    # Look for where our impulse ended up
    print(f"\nLooking for impulse reconstruction:")
    
    # Check around expected position (64 -> 32+32 = 64 in middle region)
    expected_pos_in_middle = 32  # Position 64 in input -> position 32 in middle region
    middle_region = low_out[32:224]
    
    if expected_pos_in_middle < len(middle_region):
        response_at_expected = middle_region[expected_pos_in_middle]
        print(f"  Response at expected position (middle[{expected_pos_in_middle}]): {response_at_expected:.6f}")
        
        # Find actual peak in middle region
        peak_idx = np.argmax(np.abs(middle_region))
        peak_val = middle_region[peak_idx]
        print(f"  Actual peak in middle: position {peak_idx}, value {peak_val:.6f}")
        
        if abs(peak_val) > 0.1:
            print(f"  ✅ Found strong peak response")
        else:
            print(f"  ❌ No strong peak found")
    
    # Step 5: Alternative extractions test
    print(f"\n=== Step 5: Alternative Extractions ===")
    
    # Test what would happen with different extraction windows
    extractions_to_test = [
        ("atracdenc [64:192]", 64, 192),
        ("our current [80:208]", 80, 208),
        ("alternative [96:224]", 96, 224),
        ("alt2 [48:176]", 48, 176),
    ]
    
    for name, start, end in extractions_to_test:
        if end <= len(raw_imdct):
            test_extraction = raw_imdct[start:end]
            test_middle = test_extraction[16:16+112] if len(test_extraction) >= 128 else test_extraction[16:]
            
            # Where would our impulse be in this extraction?
            impulse_offset_in_extraction = impulse_pos - start + 16  # +16 for middle copy offset
            
            if 0 <= impulse_offset_in_extraction < len(test_middle):
                response = test_middle[impulse_offset_in_extraction]
                print(f"  {name}: impulse response = {response:.6f}")
            else:
                print(f"  {name}: impulse outside middle region")

def test_perfect_dc_response():
    """Test with perfect DC to see if the issue is with non-constant signals."""
    
    print(f"\n=== Perfect DC Response Test ===")
    
    mdct = Atrac1MDCT()
    
    # Pure DC input
    dc_level = 1.0
    low_input = np.ones(128, dtype=np.float32) * dc_level
    
    print(f"Input: constant {dc_level}")
    
    # MDCT->IMDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    # For DC, middle region should be constant
    middle_region = low_out[32:224]
    mean_val = np.mean(middle_region)
    std_val = np.std(middle_region)
    
    print(f"Middle region analysis:")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std: {std_val:.6f}")
    print(f"  Expected: ~{dc_level * 0.25:.6f} (25% scaling)")
    print(f"  Scaling factor: {mean_val / dc_level:.6f}")
    
    # Check if constant
    if std_val < 0.01 * abs(mean_val):
        print(f"  ✅ Middle region is constant")
        
        # Check scaling
        expected_scaling = 0.25
        actual_scaling = mean_val / dc_level
        scaling_error = abs(actual_scaling - expected_scaling)
        
        if scaling_error < 0.05:
            print(f"  ✅ Scaling is correct")
        else:
            print(f"  ⚠️  Scaling error: {scaling_error:.6f}")
    else:
        print(f"  ❌ Middle region is not constant")

if __name__ == "__main__":
    trace_impulse_pipeline()
    test_perfect_dc_response()