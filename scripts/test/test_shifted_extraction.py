#!/usr/bin/env python3
"""
Test the shifted extraction fix for vector_fmul_window.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_shifted_extraction():
    """Test if the shifted extraction now provides meaningful overlap values."""
    
    print("=== Shifted Extraction Test ===")
    
    mdct = Atrac1MDCT()
    
    # Simple DC test
    low_input = np.ones(128, dtype=np.float32) * 0.5
    mid_input = np.zeros(128, dtype=np.float32)
    hi_input = np.zeros(256, dtype=np.float32)
    
    print("Input: DC signal (0.5)")
    
    # Full MDCT->IMDCT pipeline
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, mid_input, hi_input, 
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False), 
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    print(f"IMDCT output energy: {np.sum(low_out**2):.6f}")
    print(f"IMDCT output max: {np.max(np.abs(low_out)):.6f}")
    
    # Check the overlap region (first 32 samples from vector_fmul_window)
    overlap_region = low_out[:32]
    print(f"Overlap region:")
    print(f"  Values: {overlap_region[:8]} ...")
    print(f"  Max: {np.max(np.abs(overlap_region)):.6f}")
    print(f"  Energy: {np.sum(overlap_region**2):.6f}")
    
    # Check middle section (should be constant for DC input)
    middle_section = low_out[32:144]  # Skip overlap regions
    print(f"Middle section:")
    print(f"  Values: {middle_section[:8]} ...")
    print(f"  Mean: {np.mean(middle_section):.6f}")
    print(f"  Std: {np.std(middle_section):.6f}")
    print(f"  Expected: ~0.125 (0.5 * 0.25 scaling)")
    
    # Improvement assessment
    if np.max(np.abs(overlap_region)) > 0.01:
        print("  ✅ Overlap region now has meaningful values!")
    else:
        print("  ❌ Overlap region still has tiny values")
    
    if np.std(middle_section) < 0.1 * abs(np.mean(middle_section)):
        print("  ✅ Middle section is reasonably constant")
    else:
        print("  ⚠️  Middle section still varies too much")
    
    return np.max(np.abs(overlap_region))

def test_frame_to_frame_overlap():
    """Test overlap between consecutive frames with shifted extraction."""
    
    print(f"\n=== Frame-to-Frame Overlap Test ===")
    
    mdct = Atrac1MDCT()
    
    # Two consecutive frames with different DC values
    frames = [
        np.ones(128, dtype=np.float32) * 0.3,  # Frame 1
        np.ones(128, dtype=np.float32) * 0.7,  # Frame 2
    ]
    
    frame_outputs = []
    
    for frame_idx, frame_input in enumerate(frames):
        print(f"\nFrame {frame_idx + 1}:")
        print(f"  Input: constant {frame_input[0]}")
        
        # MDCT->IMDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, frame_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=frame_idx)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32) 
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        frame_outputs.append(low_out.copy())
        
        print(f"  Output energy: {np.sum(low_out**2):.6f}")
        print(f"  Overlap region [0:32]: max={np.max(np.abs(low_out[:32])):.6f}")
        print(f"  Middle section [32:144]: mean={np.mean(low_out[32:144]):.6f}")
    
    # Analyze transition
    print(f"\nFrame transition analysis:")
    frame1_tail = frame_outputs[0][240:256]  # Last 16 samples
    frame2_overlap = frame_outputs[1][:32]   # Overlap region
    
    print(f"  Frame 1 tail: {frame1_tail[:4]} ... {frame1_tail[-4:]}")
    print(f"  Frame 2 overlap: {frame2_overlap[:4]} ... {frame2_overlap[-4:]}")
    
    # For good TDAC, the overlap should show smooth transition
    overlap_variation = np.std(frame2_overlap)
    print(f"  Frame 2 overlap variation: {overlap_variation:.6f}")
    
    if overlap_variation < 0.1:
        print("  ✅ Low overlap variation suggests good TDAC")
    else:
        print("  ⚠️  High overlap variation suggests TDAC issues")

if __name__ == "__main__":
    max_overlap = test_shifted_extraction()
    test_frame_to_frame_overlap()
    
    print(f"\n=== Assessment ===")
    if max_overlap > 0.1:
        print("🎉 SHIFTED EXTRACTION FIX WORKED!")
    elif max_overlap > 0.01:
        print("✅ Shifted extraction provides meaningful values")
    else:
        print("❌ Shifted extraction still produces tiny values")