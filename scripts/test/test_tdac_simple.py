#!/usr/bin/env python3
"""
Simple TDAC test focusing on buffer indexing issues.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_simple_tdac():
    """Test MDCT->IMDCT with identity transform."""
    
    mdct_processor = Atrac1MDCT()
    
    # Create simple test pattern that should reconstruct perfectly
    # Use long blocks only for simplicity
    block_size_mode = BlockSizeMode(low_band_short=False, mid_band_short=False, high_band_short=False)
    
    # Generate simple test signals for each band
    low_samples = np.zeros(128, dtype=np.float32)
    low_samples[0] = 1.0  # Impulse at start
    
    mid_samples = np.zeros(128, dtype=np.float32) 
    mid_samples[64] = 1.0  # Impulse at middle
    
    hi_samples = np.zeros(256, dtype=np.float32)
    hi_samples[128] = 1.0  # Impulse at middle
    
    print("Testing 2-frame TDAC reconstruction...")
    
    # Frame 1: Process first frame
    specs1 = np.zeros(512, dtype=np.float32)
    mdct_processor.mdct(specs1, low_samples, mid_samples, hi_samples, block_size_mode, channel=0, frame=0)
    
    print(f"Frame 1 - MDCT coeffs (first 16): {specs1[:16]}")
    
    # Frame 1: IMDCT back 
    low_out1 = np.zeros(256 + 16, dtype=np.float32)
    mid_out1 = np.zeros(256 + 16, dtype=np.float32)
    hi_out1 = np.zeros(512 + 16, dtype=np.float32)
    
    mdct_processor.imdct(specs1, block_size_mode, low_out1, mid_out1, hi_out1, channel=0, frame=0)
    
    print(f"Frame 1 - Low band output (first 16): {low_out1[:16]}")
    print(f"Frame 1 - Low band overlap region: {low_out1[240:256]}")
    
    # Frame 2: Process second frame (same input)
    specs2 = np.zeros(512, dtype=np.float32)
    mdct_processor.mdct(specs2, low_samples, mid_samples, hi_samples, block_size_mode, channel=0, frame=1)
    
    # Frame 2: IMDCT back (should use overlap from frame 1)
    low_out2 = np.zeros(256 + 16, dtype=np.float32)
    mid_out2 = np.zeros(256 + 16, dtype=np.float32) 
    hi_out2 = np.zeros(512 + 16, dtype=np.float32)
    
    # Copy previous frame's overlap data to simulate persistent buffer
    low_out2[240:256] = low_out1[240:256]
    mid_out2[240:256] = mid_out1[240:256]
    hi_out2[496:512] = hi_out1[496:512]
    
    mdct_processor.imdct(specs2, block_size_mode, low_out2, mid_out2, hi_out2, channel=0, frame=1)
    
    print(f"Frame 2 - Low band output (first 16): {low_out2[:16]}")
    
    # Check for frame boundary continuity
    # The end of frame 1 should match the start of frame 2 after overlap-add
    frame1_end = low_out1[112:128]  # Last 16 samples of usable output
    frame2_start = low_out2[0:16]   # First 16 samples of frame 2
    
    print(f"Frame 1 end: {frame1_end}")
    print(f"Frame 2 start: {frame2_start}")
    
    boundary_error = np.mean(np.abs(frame1_end - frame2_start))
    print(f"Boundary continuity error: {boundary_error}")
    
    if boundary_error > 1e-6:
        print("❌ TDAC FAILURE: Frame boundary discontinuity detected")
        return False
    else:
        print("✅ TDAC SUCCESS: Frame boundary is continuous")
        return True

if __name__ == "__main__":
    test_simple_tdac()