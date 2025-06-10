#!/usr/bin/env python3
"""
Debug the "torch passing" of overlap data between QMF->MDCT and IMDCT->QMF.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_overlap_preservation():
    """Test if overlap data is properly preserved between calls."""
    
    print("=== Overlap Preservation Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)  # Long blocks
    
    # Frame 1: Create distinctive data patterns
    low1 = np.arange(128, dtype=np.float32) / 128.0  # 0 to 0.99
    mid1 = np.ones(128, dtype=np.float32) * 0.5
    hi1 = np.ones(256, dtype=np.float32) * 0.3
    
    specs1 = np.zeros(512, dtype=np.float32)
    
    print("Frame 1 processing:")
    print(f"Input low[120:128]: {low1[120:128]}")
    
    # Process frame 1
    mdct.mdct(specs1, low1, mid1, hi1, block_size_mode, channel=0, frame=0)
    
    print(f"After frame 1:")
    print(f"  pcm_buf_low[0][120:128]: {mdct.pcm_buf_low[0][120:128]}")
    print(f"  pcm_buf_low[0][128:136]: {mdct.pcm_buf_low[0][128:136]}")
    
    # Store the overlap state after frame 1
    overlap_after_frame1 = mdct.pcm_buf_low[0][128:].copy()
    
    # Frame 2: Different data
    low2 = np.ones(128, dtype=np.float32) * 0.8
    mid2 = np.ones(128, dtype=np.float32) * 0.6  
    hi2 = np.ones(256, dtype=np.float32) * 0.4
    
    specs2 = np.zeros(512, dtype=np.float32)
    
    print(f"\nFrame 2 processing:")
    print(f"Input low2[0:8]: {low2[0:8]}")
    print(f"Overlap before frame 2: {mdct.pcm_buf_low[0][128:136]}")
    
    # Process frame 2
    mdct.mdct(specs2, low2, mid2, hi2, block_size_mode, channel=0, frame=1)
    
    print(f"After frame 2:")
    print(f"  pcm_buf_low[0][0:8]: {mdct.pcm_buf_low[0][0:8]}")  # Should be new input
    print(f"  pcm_buf_low[0][120:128]: {mdct.pcm_buf_low[0][120:128]}")  # End of new data
    print(f"  pcm_buf_low[0][128:136]: {mdct.pcm_buf_low[0][128:136]}")  # New overlap
    
    # Check if the new input was correctly placed
    if np.allclose(mdct.pcm_buf_low[0][:128], low2):
        print("  ✅ Frame 2 input correctly overwrites frame 1 input area")
    else:
        print("  ❌ Frame 2 input NOT correctly placed")
        print(f"    Expected: {low2[0:4]}")
        print(f"    Got: {mdct.pcm_buf_low[0][0:4]}")
    
    # Check if the overlap changed appropriately
    overlap_after_frame2 = mdct.pcm_buf_low[0][128:].copy()
    overlap_changed = not np.allclose(overlap_after_frame1, overlap_after_frame2)
    
    print(f"  Overlap changed: {overlap_changed}")
    if overlap_changed:
        print("  ✅ Overlap state properly updated")
    else:
        print("  ⚠️  Overlap state did not change (may be issue)")

def test_mdct_imdct_roundtrip():
    """Test MDCT->IMDCT roundtrip with persistent buffers."""
    
    print("\n=== MDCT->IMDCT Roundtrip Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)
    
    # Simple test data
    low_in = np.ones(128, dtype=np.float32) * 0.5
    mid_in = np.zeros(128, dtype=np.float32) 
    hi_in = np.zeros(256, dtype=np.float32)
    
    print("Input data:")
    print(f"  Low: {low_in[0]} (constant)")
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_in, mid_in, hi_in, block_size_mode, channel=0, frame=0)
    
    print(f"MDCT output energy: {np.sum(specs**2):.6f}")
    print(f"MDCT max coeff: {np.max(np.abs(specs)):.6f}")
    
    # Inverse MDCT with proper buffer sizes
    # Based on the QMF synthesis error, use sizes that match expectations
    low_out = np.zeros(256, dtype=np.float32)  # QMF expects this size
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(256, dtype=np.float32)   # QMF expects 256, not 512
    
    try:
        mdct.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=0)
        
        print(f"IMDCT output energy: {np.sum(low_out**2):.6f}")
        print(f"IMDCT max: {np.max(np.abs(low_out)):.6f}")
        
        # Check reconstruction quality
        # For a DC input, we expect some specific pattern
        dc_region = low_out[32:224]  # Avoid edges
        dc_mean = np.mean(dc_region)
        dc_std = np.std(dc_region)
        
        print(f"DC region stats: mean={dc_mean:.4f}, std={dc_std:.4f}")
        
        # Rough quality check
        if dc_std < 0.1 * abs(dc_mean):
            print("  ✅ Output appears to be roughly constant (good for DC input)")
        else:
            print("  ⚠️  Output has high variation (may indicate aliasing)")
            
    except Exception as e:
        print(f"  ❌ IMDCT failed: {e}")

def test_buffer_size_mismatch():
    """Test what happens with different buffer sizes."""
    
    print("\n=== Buffer Size Mismatch Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)
    
    # Test what buffer sizes the IMDCT actually expects/produces
    specs = np.ones(512, dtype=np.float32) * 0.1  # Small test signal
    
    buffer_size_tests = [
        (128, 128, 128),   # Too small?
        (256, 256, 256),   # QMF expected size
        (256, 256, 512),   # Mixed sizes
        (512, 512, 512),   # Large
    ]
    
    for low_sz, mid_sz, hi_sz in buffer_size_tests:
        print(f"\nTesting buffer sizes: Low={low_sz}, Mid={mid_sz}, Hi={hi_sz}")
        
        low_out = np.zeros(low_sz, dtype=np.float32)
        mid_out = np.zeros(mid_sz, dtype=np.float32)
        hi_out = np.zeros(hi_sz, dtype=np.float32)
        
        try:
            mdct.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=0)
            print(f"  ✅ SUCCESS - Max outputs: Low={np.max(np.abs(low_out)):.4f}, "
                  f"Mid={np.max(np.abs(mid_out)):.4f}, Hi={np.max(np.abs(hi_out)):.4f}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

if __name__ == "__main__":
    test_overlap_preservation()
    test_mdct_imdct_roundtrip()
    test_buffer_size_mismatch()