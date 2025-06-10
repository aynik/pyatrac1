#!/usr/bin/env python3
"""
Debug TDAC buffer state management and initialization to find off-by-one errors.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def debug_buffer_initialization():
    """Check if buffers are properly initialized to zero."""
    
    print("=== Buffer Initialization Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Check initial buffer states
    print("Initial buffer states:")
    for channel in [0, 1]:
        print(f"  Channel {channel}:")
        print(f"    pcm_buf_low: {np.sum(np.abs(mdct.pcm_buf_low[channel]))}")
        print(f"    pcm_buf_mid: {np.sum(np.abs(mdct.pcm_buf_mid[channel]))}")
        print(f"    pcm_buf_hi: {np.sum(np.abs(mdct.pcm_buf_hi[channel]))}")
        print(f"    tmp_buffers[0]: {np.sum(np.abs(mdct.tmp_buffers[channel][0]))}")
        print(f"    tmp_buffers[1]: {np.sum(np.abs(mdct.tmp_buffers[channel][1]))}")
        print(f"    tmp_buffers[2]: {np.sum(np.abs(mdct.tmp_buffers[channel][2]))}")
    
    # Check overlap regions specifically
    print("\nOverlap regions (should be zero initially):")
    for channel in [0, 1]:
        print(f"  Channel {channel}:")
        # Check if overlap regions are zero (samples beyond the main band size)
        low_overlap = mdct.pcm_buf_low[channel][128:]  # Samples 128-271 (144 samples)
        mid_overlap = mdct.pcm_buf_mid[channel][128:]  # Samples 128-271 (144 samples)  
        hi_overlap = mdct.pcm_buf_hi[channel][256:]    # Samples 256-527 (272 samples)
        
        print(f"    Low overlap sum: {np.sum(np.abs(low_overlap))}")
        print(f"    Mid overlap sum: {np.sum(np.abs(mid_overlap))}")
        print(f"    Hi overlap sum: {np.sum(np.abs(hi_overlap))}")
        
        if np.sum(np.abs(low_overlap)) == 0 and np.sum(np.abs(mid_overlap)) == 0 and np.sum(np.abs(hi_overlap)) == 0:
            print(f"    ✅ Channel {channel} overlaps properly zeroed")
        else:
            print(f"    ❌ Channel {channel} overlaps NOT zeroed")

def debug_buffer_updates():
    """Debug how buffers are updated during MDCT processing."""
    
    print("\n=== Buffer Update Debug ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)  # Long blocks
    
    # Create test data
    low_input = np.ones(128, dtype=np.float32) * 0.5
    mid_input = np.ones(128, dtype=np.float32) * 0.3
    hi_input = np.ones(256, dtype=np.float32) * 0.2
    
    specs = np.zeros(512, dtype=np.float32)
    
    print("Before first MDCT call:")
    print(f"  pcm_buf_low[0][:8]: {mdct.pcm_buf_low[0][:8]}")
    print(f"  pcm_buf_low[0][128:136]: {mdct.pcm_buf_low[0][128:136]}")
    
    # First MDCT call
    mdct.mdct(specs, low_input, mid_input, hi_input, block_size_mode, channel=0, frame=0)
    
    print("\nAfter first MDCT call:")
    print(f"  pcm_buf_low[0][:8]: {mdct.pcm_buf_low[0][:8]}")
    print(f"  pcm_buf_low[0][120:128]: {mdct.pcm_buf_low[0][120:128]}")  # End of new data
    print(f"  pcm_buf_low[0][128:136]: {mdct.pcm_buf_low[0][128:136]}")  # Start of overlap
    
    # Check if input data was copied correctly
    if np.allclose(mdct.pcm_buf_low[0][:128], low_input):
        print("  ✅ Low band input copied correctly")
    else:
        print("  ❌ Low band input NOT copied correctly")
        print(f"    Expected: {low_input[:8]}")
        print(f"    Got: {mdct.pcm_buf_low[0][:8]}")
    
    # Second MDCT call with different data
    low_input2 = np.ones(128, dtype=np.float32) * 0.7
    mid_input2 = np.ones(128, dtype=np.float32) * 0.4
    hi_input2 = np.ones(256, dtype=np.float32) * 0.1
    
    print("\nBefore second MDCT call:")
    print(f"  pcm_buf_low[0][128:136]: {mdct.pcm_buf_low[0][128:136]}")
    
    mdct.mdct(specs, low_input2, mid_input2, hi_input2, block_size_mode, channel=0, frame=1)
    
    print("\nAfter second MDCT call:")
    print(f"  pcm_buf_low[0][:8]: {mdct.pcm_buf_low[0][:8]}")
    print(f"  pcm_buf_low[0][128:136]: {mdct.pcm_buf_low[0][128:136]}")
    
    # Check if new data overwrote old correctly
    if np.allclose(mdct.pcm_buf_low[0][:128], low_input2):
        print("  ✅ Low band second input copied correctly")
    else:
        print("  ❌ Low band second input NOT copied correctly")

def debug_slice_indexing():
    """Check for off-by-one errors in critical buffer slicing."""
    
    print("\n=== Slice Indexing Debug ===")
    
    # Test the critical buffer slices used in windowing
    mdct = Atrac1MDCT()
    
    # Create test buffers with known patterns
    test_low = np.arange(272, dtype=np.float32)  # 256 + 16
    test_mid = np.arange(272, dtype=np.float32)  # 256 + 16  
    test_hi = np.arange(528, dtype=np.float32)   # 512 + 16
    
    mdct.pcm_buf_low[0] = test_low.copy()
    mdct.pcm_buf_mid[0] = test_mid.copy()
    mdct.pcm_buf_hi[0] = test_hi.copy()
    
    print("Test buffer patterns created")
    print(f"Low buffer: {test_low[:8]} ... {test_low[-8:]}")
    
    # Check specific slices that are used in windowing
    # From mdct() method:
    # tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]
    # where buf_sz=128 for low/mid, buf_sz=256 for hi
    
    print("\nCritical slice checks:")
    
    # Low band: src_buf[128:128+32] should be src_buf[128:160]
    low_slice = mdct.pcm_buf_low[0][128:160]
    expected_low = test_low[128:160]
    print(f"Low [128:160]: {low_slice[:4]} ... {low_slice[-4:]}")
    print(f"Expected:     {expected_low[:4]} ... {expected_low[-4:]}")
    print(f"  ✅ Match: {np.allclose(low_slice, expected_low)}")
    
    # Check boundary conditions
    print(f"Low buffer length: {len(mdct.pcm_buf_low[0])}")
    print(f"Valid indices: 0 to {len(mdct.pcm_buf_low[0])-1}")
    print(f"Slice [128:160] accesses indices 128 to 159")
    
    if 160 <= len(mdct.pcm_buf_low[0]):
        print("  ✅ Slice within bounds")
    else:
        print("  ❌ Slice OUT OF BOUNDS")

def debug_windowing_state():
    """Debug the windowing operation step by step."""
    
    print("\n=== Windowing State Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Create simple test data
    low_input = np.ones(128, dtype=np.float32)
    mid_input = np.zeros(128, dtype=np.float32)
    hi_input = np.zeros(256, dtype=np.float32)
    
    # Set up for long blocks
    block_size_mode = BlockSizeMode(False, False, False)
    
    # Copy input to persistent buffers (like mdct() does)
    mdct.pcm_buf_low[0][:128] = low_input
    
    print("Before windowing:")
    print(f"  Input data [120:128]: {low_input[120:128]}")
    print(f"  pcm_buf [120:128]: {mdct.pcm_buf_low[0][120:128]}")
    print(f"  pcm_buf [128:136]: {mdct.pcm_buf_low[0][128:136]}")
    
    # Simulate the windowing operation for low band
    # From the mdct() method:
    src_buf = mdct.pcm_buf_low[0]
    buf_sz = 128
    block_sz = 128  # Long block
    win_start = 48  # For long blocks, non-high band
    tmp = mdct.tmp_buffers[0][0]  # Low band tmp buffer
    
    print(f"\nWindowing parameters:")
    print(f"  buf_sz: {buf_sz}")
    print(f"  block_sz: {block_sz}")
    print(f"  win_start: {win_start}")
    
    # Step 1: tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]
    print(f"\nStep 1: tmp[{win_start}:{win_start+32}] = src_buf[{buf_sz}:{buf_sz+32}]")
    
    original_tmp = tmp[win_start:win_start + 32].copy()
    tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]
    
    print(f"  Before: tmp[{win_start}:{win_start+32}] = {original_tmp[:4]} ...")
    print(f"  Source: src_buf[{buf_sz}:{buf_sz+32}] = {src_buf[buf_sz:buf_sz+32][:4]} ...")
    print(f"  After:  tmp[{win_start}:{win_start+32}] = {tmp[win_start:win_start+32][:4]} ...")
    
    # Check if this makes sense
    if buf_sz + 32 <= len(src_buf):
        print(f"  ✅ Source slice [{buf_sz}:{buf_sz+32}] is valid")
    else:
        print(f"  ❌ Source slice [{buf_sz}:{buf_sz+32}] OUT OF BOUNDS (buffer len: {len(src_buf)})")

if __name__ == "__main__":
    debug_buffer_initialization()
    debug_buffer_updates()
    debug_slice_indexing()
    debug_windowing_state()