#!/usr/bin/env python3
"""
Debug windowing implementation details vs atracdenc.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def debug_windowing_step_by_step():
    """Debug windowing step by step to find differences."""
    
    print("=== Windowing Step-by-Step Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Use simple test data to trace windowing
    low_input = np.arange(128, dtype=np.float32) / 128.0  # 0, 1/128, 2/128, ..., 127/128
    
    print("Test input (linear ramp):")
    print(f"  First 8: {low_input[:8]}")
    print(f"  Last 8: {low_input[-8:]}")
    
    # Manually trace the windowing operation
    # First, copy input to persistent buffer
    mdct.pcm_buf_low[0][:128] = low_input
    
    print(f"\nAfter copying to persistent buffer:")
    print(f"  pcm_buf[120:128]: {mdct.pcm_buf_low[0][120:128]}")
    print(f"  pcm_buf[128:136]: {mdct.pcm_buf_low[0][128:136]}")  # Should be zeros initially
    
    # Now manually apply windowing like the MDCT method does
    # Based on MDCT code:
    # - band = 0 (low band)
    # - num_mdct_blocks = 1 (long blocks)  
    # - buf_sz = 128
    # - block_sz = 128
    # - win_start = 48 (for long blocks, non-high band)
    
    src_buf = mdct.pcm_buf_low[0]
    buf_sz = 128
    block_sz = 128  
    win_start = 48
    tmp = mdct.tmp_buffers[0][0]
    block_pos = 0
    
    print(f"\nWindowing parameters:")
    print(f"  buf_sz={buf_sz}, block_sz={block_sz}, win_start={win_start}, block_pos={block_pos}")
    
    # Step 1: tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]
    print(f"\nStep 1: Store overlap data")
    print(f"  tmp[{win_start}:{win_start+32}] = src_buf[{buf_sz}:{buf_sz+32}]")
    print(f"  Source data: {src_buf[buf_sz:buf_sz+32][:4]} ...")  # Should be zeros
    
    tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]
    
    print(f"  Result: tmp[{win_start}:{win_start+32}] = {tmp[win_start:win_start+32][:4]} ...")
    
    # Step 2: Apply windowing
    print(f"\nStep 2: Apply sine windowing")
    print(f"  Sine window first 4: {mdct.SINE_WINDOW[:4]}")
    print(f"  Sine window last 4: {mdct.SINE_WINDOW[-4:]}")
    
    # Store original values before windowing
    orig_end = src_buf[block_pos + block_sz - 32:block_pos + block_sz].copy()
    orig_overlap = src_buf[buf_sz:buf_sz + 32].copy()
    
    print(f"  Before windowing:")
    print(f"    src_buf[{block_pos + block_sz - 32}:{block_pos + block_sz}]: {orig_end[:4]} ... {orig_end[-4:]}")
    print(f"    src_buf[{buf_sz}:{buf_sz + 32}]: {orig_overlap[:4]} ... {orig_overlap[-4:]}")
    
    # Apply windowing (this modifies src_buf in place!)
    for i in range(32):
        idx1 = buf_sz + i
        idx2 = block_pos + block_sz - 32 + i
        
        # Before modification
        old_val1 = src_buf[idx1]
        old_val2 = src_buf[idx2]
        
        # Apply windowing
        src_buf[idx1] = mdct.SINE_WINDOW[i] * src_buf[idx2]
        src_buf[idx2] = mdct.SINE_WINDOW[31 - i] * src_buf[idx2]
        
        if i < 4:  # Show first few operations
            print(f"    i={i}: idx1={idx1}, idx2={idx2}")
            print(f"      src_buf[{idx1}]: {old_val1:.6f} -> {src_buf[idx1]:.6f} (= {mdct.SINE_WINDOW[i]:.6f} * {old_val2:.6f})")
            print(f"      src_buf[{idx2}]: {old_val2:.6f} -> {src_buf[idx2]:.6f} (= {mdct.SINE_WINDOW[31-i]:.6f} * {old_val2:.6f})")
    
    print(f"\n  After windowing:")
    print(f"    src_buf[{block_pos + block_sz - 32}:{block_pos + block_sz}]: {src_buf[block_pos + block_sz - 32:block_pos + block_sz][:4]} ... {src_buf[block_pos + block_sz - 32:block_pos + block_sz][-4:]}")
    print(f"    src_buf[{buf_sz}:{buf_sz + 32}]: {src_buf[buf_sz:buf_sz + 32][:4]} ... {src_buf[buf_sz:buf_sz + 32][-4:]}")
    
    # Step 3: Copy windowed data to tmp buffer
    print(f"\nStep 3: Copy windowed data to tmp")
    print(f"  tmp[{win_start + 32}:{win_start + 32 + block_sz}] = src_buf[{block_pos}:{block_pos + block_sz}]")
    
    tmp[win_start + 32:win_start + 32 + block_sz] = src_buf[block_pos:block_pos + block_sz]
    
    print(f"  tmp buffer sections:")
    print(f"    tmp[{win_start}:{win_start+8}]: {tmp[win_start:win_start+8]}")  # Overlap data
    print(f"    tmp[{win_start+32}:{win_start+40}]: {tmp[win_start+32:win_start+40]}")  # Start of windowed data
    print(f"    tmp[{win_start+32+120}:{win_start+32+128}]: {tmp[win_start+32+120:win_start+32+128]}")  # End of windowed data

def check_sine_window():
    """Check if our sine window matches atracdenc."""
    
    print(f"\n=== Sine Window Check ===")
    
    mdct = Atrac1MDCT()
    
    print("Our sine window:")
    print(f"  Length: {len(mdct.SINE_WINDOW)}")
    print(f"  First 8: {mdct.SINE_WINDOW[:8]}")
    print(f"  Last 8: {mdct.SINE_WINDOW[-8:]}")
    
    # Check if it's monotonic increasing (should be for sine)
    is_increasing = all(mdct.SINE_WINDOW[i] <= mdct.SINE_WINDOW[i+1] for i in range(len(mdct.SINE_WINDOW)-1))
    print(f"  Monotonic increasing: {is_increasing}")
    
    # Check boundary values
    print(f"  First value: {mdct.SINE_WINDOW[0]:.6f} (should be close to 0)")
    print(f"  Last value: {mdct.SINE_WINDOW[-1]:.6f} (should be close to 1)")
    
    # Verify the calculation
    expected_values = []
    for i in range(32):
        expected = np.sin((i + 0.5) * (np.pi / (2.0 * 32.0)))
        expected_values.append(expected)
    
    print(f"  Expected first 4: {expected_values[:4]}")
    print(f"  Actual first 4: {mdct.SINE_WINDOW[:4]}")
    
    if np.allclose(mdct.SINE_WINDOW, expected_values):
        print("  ✅ Sine window calculation is correct")
    else:
        print("  ❌ Sine window calculation is WRONG")
        max_diff = np.max(np.abs(np.array(mdct.SINE_WINDOW) - np.array(expected_values)))
        print(f"     Max difference: {max_diff}")

def test_buffer_state_corruption():
    """Test if buffer state gets corrupted between calls."""
    
    print(f"\n=== Buffer State Corruption Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)
    
    # First call with known data
    low1 = np.ones(128, dtype=np.float32) * 0.5
    mid1 = np.zeros(128, dtype=np.float32)
    hi1 = np.zeros(256, dtype=np.float32)
    
    specs1 = np.zeros(512, dtype=np.float32)
    
    print("First MDCT call:")
    print(f"  Input low[0]: {low1[0]}")
    
    # Capture buffer state before
    buf_before = mdct.pcm_buf_low[0].copy()
    
    mdct.mdct(specs1, low1, mid1, hi1, block_size_mode, channel=0, frame=0)
    
    # Capture buffer state after
    buf_after = mdct.pcm_buf_low[0].copy()
    
    print(f"  Buffer changes:")
    print(f"    Input region [0:8]: {buf_before[:8]} -> {buf_after[:8]}")
    print(f"    Overlap region [128:136]: {buf_before[128:136]} -> {buf_after[128:136]}")
    
    # Second call with same data
    print(f"\nSecond MDCT call (same data):")
    
    buf_before_2 = mdct.pcm_buf_low[0].copy()
    
    mdct.mdct(specs1, low1, mid1, hi1, block_size_mode, channel=0, frame=1)
    
    buf_after_2 = mdct.pcm_buf_low[0].copy()
    
    print(f"  Buffer changes:")
    print(f"    Input region [0:8]: {buf_before_2[:8]} -> {buf_after_2[:8]}")
    print(f"    Overlap region [128:136]: {buf_before_2[128:136]} -> {buf_after_2[128:136]}")
    
    # Check if the input region is the same both times
    input_consistent = np.allclose(buf_after[:128], buf_after_2[:128])
    overlap_changed = not np.allclose(buf_after[128:], buf_after_2[128:])
    
    print(f"  Input region consistent: {input_consistent}")
    print(f"  Overlap region changed: {overlap_changed}")
    
    if not input_consistent:
        print("  ❌ Input region was corrupted between calls!")
    if not overlap_changed:
        print("  ⚠️  Overlap region didn't change (unexpected for windowing)")

if __name__ == "__main__":
    debug_windowing_step_by_step()
    check_sine_window()
    test_buffer_state_corruption()