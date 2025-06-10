#!/usr/bin/env python3
"""
Debug vector_fmul_window TDAC implementation in IMDCT.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode, vector_fmul_window

def debug_vector_fmul_window_isolated():
    """Test vector_fmul_window function in isolation."""
    
    print("=== vector_fmul_window Isolated Test ===")
    
    # Create simple test data
    length = 4  # Small test for easy verification
    
    src0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)     # Previous frame data
    src1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)    # Current frame data (reversed access)
    win = np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32)   # Window function
    dst = np.zeros(length * 2, dtype=np.float32)               # Output buffer
    
    print(f"Inputs:")
    print(f"  src0: {src0}")
    print(f"  src1: {src1}")  
    print(f"  win:  {win}")
    print(f"  length: {length}")
    
    # Apply vector_fmul_window
    vector_fmul_window(dst, src0, src1, win, length)
    
    print(f"\nOutput:")
    print(f"  dst: {dst}")
    
    # Manual calculation to verify
    print(f"\nManual verification:")
    expected_dst = np.zeros(length * 2, dtype=np.float32)
    
    for i in range(length):
        j = length - 1 - i
        s0 = src0[i]
        s1 = src1[j]
        wi = win[i]
        wj = win[j]
        
        expected_dst[i] = s0 * wj - s1 * wi
        expected_dst[length + j] = s0 * wi + s1 * wj
        
        print(f"  i={i}, j={j}: s0={s0}, s1={s1}, wi={wi}, wj={wj}")
        print(f"    dst[{i}] = {s0} * {wj} - {s1} * {wi} = {expected_dst[i]}")
        print(f"    dst[{length + j}] = {s0} * {wi} + {s1} * {wj} = {expected_dst[length + j]}")
    
    print(f"  Expected dst: {expected_dst}")
    
    # Check if our implementation matches
    if np.allclose(dst, expected_dst):
        print("  ✅ vector_fmul_window implementation is correct")
    else:
        print("  ❌ vector_fmul_window implementation is WRONG")
        print(f"     Differences: {dst - expected_dst}")

def debug_imdct_vector_fmul_usage():
    """Debug how vector_fmul_window is used in IMDCT context."""
    
    print(f"\n=== IMDCT vector_fmul_window Usage Debug ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)  # Long blocks
    
    # Create test data for two consecutive frames to test overlap-add
    print("Testing overlap-add with two consecutive frames...")
    
    # Frame 1: DC signal
    low1 = np.ones(128, dtype=np.float32) * 0.5
    mid1 = np.zeros(128, dtype=np.float32)
    hi1 = np.zeros(256, dtype=np.float32)
    
    specs1 = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs1, low1, mid1, hi1, block_size_mode, channel=0, frame=0)
    
    print(f"Frame 1 - MDCT coeffs energy: {np.sum(specs1**2):.6f}")
    
    # IMDCT Frame 1 with debug output
    low_out1 = np.zeros(256, dtype=np.float32)
    mid_out1 = np.zeros(256, dtype=np.float32)
    hi_out1 = np.zeros(512, dtype=np.float32)
    
    # We need to manually step through IMDCT to debug vector_fmul_window
    print(f"\nManual IMDCT for Frame 1 (low band only):")
    
    # Focus on low band (band=0) processing
    band = 0
    num_mdct_blocks = 1  # Long blocks
    buf_sz = 128
    block_sz = 128
    dst_buf = low_out1
    
    # Get MDCT coefficients for low band (first 128 coeffs)
    low_coeffs = specs1[:128]
    print(f"Low coeffs energy: {np.sum(low_coeffs**2):.6f}")
    print(f"Low coeffs[0:4]: {low_coeffs[:4]}")
    
    # Run IMDCT on these coefficients
    imdct128 = mdct.imdct256  # 128-point produces 256 samples
    inv = imdct128(low_coeffs)
    
    print(f"IMDCT raw output: length={len(inv)}, energy={np.sum(inv**2):.6f}")
    print(f"IMDCT raw[0:4]: {inv[:4]}")
    print(f"IMDCT raw[252:256]: {inv[252:256]}")
    
    # Extract middle half (what gets stored in inv_buf)
    inv_len = len(inv)
    middle_start = inv_len // 4      # 64
    middle_length = inv_len // 2     # 128
    inv_buf_section = inv[middle_start:middle_start + middle_length]
    
    print(f"Middle half extraction: start={middle_start}, length={middle_length}")
    print(f"inv_buf_section: length={len(inv_buf_section)}, energy={np.sum(inv_buf_section**2):.6f}")
    print(f"inv_buf_section[0:4]: {inv_buf_section[:4]}")
    print(f"inv_buf_section[124:128]: {inv_buf_section[124:128]}")
    
    # For first frame, prev_buf should be zeros (from dst_buf[buf_sz*2-16:])
    prev_buf_start = buf_sz * 2 - 16  # 256 - 16 = 240
    prev_buf = dst_buf[prev_buf_start:prev_buf_start + 16]
    
    print(f"prev_buf from dst_buf[{prev_buf_start}:{prev_buf_start + 16}]: {prev_buf}")
    
    # Apply vector_fmul_window for first block
    temp_dst = np.zeros(32, dtype=np.float32)
    temp_src1 = inv_buf_section[:16]  # First 16 samples of middle half
    
    print(f"vector_fmul_window inputs:")
    print(f"  prev_buf (src0): {prev_buf}")
    print(f"  temp_src1 (src1): {temp_src1}")
    print(f"  sine_window: {mdct.SINE_WINDOW}")
    
    vector_fmul_window(temp_dst, prev_buf, temp_src1, np.array(mdct.SINE_WINDOW, dtype=np.float32), 16)
    
    print(f"vector_fmul_window output: {temp_dst}")
    
    # This should be copied to dst_buf[0:32]
    dst_buf[:32] = temp_dst
    
    print(f"dst_buf[0:32] after vector_fmul_window: {dst_buf[:32]}")
    
    # For long blocks, copy middle section
    length = 112  # For low band long block
    dst_buf[32:32 + length] = inv_buf_section[16:16 + length]
    
    print(f"dst_buf[32:144] after middle copy: {dst_buf[32:36]} ... {dst_buf[140:144]}")
    
    # Update tail for next frame
    for j in range(16):
        dst_buf[buf_sz * 2 - 16 + j] = inv_buf_section[buf_sz - 16 + j]
    
    tail_region = dst_buf[buf_sz * 2 - 16:]
    print(f"Updated tail dst_buf[{buf_sz * 2 - 16}:]: {tail_region}")
    
    print(f"\nFinal Frame 1 output:")
    print(f"  Energy: {np.sum(dst_buf**2):.6f}")
    print(f"  Max: {np.max(np.abs(dst_buf)):.6f}")
    print(f"  dst_buf[30:34]: {dst_buf[30:34]}")  # Around overlap boundary

def test_frame_to_frame_overlap():
    """Test overlap-add between consecutive frames."""
    
    print(f"\n=== Frame-to-Frame Overlap Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)
    
    # Process two frames and check if TDAC works
    frames = [
        np.ones(128, dtype=np.float32) * 0.5,   # Frame 1: DC
        np.ones(128, dtype=np.float32) * 0.7,   # Frame 2: Different DC
    ]
    
    frame_outputs = []
    
    for frame_idx, frame_input in enumerate(frames):
        print(f"\nProcessing Frame {frame_idx + 1}:")
        
        # MDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, frame_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32), 
                 block_size_mode, channel=0, frame=frame_idx)
        
        # IMDCT
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        frame_outputs.append(low_out.copy())
        
        print(f"  Input: {frame_input[0]} (constant)")
        print(f"  Output energy: {np.sum(low_out**2):.6f}")
        print(f"  Output max: {np.max(np.abs(low_out)):.6f}")
        print(f"  Output[30:34]: {low_out[30:34]}")  # Check overlap region
        
        # Check the persistent tail state
        tail_start = 256 - 16
        tail_state = low_out[tail_start:]
        print(f"  Tail state for next frame: {tail_state[:4]} ... {tail_state[-4:]}")
    
    # Analyze overlap between frames
    print(f"\nOverlap Analysis:")
    
    # The tail of frame 1 should become the prev_buf for frame 2
    frame1_tail = frame_outputs[0][240:256]  # Last 16 samples
    frame2_start = frame_outputs[1][:32]     # First 32 samples (overlap region)
    
    print(f"Frame 1 tail (becomes prev_buf): {frame1_tail[:4]} ... {frame1_tail[-4:]}")
    print(f"Frame 2 start (overlap result): {frame2_start[:4]} ... {frame2_start[-4:]}")
    
    # For TDAC to work, there should be smooth transition
    # Check if overlap region shows signs of proper windowing
    overlap_variation = np.std(frame2_start)
    print(f"Frame 2 overlap variation (std): {overlap_variation:.6f}")
    
    if overlap_variation < 0.1:
        print("  ✅ Low variation suggests good overlap-add")
    else:
        print("  ⚠️  High variation suggests overlap-add issues")

if __name__ == "__main__":
    debug_vector_fmul_window_isolated()
    debug_imdct_vector_fmul_usage()
    test_frame_to_frame_overlap()