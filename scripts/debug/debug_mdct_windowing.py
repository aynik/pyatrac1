#!/usr/bin/env python3
"""
Debug the MDCT windowing stage to find where energy is lost.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def debug_mdct_windowing_step_by_step():
    """Debug each step of MDCT windowing to find energy loss."""
    
    print("=== MDCT Windowing Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Test input
    test_input = np.ones(128, dtype=np.float32) * 0.5
    input_energy = np.sum(test_input**2)
    print(f"Input: DC=0.5, energy={input_energy:.6f}")
    
    # Initialize buffers like the real MDCT does
    mdct.initialize_windowing_state(channel=0)
    
    # Copy input into persistent buffer (what mdct() does)
    mdct.pcm_buf_low[0][:128] = test_input
    print(f"After copying to persistent buffer:")
    print(f"  pcm_buf_low[0][:128] energy: {np.sum(mdct.pcm_buf_low[0][:128]**2):.6f}")
    print(f"  pcm_buf_low[0][128:] energy: {np.sum(mdct.pcm_buf_low[0][128:]**2):.6f}")
    
    # Simulate the windowing process for LOW band (like mdct() does)
    persistent_low = mdct.pcm_buf_low[0]
    src_buf = persistent_low
    buf_sz = 128
    block_sz = 128  # Long block
    win_start = 48  # For long blocks
    tmp = mdct.tmp_buffers[0][0]  # LOW band tmp buffer
    
    print(f"\nWindowing parameters:")
    print(f"  buf_sz={buf_sz}, block_sz={block_sz}, win_start={win_start}")
    print(f"  src_buf energy: {np.sum(src_buf**2):.6f}")
    
    # Step 1: Copy tail to tmp (line 409 in mdct.py)
    tmp[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]
    print(f"\nStep 1 - Copy tail to tmp:")
    print(f"  tmp[{win_start}:{win_start+32}] = src_buf[{buf_sz}:{buf_sz+32}]")
    print(f"  Values copied: {src_buf[buf_sz:buf_sz + 32][:8]}...")
    print(f"  tmp energy after copy: {np.sum(tmp**2):.6f}")
    
    # Step 2: Apply sine windowing (lines 414-418)
    print(f"\nStep 2 - Apply sine windowing:")
    orig_values = src_buf[0:block_sz].copy()
    print(f"  Original src_buf[0:{block_sz}] energy: {np.sum(orig_values**2):.6f}")
    
    for i in range(32):
        idx1 = buf_sz + i  # 128 + i
        idx2 = 0 + block_sz - 32 + i  # 0 + 128 - 32 + i = 96 + i
        print(f"    i={i}: idx1={idx1}, idx2={idx2}")
        print(f"      Before: src_buf[{idx1}]={src_buf[idx1]:.6f}, src_buf[{idx2}]={src_buf[idx2]:.6f}")
        src_buf[idx1] = mdct.SINE_WINDOW[i] * src_buf[idx2]
        src_buf[idx2] = mdct.SINE_WINDOW[31 - i] * src_buf[idx2]
        print(f"      After:  src_buf[{idx1}]={src_buf[idx1]:.6f}, src_buf[{idx2}]={src_buf[idx2]:.6f}")
        if i >= 4:  # Only show first few
            break
    
    # Complete the windowing
    for i in range(32):
        idx1 = buf_sz + i
        idx2 = 0 + block_sz - 32 + i
        src_buf[idx1] = mdct.SINE_WINDOW[i] * src_buf[idx2]
        src_buf[idx2] = mdct.SINE_WINDOW[31 - i] * src_buf[idx2]
    
    print(f"  src_buf energy after windowing: {np.sum(src_buf**2):.6f}")
    print(f"  Windowing region [96:128] energy: {np.sum(src_buf[96:128]**2):.6f}")
    print(f"  Windowing region [128:160] energy: {np.sum(src_buf[128:160]**2):.6f}")
    
    # Step 3: Copy windowed data to tmp (line 425)
    tmp[win_start + 32:win_start + 32 + block_sz] = src_buf[0:block_sz]
    print(f"\nStep 3 - Copy windowed data to tmp:")
    print(f"  tmp[{win_start+32}:{win_start+32+block_sz}] = src_buf[0:{block_sz}]")
    print(f"  tmp energy after final copy: {np.sum(tmp**2):.6f}")
    
    # Step 4: Feed to MDCT engine
    print(f"\nStep 4 - MDCT transform:")
    print(f"  tmp input to MDCT:")
    print(f"    First 8: {tmp[:8]}")
    print(f"    Range [48:56]: {tmp[48:56]}")
    print(f"    Range [80:88]: {tmp[80:88]}")
    print(f"    Last 8: {tmp[-8:]}")
    
    mdct_output = mdct.mdct256(tmp)
    print(f"  MDCT output energy: {np.sum(mdct_output**2):.6f}")
    print(f"  MDCT DC coefficient: {mdct_output[0]:.6f}")
    
    # Compare with direct test
    print(f"\n=== Direct MDCT Test (no windowing) ===")
    direct_input = np.zeros(256, dtype=np.float32)
    direct_input[:128] = test_input
    direct_output = mdct.mdct256(direct_input)
    print(f"Direct MDCT input energy: {np.sum(direct_input**2):.6f}")
    print(f"Direct MDCT output energy: {np.sum(direct_output**2):.6f}")
    print(f"Direct MDCT DC coefficient: {direct_output[0]:.6f}")
    
    # The issue might be in how we prepare the tmp buffer
    print(f"\n=== tmp Buffer Analysis ===")
    print(f"tmp buffer layout:")
    print(f"  [0:48]: {np.sum(tmp[0:48]**2):.6f} energy (should be zeros)")
    print(f"  [48:80]: {np.sum(tmp[48:80]**2):.6f} energy (copied tail)")
    print(f"  [80:208]: {np.sum(tmp[80:208]**2):.6f} energy (windowed main data)")
    print(f"  [208:256]: {np.sum(tmp[208:256]**2):.6f} energy (should be zeros)")

def test_atracdenc_windowing_pattern():
    """Test if our windowing matches atracdenc pattern exactly."""
    
    print(f"\n=== atracdenc Windowing Pattern Test ===")
    
    # Based on atracdenc source, the windowing should work like this:
    # 1. tmpBuf[winStart:winStart+32] = srcBuf[bufSz:bufSz+32] (tail copy)
    # 2. Apply sine window to srcBuf[blockPos+blockSz-32:blockPos+blockSz]
    # 3. tmpBuf[winStart+32:winStart+32+blockSz] = srcBuf[blockPos:blockPos+blockSz]
    
    mdct = Atrac1MDCT()
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    # Manual atracdenc-style windowing
    src_buf = np.zeros(256 + 16, dtype=np.float32)  # Buffer with overlap
    src_buf[:128] = test_input  # New data
    src_buf[128:] = 0.0  # Previous frame overlap (zeros for first frame)
    
    tmp_buf = np.zeros(256, dtype=np.float32)
    
    # atracdenc parameters for LOW band, long block
    buf_sz = 128
    block_sz = 128
    win_start = 48
    block_pos = 0
    
    print(f"atracdenc windowing:")
    print(f"  buf_sz={buf_sz}, block_sz={block_sz}, win_start={win_start}, block_pos={block_pos}")
    
    # Step 1: Copy tail (atracdenc line equivalent)
    tmp_buf[win_start:win_start + 32] = src_buf[buf_sz:buf_sz + 32]
    print(f"  Step 1: Copy tail [128:160] → tmp[48:80]")
    print(f"    Values: {src_buf[buf_sz:buf_sz + 32][:8]}...")
    
    # Step 2: Apply windowing to the last 32 samples of current block
    for i in range(32):
        idx = block_pos + block_sz - 32 + i  # 0 + 128 - 32 + i = 96 + i
        src_buf[buf_sz + i] = mdct.SINE_WINDOW[i] * src_buf[idx]  # Window into tail
        src_buf[idx] = mdct.SINE_WINDOW[31 - i] * src_buf[idx]    # Window in-place
    
    print(f"  Step 2: Apply windowing to src_buf[96:128]")
    print(f"    Windowed values [96:104]: {src_buf[96:104]}")
    print(f"    Windowed tail [128:136]: {src_buf[128:136]}")
    
    # Step 3: Copy windowed block to tmp
    tmp_buf[win_start + 32:win_start + 32 + block_sz] = src_buf[block_pos:block_pos + block_sz]
    print(f"  Step 3: Copy windowed block → tmp[80:208]")
    
    print(f"\nResulting tmp buffer:")
    print(f"  Energy: {np.sum(tmp_buf**2):.6f}")
    print(f"  Non-zero regions:")
    print(f"    [48:80]: {np.sum(tmp_buf[48:80]**2):.6f}")
    print(f"    [80:208]: {np.sum(tmp_buf[80:208]**2):.6f}")
    
    # Test with MDCT
    mdct_result = mdct.mdct256(tmp_buf)
    print(f"  MDCT result energy: {np.sum(mdct_result**2):.6f}")
    print(f"  MDCT DC: {mdct_result[0]:.6f}")

if __name__ == "__main__":
    debug_mdct_windowing_step_by_step()
    test_atracdenc_windowing_pattern()
    
    print(f"\n=== WINDOWING DEBUG SUMMARY ===")
    print("Look for:")
    print("1. Energy loss during windowing steps")
    print("2. Incorrect tmp buffer preparation")
    print("3. Window function application errors")
    print("4. Buffer layout mismatches with atracdenc")