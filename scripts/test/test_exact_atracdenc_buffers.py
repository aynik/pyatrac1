#!/usr/bin/env python3
"""
Test exact atracdenc buffer usage pattern to achieve target SNR >40dB.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_exact_atracdenc_buffer_usage():
    """Test exactly how atracdenc uses its buffers for high SNR."""
    
    print("=== Exact atracdenc Buffer Usage Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test with DC signal to understand the pattern
    test_input = np.ones(128, dtype=np.float32)
    
    print(f"Test input: DC signal (1.0)")
    print(f"Input energy: {np.sum(test_input**2):.6f}")
    
    # MDCT transform
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"MDCT DC coeff: {low_coeffs[0]:.6f}")
    
    # Get raw IMDCT output
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"Raw IMDCT energy: {np.sum(raw_imdct**2):.6f}")
    
    # Test atracdenc extraction exactly
    atracdenc_extraction = raw_imdct[64:192]  # inv[i + inv.size()/4]
    
    print(f"\natracdenc extraction analysis:")
    print(f"  Total energy: {np.sum(atracdenc_extraction**2):.6f}")
    print(f"  First 16 (overlap data): {atracdenc_extraction[:16]}")
    print(f"  First 16 energy: {np.sum(atracdenc_extraction[:16]**2):.6f}")
    print(f"  Middle 96 energy: {np.sum(atracdenc_extraction[16:112]**2):.6f}")
    print(f"  Last 16 (tail data): {atracdenc_extraction[112:128]}")
    print(f"  Last 16 energy: {np.sum(atracdenc_extraction[112:128]**2):.6f}")
    
    # The key insight: For first frame, the first 16 should be near-zero
    # because prev_buf is zeros. This is CORRECT behavior.
    # The meaningful reconstruction data is in the middle and tail.
    
    print(f"\n=== Manual IMDCT Pipeline ===")
    
    # Manually implement atracdenc IMDCT pipeline
    inv_buf = atracdenc_extraction.copy()  # This is what atracdenc extracts
    dst_buf = np.zeros(256, dtype=np.float32)
    prev_buf = np.zeros(16, dtype=np.float32)  # First frame: zeros
    
    print(f"Initial state:")
    print(f"  prev_buf: {prev_buf}")
    print(f"  inv_buf first 16: {inv_buf[:16]}")
    
    # atracdenc vector_fmul_window: dst[0:32] = f(prev_buf, inv_buf[0:16])
    from pyatrac1.core.mdct import vector_fmul_window
    
    # Create properly sized arrays for vector_fmul_window
    temp_dst = np.zeros(32, dtype=np.float32)
    temp_src1 = inv_buf[:16]
    
    vector_fmul_window(
        temp_dst,  
        prev_buf,  # src0 (zeros for first frame)
        temp_src1,  # src1 (inv_buf first 16)
        np.array(mdct.SINE_WINDOW, dtype=np.float32),
        16
    )
    
    dst_buf[:32] = temp_dst
    print(f"After vector_fmul_window:")
    print(f"  dst_buf[0:32]: {dst_buf[:32]}")
    print(f"  Energy: {np.sum(dst_buf[:32]**2):.6f}")
    
    # atracdenc memcpy: dst_buf[32:144] = inv_buf[16:128]
    dst_buf[32:144] = inv_buf[16:128]
    print(f"After middle copy:")
    print(f"  dst_buf[32:144] first 8: {dst_buf[32:40]}")
    print(f"  dst_buf[32:144] energy: {np.sum(dst_buf[32:144]**2):.6f}")
    
    # atracdenc tail update: dst_buf[240:256] = inv_buf[112:128]
    dst_buf[240:256] = inv_buf[112:128]
    print(f"After tail update:")
    print(f"  dst_buf[240:256]: {dst_buf[240:256]}")
    
    print(f"\nFinal manual reconstruction:")
    print(f"  Total energy: {np.sum(dst_buf**2):.6f}")
    print(f"  Middle region [32:160]: {dst_buf[32:160][:8]} ...")
    
    # Compare with input
    useful_output = dst_buf[32:160]  # 128 samples
    if len(useful_output) == len(test_input):
        error = useful_output - test_input
        error_energy = np.sum(error**2)
        signal_energy = np.sum(test_input**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
            print(f"  Manual reconstruction SNR: {snr_db:.2f} dB")
        else:
            print(f"  Perfect reconstruction")
    
    return dst_buf

def debug_imdct_scaling_issue():
    """Debug why our IMDCT doesn't put meaningful data in [64:192]."""
    
    print(f"\n=== IMDCT Scaling Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Simple test: pure DC coefficient should produce constant output
    print("Test: Pure DC coefficient")
    
    dc_coeffs = np.zeros(128, dtype=np.float32)
    dc_coeffs[0] = 1.0  # Pure DC
    
    raw_imdct = mdct.imdct256(dc_coeffs)
    
    print(f"IMDCT of pure DC=1.0:")
    print(f"  Shape: {raw_imdct.shape}")
    print(f"  Energy: {np.sum(raw_imdct**2):.6f}")
    print(f"  Mean: {np.mean(raw_imdct):.6f}")
    
    # Check each quarter
    quarters = [
        raw_imdct[:64],      # Q1 [0:64]
        raw_imdct[64:128],   # Q2 [64:128] - atracdenc extraction start
        raw_imdct[128:192],  # Q3 [128:192] - atracdenc extraction end
        raw_imdct[192:256]   # Q4 [192:256]
    ]
    
    for i, quarter in enumerate(quarters):
        print(f"  Q{i+1} [{i*64}:{(i+1)*64}]: mean={np.mean(quarter):.6f}, std={np.std(quarter):.6f}")
        print(f"    First 4: {quarter[:4]}")
        print(f"    Last 4: {quarter[-4:]}")
    
    # For theoretical perfect reconstruction:
    # DC coefficient should produce constant values in the useful region
    # The magnitude depends on the IMDCT scaling
    
    # Check what theoretical DC reconstruction should be
    print(f"\nTheoretical analysis:")
    print(f"  IMDCT scaling: {mdct.imdct256.Scale}")
    print(f"  Expected output for DC=1: depends on IMDCT normalization")
    
    # Test with our current implementation vs expected atracdenc behavior
    print(f"\nComparing quarters for meaningful data:")
    atracdenc_region = raw_imdct[64:192]  # What atracdenc extracts
    
    # For DC input, this should contain constant reconstruction values
    # If it's near-zero while other regions have meaningful data,
    # then our IMDCT has wrong scaling or output placement
    
    if np.max(np.abs(atracdenc_region)) < 0.1:
        print(f"  ❌ atracdenc region has tiny values - IMDCT output placement wrong")
        print(f"  Where is the meaningful data?")
        
        max_quarter = np.argmax([np.max(np.abs(q)) for q in quarters])
        print(f"  Strongest data in Q{max_quarter+1}: {np.max(np.abs(quarters[max_quarter])):.6f}")
        
    else:
        print(f"  ✅ atracdenc region has meaningful data: {np.max(np.abs(atracdenc_region)):.6f}")

def test_frame_to_frame_tdac():
    """Test frame-to-frame TDAC to verify overlap handling."""
    
    print(f"\n=== Frame-to-Frame TDAC Test ===")
    
    mdct = Atrac1MDCT()
    
    # Frame 1: DC signal
    frame1_input = np.ones(128, dtype=np.float32) * 0.5
    
    specs1 = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs1, frame1_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out1 = np.zeros(256, dtype=np.float32)
    mid_out1 = np.zeros(256, dtype=np.float32)
    hi_out1 = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs1, BlockSizeMode(False, False, False),
               low_out1, mid_out1, hi_out1, channel=0, frame=0)
    
    print(f"Frame 1 output:")
    print(f"  Overlap region [0:32] max: {np.max(np.abs(low_out1[:32])):.6f}")
    print(f"  Middle region [32:160] mean: {np.mean(low_out1[32:160]):.6f}")
    print(f"  Tail region [240:256]: {low_out1[240:256][:4]} ...")
    
    # Frame 2: Different DC signal
    frame2_input = np.ones(128, dtype=np.float32) * 0.8
    
    specs2 = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs2, frame2_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=1)
    
    low_out2 = np.zeros(256, dtype=np.float32)
    mid_out2 = np.zeros(256, dtype=np.float32)
    hi_out2 = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs2, BlockSizeMode(False, False, False),
               low_out2, mid_out2, hi_out2, channel=0, frame=1)
    
    print(f"Frame 2 output:")
    print(f"  Overlap region [0:32] max: {np.max(np.abs(low_out2[:32])):.6f}")
    print(f"  Middle region [32:160] mean: {np.mean(low_out2[32:160]):.6f}")
    
    # Frame 2 should have meaningful overlap because prev_buf has data from Frame 1
    if np.max(np.abs(low_out2[:32])) > 0.01:
        print(f"  ✅ Frame 2 has meaningful overlap (TDAC working)")
    else:
        print(f"  ❌ Frame 2 still has tiny overlap (TDAC not working)")
    
    # Calculate SNR for each frame
    for frame_idx, (frame_input, frame_output) in enumerate([(frame1_input, low_out1), (frame2_input, low_out2)]):
        useful_output = frame_output[32:160]
        error = useful_output - frame_input
        error_energy = np.sum(error**2)
        signal_energy = np.sum(frame_input**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
            print(f"  Frame {frame_idx+1} SNR: {snr_db:.2f} dB")

if __name__ == "__main__":
    manual_result = test_exact_atracdenc_buffer_usage()
    debug_imdct_scaling_issue()
    test_frame_to_frame_tdac()
    
    print(f"\n=== CRITICAL FINDINGS ===")
    print("1. atracdenc extraction [64:192] DOES contain meaningful data")
    print("2. First 16 values are near-zero for first frame (correct behavior)")
    print("3. If SNR is still poor, the issue is in our IMDCT implementation")
    print("4. Need to verify IMDCT scaling and output placement matches atracdenc")