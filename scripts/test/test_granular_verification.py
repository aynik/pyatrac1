#!/usr/bin/env python3
"""
Granular verification tests to find remaining TDAC issues.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_perfect_reconstruction_properties():
    """Test if MDCT->IMDCT has perfect reconstruction properties for overlapping windows."""
    
    print("=== Perfect Reconstruction Properties Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test with impulse at different positions
    print("Testing impulse reconstruction:")
    
    for impulse_pos in [32, 64, 96]:
        print(f"\nImpulse at position {impulse_pos}:")
        
        # Create impulse
        low_input = np.zeros(128, dtype=np.float32)
        low_input[impulse_pos] = 1.0
        
        # MDCT->IMDCT round trip
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Analyze reconstruction
        input_energy = np.sum(low_input**2)
        output_energy = np.sum(low_out**2)
        
        # For TDAC, we expect the useful signal to be in the middle part
        # Skip overlap regions [0:32] and [224:256], use middle [32:224]
        useful_output = low_out[32:224]  # 192 samples
        useful_input = low_input[32:224] if len(low_input) >= 224 else low_input[:len(useful_output)]
        
        # Ensure both arrays have same length for correlation
        min_len = min(len(useful_input), len(useful_output))
        useful_input = useful_input[:min_len]
        useful_output = useful_output[:min_len]
        
        if impulse_pos < len(useful_output):
            peak_response = useful_output[impulse_pos - 32] if impulse_pos >= 32 else 0
            correlation = np.corrcoef(useful_input, useful_output)[0, 1] if np.std(useful_output) > 0 and np.std(useful_input) > 0 else 0
            
            print(f"  Input energy: {input_energy:.6f}")
            print(f"  Total output energy: {output_energy:.6f}")
            print(f"  Useful output energy: {np.sum(useful_output**2):.6f}")
            print(f"  Peak response at {impulse_pos}: {peak_response:.6f}")
            print(f"  Correlation: {correlation:.6f}")
            
            if abs(peak_response) > 0.1:
                print(f"  ✅ Strong impulse response")
            else:
                print(f"  ❌ Weak impulse response")
                
            if correlation > 0.8:
                print(f"  ✅ Good correlation")
            else:
                print(f"  ⚠️  Poor correlation")

def test_windowing_compensation():
    """Test if windowing effects are properly compensated in TDAC."""
    
    print(f"\n=== Windowing Compensation Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test with constant signal to see windowing effects
    print("Testing constant signal windowing:")
    
    levels = [0.1, 0.5, 1.0]
    for level in levels:
        print(f"\nConstant level {level}:")
        
        low_input = np.ones(128, dtype=np.float32) * level
        
        # Single frame MDCT->IMDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Analyze different regions
        overlap_start = low_out[:32]
        middle_region = low_out[32:224]
        overlap_end = low_out[224:]
        
        print(f"  Input: constant {level}")
        print(f"  Overlap start: mean={np.mean(overlap_start):.6f}, std={np.std(overlap_start):.6f}")
        print(f"  Middle region: mean={np.mean(middle_region):.6f}, std={np.std(middle_region):.6f}")
        print(f"  Overlap end: mean={np.mean(overlap_end):.6f}, std={np.std(overlap_end):.6f}")
        
        # Expected scaling for low band is ~0.25
        expected_output = level * 0.25
        middle_error = abs(np.mean(middle_region) - expected_output)
        
        print(f"  Expected middle: {expected_output:.6f}")
        print(f"  Middle error: {middle_error:.6f}")
        
        if middle_error < 0.01:
            print(f"  ✅ Accurate scaling")
        else:
            print(f"  ⚠️  Scaling error")
        
        if np.std(middle_region) < 0.1 * abs(np.mean(middle_region)):
            print(f"  ✅ Middle region is constant")
        else:
            print(f"  ❌ Middle region varies too much")

def test_two_frame_tdac():
    """Test TDAC reconstruction with two overlapping frames."""
    
    print(f"\n=== Two-Frame TDAC Test ===")
    
    mdct = Atrac1MDCT()
    
    # Create two frames with known overlap pattern
    frame1 = np.zeros(128, dtype=np.float32)
    frame1[64:] = 1.0  # Step function at middle
    
    frame2 = np.ones(128, dtype=np.float32) * 0.5  # Constant
    
    print("Frame 1: step function (0 -> 1 at position 64)")
    print("Frame 2: constant 0.5")
    print("Expected TDAC: smooth transition in overlap region")
    
    # Process both frames
    frame_outputs = []
    
    for frame_idx, frame_input in enumerate([frame1, frame2]):
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, frame_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=frame_idx)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        frame_outputs.append(low_out)
        
        print(f"\nFrame {frame_idx + 1} output analysis:")
        print(f"  Total energy: {np.sum(low_out**2):.6f}")
        print(f"  Middle [96:160] mean: {np.mean(low_out[96:160]):.6f}")
        print(f"  End region [200:240] mean: {np.mean(low_out[200:240]):.6f}")
    
    # Reconstruct with overlap-add
    print(f"\nTDAC overlap-add reconstruction:")
    
    # Simulate 64-sample overlap like ATRAC1
    total_length = 256 + 64  # Frame1 + 64 samples from Frame2
    reconstructed = np.zeros(total_length, dtype=np.float32)
    
    # Add frame 1 completely
    reconstructed[:256] = frame_outputs[0]
    
    # Add frame 2 with 64-sample offset (overlap region)
    overlap_start = 64
    overlap_length = 64
    reconstructed[overlap_start:overlap_start + overlap_length] += frame_outputs[1][:overlap_length]
    
    # Analyze overlap region
    overlap_region = reconstructed[overlap_start:overlap_start + overlap_length]
    
    print(f"  Overlap region [64:128]:")
    print(f"    Values: {overlap_region[:8]} ...")
    print(f"    Mean: {np.mean(overlap_region):.6f}")
    print(f"    Std: {np.std(overlap_region):.6f}")
    print(f"    Min: {np.min(overlap_region):.6f}")
    print(f"    Max: {np.max(overlap_region):.6f}")
    
    # Check for smooth transition
    transition_gradient = np.diff(overlap_region)
    max_jump = np.max(np.abs(transition_gradient))
    
    print(f"    Max jump in overlap: {max_jump:.6f}")
    
    if max_jump < 0.1:
        print(f"    ✅ Smooth overlap transition")
    else:
        print(f"    ❌ Discontinuous overlap transition")
    
    # Expected behavior: overlap region should show smooth transition
    # from frame1 values to frame2 values
    
    return max_jump

def test_energy_conservation():
    """Test if energy is conserved through MDCT->IMDCT pipeline."""
    
    print(f"\n=== Energy Conservation Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test with different signal types
    test_signals = [
        ("DC", np.ones(128, dtype=np.float32) * 0.5),
        ("Ramp", np.arange(128, dtype=np.float32) / 128.0),
        ("Sine", np.sin(2 * np.pi * np.arange(128) / 16).astype(np.float32) * 0.5),
        ("Noise", np.random.normal(0, 0.3, 128).astype(np.float32))
    ]
    
    for name, signal in test_signals:
        input_energy = np.sum(signal**2)
        
        # MDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, signal, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        mdct_energy = np.sum(specs[:128]**2)  # Low band only
        
        # IMDCT
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        output_energy = np.sum(low_out**2)
        useful_energy = np.sum(low_out[32:224]**2)  # Middle region only
        
        print(f"\n{name} signal:")
        print(f"  Input energy: {input_energy:.6f}")
        print(f"  MDCT energy: {mdct_energy:.6f}")
        print(f"  Total output energy: {output_energy:.6f}")
        print(f"  Useful output energy: {useful_energy:.6f}")
        print(f"  Energy ratio (total): {output_energy / input_energy:.6f}")
        print(f"  Energy ratio (useful): {useful_energy / input_energy:.6f}")
        
        # For perfect reconstruction, we expect some scaling but consistent ratio
        expected_ratio = 0.25 * 0.25  # Rough estimate for low band scaling
        ratio_error = abs(useful_energy / input_energy - expected_ratio)
        
        if ratio_error < 0.05:
            print(f"  ✅ Energy conservation within tolerance")
        else:
            print(f"  ⚠️  Energy conservation error: {ratio_error:.6f}")

if __name__ == "__main__":
    test_perfect_reconstruction_properties()
    test_windowing_compensation()
    max_jump = test_two_frame_tdac()
    test_energy_conservation()
    
    print(f"\n=== Granular Verification Summary ===")
    if max_jump < 0.05:
        print("✅ TDAC overlap mechanism is working well")
    elif max_jump < 0.1:
        print("⚠️  TDAC overlap has minor issues")
    else:
        print("❌ TDAC overlap has significant problems")
    
    print("Check individual test results above for specific issues to address.")