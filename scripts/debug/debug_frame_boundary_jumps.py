#!/usr/bin/env python3
"""
Debug frame boundary discontinuities in overlap-add reconstruction.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def debug_overlap_add_mechanism():
    """Debug the overlap-add mechanism that causes frame boundary jumps."""
    
    print("=== Overlap-Add Mechanism Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Create two simple frames for clear analysis
    frame1 = np.ones(128, dtype=np.float32) * 0.3  # Constant 0.3
    frame2 = np.ones(128, dtype=np.float32) * 0.7  # Constant 0.7
    
    print("Frame 1: constant 0.3")
    print("Frame 2: constant 0.7")
    print("Expected TDAC: smooth transition from 0.3 to 0.7")
    
    # Process both frames
    frame_outputs = []
    
    for frame_idx, frame_input in enumerate([frame1, frame2]):
        print(f"\n=== Processing Frame {frame_idx + 1} ===")
        
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, frame_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=frame_idx)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        frame_outputs.append(low_out.copy())
        
        # Analyze frame output structure
        print(f"Frame {frame_idx + 1} output analysis:")
        print(f"  Total energy: {np.sum(low_out**2):.6f}")
        
        # Critical regions for TDAC
        overlap_start = low_out[:32]     # Should connect to previous frame
        middle_core = low_out[32:144]    # Core reconstruction
        middle_ext = low_out[144:224]    # Extended middle
        overlap_end = low_out[224:]      # Tail for next frame
        
        print(f"  Overlap start [0:32]: mean={np.mean(overlap_start):.6f}, range=[{np.min(overlap_start):.6f}, {np.max(overlap_start):.6f}]")
        print(f"  Middle core [32:144]: mean={np.mean(middle_core):.6f}, std={np.std(middle_core):.6f}")
        print(f"  Middle ext [144:224]: mean={np.mean(middle_ext):.6f}, std={np.std(middle_ext):.6f}")
        print(f"  Overlap end [224:256]: mean={np.mean(overlap_end):.6f}, range=[{np.min(overlap_end):.6f}, {np.max(overlap_end):.6f}]")
        
        # Expected values
        expected_mean = frame_input[0] * 0.269526  # Using observed scaling factor
        print(f"  Expected mean: {expected_mean:.6f}")
        print(f"  Core error: {abs(np.mean(middle_core) - expected_mean):.6f}")
    
    # TDAC reconstruction analysis
    print(f"\n=== TDAC Reconstruction Analysis ===")
    
    # Simulate proper overlap-add like ATRAC1
    # Frame 1 contributes samples [0:64] from its middle region [32:96]
    # Frame 2 gets added at offset 64, overlapping [64:128]
    
    # Extract reconstruction portions
    frame1_contribution = frame_outputs[0][32:96]  # 64 samples from middle
    frame2_overlap = frame_outputs[1][:64]         # First 64 samples (overlap + start of middle)
    frame2_middle = frame_outputs[1][32:96]        # Middle portion for comparison
    
    print(f"Frame 1 contribution [32:96]:")
    print(f"  Values: {frame1_contribution[:8]} ...")
    print(f"  Mean: {np.mean(frame1_contribution):.6f}")
    print(f"  Range: [{np.min(frame1_contribution):.6f}, {np.max(frame1_contribution):.6f}]")
    
    print(f"Frame 2 overlap [0:64]:")
    print(f"  Values: {frame2_overlap[:8]} ...")
    print(f"  Mean: {np.mean(frame2_overlap):.6f}")
    print(f"  Range: [{np.min(frame2_overlap):.6f}, {np.max(frame2_overlap):.6f}]")
    
    # Perform overlap-add
    overlap_region = frame1_contribution + frame2_overlap
    
    print(f"Overlap-add result:")
    print(f"  Values: {overlap_region[:8]} ...")
    print(f"  Mean: {np.mean(overlap_region):.6f}")
    print(f"  Std: {np.std(overlap_region):.6f}")
    print(f"  Range: [{np.min(overlap_region):.6f}, {np.max(overlap_region):.6f}]")
    
    # Check for smooth transition
    transition_gradient = np.diff(overlap_region)
    max_jump = np.max(np.abs(transition_gradient))
    
    print(f"  Max jump: {max_jump:.6f}")
    
    if max_jump < 0.05:
        print("  ✅ Smooth overlap transition")
    elif max_jump < 0.1:
        print("  ⚠️  Moderate overlap discontinuity")
    else:
        print("  ❌ Large overlap discontinuity")
    
    # Analyze why we might have discontinuities
    print(f"\nDiscontinuity analysis:")
    
    # Expected smooth transition from frame1 mean to frame2 mean
    frame1_expected = 0.3 * 0.269526
    frame2_expected = 0.7 * 0.269526
    
    print(f"  Frame 1 expected: {frame1_expected:.6f}")
    print(f"  Frame 2 expected: {frame2_expected:.6f}")
    print(f"  Expected smooth transition range: [{frame1_expected:.6f}, {frame2_expected:.6f}]")
    
    # Check if overlap region spans this range
    overlap_min = np.min(overlap_region)
    overlap_max = np.max(overlap_region)
    
    print(f"  Actual overlap range: [{overlap_min:.6f}, {overlap_max:.6f}]")
    
    if overlap_min <= frame1_expected and overlap_max >= frame2_expected:
        print("  ✅ Overlap range spans expected transition")
    else:
        print("  ❌ Overlap range doesn't span expected transition")
    
    return max_jump, overlap_region

def debug_vector_fmul_window_state():
    """Debug vector_fmul_window frame-to-frame state management."""
    
    print(f"\n=== vector_fmul_window State Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Test consecutive frames to see state preservation
    frames = [
        np.ones(128, dtype=np.float32) * 0.4,  # Frame 0
        np.ones(128, dtype=np.float32) * 0.8,  # Frame 1
    ]
    
    print("Testing vector_fmul_window state across frames")
    
    frame_overlaps = []
    
    for frame_idx, frame_input in enumerate(frames):
        print(f"\nFrame {frame_idx}:")
        print(f"  Input: constant {frame_input[0]}")
        
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, frame_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=frame_idx)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        # The vector_fmul_window output is in [0:32]
        overlap_output = low_out[:32]
        frame_overlaps.append(overlap_output)
        
        print(f"  vector_fmul_window output [0:32]:")
        print(f"    Values: {overlap_output[:8]} ...")
        print(f"    Mean: {np.mean(overlap_output):.6f}")
        print(f"    Energy: {np.sum(overlap_output**2):.6f}")
        print(f"    Max: {np.max(np.abs(overlap_output)):.6f}")
        
        # For frame 0, this should be influenced by zero prev_buf
        # For frame 1, this should be influenced by frame 0's tail
        
        if frame_idx == 0:
            print(f"    (First frame - prev_buf should be zeros)")
        else:
            print(f"    (Subsequent frame - should reflect previous frame state)")
    
    # Analyze frame-to-frame transition
    if len(frame_overlaps) >= 2:
        print(f"\nFrame-to-frame transition analysis:")
        
        frame0_overlap = frame_overlaps[0]
        frame1_overlap = frame_overlaps[1]
        
        # The end of frame 0 should influence the start of frame 1
        frame0_tail = frame0_overlap[-8:]  # Last part of frame 0 overlap
        frame1_start = frame1_overlap[:8]  # Start of frame 1 overlap
        
        print(f"  Frame 0 tail: {frame0_tail}")
        print(f"  Frame 1 start: {frame1_start}")
        
        transition_smoothness = np.mean(np.abs(np.diff(np.concatenate([frame0_tail, frame1_start]))))
        print(f"  Transition smoothness: {transition_smoothness:.6f}")
        
        if transition_smoothness < 0.05:
            print("  ✅ Smooth frame transition")
        else:
            print("  ❌ Rough frame transition")

def test_perfect_overlap_add():
    """Test what perfect overlap-add reconstruction should look like."""
    
    print(f"\n=== Perfect Overlap-Add Test ===")
    
    # Create ideal overlapping frames
    frame_size = 64
    overlap_size = 32
    
    # Frame 1: linear ramp from 0 to 1
    frame1 = np.linspace(0, 1, frame_size, dtype=np.float32)
    
    # Frame 2: linear ramp from 0.5 to 1.5 (overlaps with frame1)
    frame2 = np.linspace(0.5, 1.5, frame_size, dtype=np.float32)
    
    print("Perfect overlap-add example:")
    print(f"  Frame 1: {frame1[:8]} ... {frame1[-8:]}")
    print(f"  Frame 2: {frame2[:8]} ... {frame2[-8:]}")
    
    # Reconstruct with perfect overlap-add
    total_length = frame_size + frame_size - overlap_size
    reconstructed = np.zeros(total_length, dtype=np.float32)
    
    # Add frame 1
    reconstructed[:frame_size] = frame1
    
    # Add frame 2 with overlap
    start_pos = frame_size - overlap_size
    reconstructed[start_pos:start_pos + frame_size] += frame2
    
    print(f"  Reconstructed: {reconstructed[:8]} ... {reconstructed[-8:]}")
    
    # Check overlap region
    overlap_region = reconstructed[start_pos:start_pos + overlap_size]
    print(f"  Overlap region [{start_pos}:{start_pos + overlap_size}]: {overlap_region}")
    
    # Check smoothness
    gradient = np.diff(reconstructed)
    max_jump = np.max(np.abs(gradient))
    
    print(f"  Max jump: {max_jump:.6f}")
    print(f"  Perfect overlap-add achieves: {max_jump:.6f} max jump")
    
    return max_jump

if __name__ == "__main__":
    max_jump, overlap_region = debug_overlap_add_mechanism()
    debug_vector_fmul_window_state()
    perfect_jump = test_perfect_overlap_add()
    
    print(f"\n=== Frame Boundary Summary ===")
    print(f"Our TDAC max jump: {max_jump:.6f}")
    print(f"Perfect overlap-add: {perfect_jump:.6f}")
    print(f"Jump ratio: {max_jump / perfect_jump:.2f}x worse than perfect")
    
    if max_jump < 0.05:
        print("✅ Excellent frame boundaries")
    elif max_jump < 0.1:
        print("✅ Good frame boundaries")  
    elif max_jump < 0.2:
        print("⚠️  Moderate frame boundary issues")
    else:
        print("❌ Poor frame boundaries")