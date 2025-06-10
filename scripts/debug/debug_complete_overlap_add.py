#!/usr/bin/env python3
"""
Debug complete overlap-add reconstruction across multiple frames.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_complete_tdac_reconstruction():
    """Test complete TDAC reconstruction with proper overlap-add."""
    
    print("=== Complete TDAC Reconstruction Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)  # Long blocks
    
    # Create a longer test signal
    frame_size = 128  # Input frame size
    num_frames = 4
    overlap_size = 64  # 50% overlap for TDAC
    
    # Generate test signal - simple sine wave
    total_input_samples = (num_frames - 1) * overlap_size + frame_size
    freq = 1000
    sample_rate = 44100
    t = np.arange(total_input_samples) / sample_rate
    test_signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    print(f"Test signal: {freq}Hz sine wave")
    print(f"Total input samples: {total_input_samples}")
    print(f"Frame size: {frame_size}, overlap: {overlap_size}")
    print(f"Number of frames: {num_frames}")
    
    # Process each frame through MDCT->IMDCT
    frame_outputs = []
    
    for frame_idx in range(num_frames):
        start_idx = frame_idx * overlap_size
        end_idx = start_idx + frame_size
        
        if end_idx > len(test_signal):
            frame_input = np.zeros(frame_size, dtype=np.float32)
            available = len(test_signal) - start_idx
            if available > 0:
                frame_input[:available] = test_signal[start_idx:start_idx + available]
        else:
            frame_input = test_signal[start_idx:end_idx]
        
        print(f"\nFrame {frame_idx}:")
        print(f"  Input range: [{start_idx}:{end_idx}]")
        print(f"  Input energy: {np.sum(frame_input**2):.6f}")
        
        # For this test, use only low band to isolate TDAC issues
        low_input = frame_input  # Use full frame as low band input
        # Pad to required size
        low_padded = np.zeros(128, dtype=np.float32)
        low_padded[:len(low_input)] = low_input
        
        mid_input = np.zeros(128, dtype=np.float32)
        hi_input = np.zeros(256, dtype=np.float32)
        
        # MDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_padded, mid_input, hi_input, block_size_mode, channel=0, frame=frame_idx)
        
        # IMDCT with correct buffer sizes
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        # For TDAC, we need to extract the relevant reconstruction window
        # The IMDCT produces 256 samples, but we need to overlap-add properly
        frame_outputs.append(low_out.copy())
        
        print(f"  MDCT coeffs energy: {np.sum(specs**2):.6f}")
        print(f"  IMDCT output energy: {np.sum(low_out**2):.6f}")
        print(f"  IMDCT max: {np.max(np.abs(low_out)):.6f}")
    
    # Now perform overlap-add reconstruction
    print(f"\n=== Overlap-Add Reconstruction ===")
    
    # Calculate output length
    # Each frame contributes overlap_size samples to the final output
    # Plus the last frame contributes its full remaining length
    output_length = (num_frames - 1) * overlap_size + len(frame_outputs[-1])
    reconstructed = np.zeros(output_length, dtype=np.float32)
    
    print(f"Output buffer length: {output_length}")
    
    for frame_idx, frame_output in enumerate(frame_outputs):
        output_start = frame_idx * overlap_size
        
        # Determine how much of this frame to add
        if frame_idx == num_frames - 1:
            # Last frame: add everything
            samples_to_add = len(frame_output)
        else:
            # Intermediate frames: add only overlap_size samples  
            samples_to_add = overlap_size
        
        output_end = output_start + samples_to_add
        
        if output_end <= len(reconstructed):
            # Add the frame output to the reconstruction buffer
            reconstructed[output_start:output_end] += frame_output[:samples_to_add]
            
            print(f"Frame {frame_idx}: added samples [{output_start}:{output_end}] "
                  f"(length {samples_to_add})")
            
            # Show overlap regions
            if frame_idx > 0:
                overlap_start = output_start
                overlap_end = min(output_start + overlap_size, output_end)
                overlap_values = reconstructed[overlap_start:overlap_end]
                print(f"  Overlap region [{overlap_start}:{overlap_end}]: "
                      f"mean={np.mean(overlap_values):.4f}, std={np.std(overlap_values):.4f}")
        else:
            # Handle boundary case
            available_space = len(reconstructed) - output_start
            if available_space > 0:
                reconstructed[output_start:] += frame_output[:available_space]
                print(f"Frame {frame_idx}: added final {available_space} samples")
    
    # Analyze reconstruction quality
    print(f"\n=== Quality Analysis ===")
    
    # Compare with original signal
    comparison_length = min(len(test_signal), len(reconstructed))
    original_section = test_signal[:comparison_length]
    reconstructed_section = reconstructed[:comparison_length]
    
    print(f"Comparing {comparison_length} samples")
    print(f"Original energy: {np.sum(original_section**2):.6f}")
    print(f"Reconstructed energy: {np.sum(reconstructed_section**2):.6f}")
    
    # Calculate error and SNR
    error = reconstructed_section - original_section
    error_energy = np.sum(error**2)
    signal_energy = np.sum(original_section**2)
    
    if signal_energy > 0 and error_energy > 0:
        snr_db = 10 * np.log10(signal_energy / error_energy)
        energy_ratio = np.sum(reconstructed_section**2) / signal_energy
        
        print(f"SNR: {snr_db:.2f} dB")
        print(f"Energy ratio: {energy_ratio:.6f}")
        
        # Check frame boundaries for discontinuities
        print(f"\nFrame boundary analysis:")
        for frame_idx in range(1, num_frames):
            boundary_sample = frame_idx * overlap_size
            if boundary_sample > 5 and boundary_sample < len(reconstructed_section) - 5:
                window = reconstructed_section[boundary_sample-3:boundary_sample+3]
                max_jump = np.max(np.abs(np.diff(window)))
                print(f"  Frame {frame_idx} boundary (sample {boundary_sample}): max jump = {max_jump:.6f}")
        
        return snr_db
    else:
        print("Cannot calculate SNR")
        return 0

def test_single_frame_perfect_reconstruction():
    """Test if a single frame can achieve perfect reconstruction."""
    
    print(f"\n=== Single Frame Perfect Reconstruction Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)
    
    # Test with different input types
    test_cases = [
        ("DC signal", np.ones(128, dtype=np.float32) * 0.5),
        ("Linear ramp", np.arange(128, dtype=np.float32) / 128.0),
        ("Sine wave", np.sin(2 * np.pi * np.arange(128) / 32).astype(np.float32)),
    ]
    
    for name, test_input in test_cases:
        print(f"\nTesting {name}:")
        print(f"  Input energy: {np.sum(test_input**2):.6f}")
        
        # MDCT -> IMDCT round trip
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32), 
                 block_size_mode, channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=0)
        
        # For single frame, the useful reconstruction is typically in the middle
        # Avoid the overlap regions at the edges
        useful_start = 32  # Skip vector_fmul_window region
        useful_end = useful_start + 112  # Low band long block middle section
        useful_reconstruction = low_out[useful_start:useful_end]
        
        # Compare with corresponding input section
        input_section = test_input[:len(useful_reconstruction)]
        
        if len(input_section) == len(useful_reconstruction):
            error = np.mean(np.abs(useful_reconstruction - input_section))
            energy_ratio = np.sum(useful_reconstruction**2) / np.sum(input_section**2)
            
            print(f"  Reconstruction error: {error:.6f}")
            print(f"  Energy ratio: {energy_ratio:.6f}")
            
            # Check if it's roughly the expected scaling (0.25 for low band)
            expected_scaling = 0.25
            scaling_error = abs(energy_ratio - expected_scaling**2)
            
            if scaling_error < 0.01:
                print(f"  ✅ Energy ratio close to expected {expected_scaling**2:.3f}")
            else:
                print(f"  ⚠️  Energy ratio differs from expected {expected_scaling**2:.3f}")
        else:
            print(f"  ⚠️  Length mismatch: input {len(input_section)}, output {len(useful_reconstruction)}")

if __name__ == "__main__":
    snr = test_complete_tdac_reconstruction()
    test_single_frame_perfect_reconstruction()
    
    print(f"\n=== Final Assessment ===")
    if snr > 40:
        print("🎉 TDAC RECONSTRUCTION IS EXCELLENT!")
    elif snr > 20:
        print("✅ TDAC reconstruction is good")
    elif snr > 10:
        print("⚠️  TDAC reconstruction has issues")
    else:
        print("❌ TDAC reconstruction is poor")