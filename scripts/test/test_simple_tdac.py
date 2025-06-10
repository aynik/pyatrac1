#!/usr/bin/env python3
"""
Simple test to prove TDAC aliasing is resolved with corrected MDCT/IMDCT.
"""

import numpy as np
from pyatrac1.core.mdct import MDCT, IMDCT

def test_simple_tdac():
    """Test TDAC aliasing with direct MDCT/IMDCT without QMF."""
    
    print("=== Simple TDAC Test (No QMF) ===")
    
    # Test overlap-add reconstruction with 256-point MDCT
    frame_size = 256
    overlap_size = frame_size // 2  # 128 samples overlap
    
    # Create test signal - sine wave
    total_frames = 3
    total_samples = frame_size * total_frames
    freq = 1000
    sample_rate = 44100
    
    t = np.arange(total_samples) / sample_rate
    test_signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    print(f"Test signal: {freq}Hz sine, {total_samples} samples")
    print(f"Frame size: {frame_size}, overlap: {overlap_size}")
    
    # Initialize MDCT/IMDCT with corrected scaling
    mdct = MDCT(frame_size, 0.5)
    imdct = IMDCT(frame_size, frame_size * 2)
    
    # Process overlapping frames
    reconstructed = np.zeros(total_samples, dtype=np.float32)
    
    for frame_idx in range(total_frames):
        print(f"\nFrame {frame_idx}:")
        
        # Extract overlapping frame
        start_idx = frame_idx * overlap_size
        end_idx = start_idx + frame_size
        
        if end_idx > len(test_signal):
            # Pad with zeros if needed
            frame_data = np.zeros(frame_size, dtype=np.float32)
            available_samples = len(test_signal) - start_idx
            if available_samples > 0:
                frame_data[:available_samples] = test_signal[start_idx:start_idx + available_samples]
        else:
            frame_data = test_signal[start_idx:end_idx]
        
        print(f"  Input energy: {np.sum(frame_data**2):.6f}")
        
        # MDCT
        coeffs = mdct(frame_data)
        print(f"  MDCT coeffs energy: {np.sum(coeffs**2):.6f}")
        
        # IMDCT
        reconstructed_frame = imdct(coeffs)
        print(f"  IMDCT output energy: {np.sum(reconstructed_frame**2):.6f}")
        
        # Overlap-add to output buffer
        output_start = start_idx
        output_end = output_start + frame_size
        
        if output_end <= len(reconstructed):
            reconstructed[output_start:output_end] += reconstructed_frame
        else:
            # Handle end boundary
            available_output = len(reconstructed) - output_start
            if available_output > 0:
                reconstructed[output_start:output_start + available_output] += reconstructed_frame[:available_output]
        
        print(f"  Added to output at [{output_start}:{output_end}]")
    
    # Analyze reconstruction quality
    print(f"\n=== Reconstruction Analysis ===")
    
    # Compare overlapping region (avoid edges)
    comparison_start = overlap_size
    comparison_end = len(test_signal) - overlap_size
    
    if comparison_end > comparison_start:
        original_section = test_signal[comparison_start:comparison_end]
        reconstructed_section = reconstructed[comparison_start:comparison_end]
        
        print(f"Comparing samples [{comparison_start}:{comparison_end}]")
        print(f"Original energy: {np.sum(original_section**2):.6f}")
        print(f"Reconstructed energy: {np.sum(reconstructed_section**2):.6f}")
        
        # Calculate error
        error = reconstructed_section - original_section
        error_energy = np.sum(error**2)
        signal_energy = np.sum(original_section**2)
        
        if signal_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy) if error_energy > 0 else float('inf')
            print(f"SNR: {snr_db:.2f} dB")
            
            if snr_db > 40:
                print("✅ EXCELLENT reconstruction - TDAC working correctly")
            elif snr_db > 20:
                print("✅ GOOD reconstruction - TDAC mostly working")
            elif snr_db > 10:
                print("⚠️  FAIR reconstruction - some TDAC issues")
            else:
                print("❌ POOR reconstruction - TDAC not working")
        else:
            print("⚠️  Cannot calculate SNR (zero signal)")
    
    # Check for discontinuities at frame boundaries
    print(f"\n=== Frame Boundary Analysis ===")
    
    for frame_idx in range(1, total_frames):
        boundary_sample = frame_idx * overlap_size
        
        if boundary_sample > 10 and boundary_sample < len(reconstructed) - 10:
            # Check samples around boundary
            window = reconstructed[boundary_sample-5:boundary_sample+5]
            max_jump = np.max(np.abs(np.diff(window)))
            
            print(f"Frame {frame_idx} boundary (sample {boundary_sample}):")
            print(f"  Max discontinuity: {max_jump:.6f}")
            
            if max_jump < 0.1:
                print(f"  ✅ Smooth transition")
            else:
                print(f"  ⚠️  Potential discontinuity")
    
    return snr_db if 'snr_db' in locals() else 0

def test_different_sizes():
    """Test TDAC with different MDCT sizes."""
    
    print(f"\n=== Different MDCT Sizes Test ===")
    
    sizes_and_scales = [
        (64, 0.5),
        (128, 0.5), 
        (256, 0.5),
        (512, 1.0)
    ]
    
    for size, scale in sizes_and_scales:
        print(f"\nTesting {size}-point MDCT:")
        
        # Create test signal
        test_signal = np.sin(2 * np.pi * np.arange(size * 2) / size).astype(np.float32)
        
        # MDCT/IMDCT
        mdct = MDCT(size, scale)
        imdct = IMDCT(size, size * 2)
        
        # Process two overlapping frames
        frame1 = test_signal[:size]
        frame2 = test_signal[size//2:size//2 + size]
        
        coeffs1 = mdct(frame1)
        coeffs2 = mdct(frame2)
        
        recon1 = imdct(coeffs1)
        recon2 = imdct(coeffs2)
        
        # Overlap-add
        result = np.zeros(size + size//2, dtype=np.float32)
        result[:size] += recon1
        result[size//2:size//2 + size] += recon2
        
        # Check reconstruction quality
        original_middle = test_signal[size//4:size//4 + size]
        reconstructed_middle = result[size//4:size//4 + size]
        
        error = np.mean(np.abs(original_middle - reconstructed_middle))
        energy_ratio = np.sum(reconstructed_middle**2) / np.sum(original_middle**2)
        
        print(f"  Reconstruction error: {error:.6f}")
        print(f"  Energy ratio: {energy_ratio:.6f}")
        
        if error < 0.1:
            print(f"  ✅ Good reconstruction")
        else:
            print(f"  ❌ Poor reconstruction")

if __name__ == "__main__":
    snr = test_simple_tdac()
    test_different_sizes()
    
    print(f"\n=== Final Result ===")
    if snr > 20:
        print("✅ TDAC aliasing appears to be RESOLVED")
    else:
        print("❌ TDAC aliasing is NOT resolved")