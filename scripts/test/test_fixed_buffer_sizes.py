#!/usr/bin/env python3
"""
Test TDAC aliasing with corrected buffer sizes.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_tdac_with_correct_buffers():
    """Test TDAC with correct buffer sizes (256,256,512)."""
    
    print("=== TDAC Test with Correct Buffer Sizes ===")
    
    # Test overlap-add reconstruction with corrected buffer sizes
    frame_size = 128  # Use smaller frame for simpler analysis
    overlap_size = frame_size // 2  # 64 samples overlap
    
    # Create test signal - sine wave
    total_frames = 3
    total_samples = frame_size * total_frames
    freq = 1000
    sample_rate = 44100
    
    t = np.arange(total_samples) / sample_rate
    test_signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    print(f"Test signal: {freq}Hz sine, {total_samples} samples")
    print(f"Frame size: {frame_size}, overlap: {overlap_size}")
    
    # Initialize MDCT with corrected scaling
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)  # Long blocks
    
    # Process overlapping frames using ATRAC1 full pipeline
    reconstructed_frames = []
    
    for frame_idx in range(total_frames):
        print(f"\nFrame {frame_idx}:")
        
        # Extract overlapping frame for each QMF band
        start_idx = frame_idx * overlap_size
        end_idx = start_idx + frame_size
        
        if end_idx > len(test_signal):
            frame_data = np.zeros(frame_size, dtype=np.float32)
            available_samples = len(test_signal) - start_idx
            if available_samples > 0:
                frame_data[:available_samples] = test_signal[start_idx:start_idx + available_samples]
        else:
            frame_data = test_signal[start_idx:end_idx]
        
        # Simulate QMF bands (simplified - just split the signal)
        # In reality QMF would split into frequency bands, but for TDAC testing
        # we can use simple time-domain splits
        low_band = frame_data[:frame_size//2].copy()  # First half -> low band (64 samples)
        mid_band = frame_data[frame_size//2:].copy()  # Second half -> mid band (64 samples)
        hi_band = np.tile(frame_data, 2)  # Duplicate to make 256 samples for high band
        
        # Pad to expected sizes
        low_input = np.zeros(128, dtype=np.float32)
        mid_input = np.zeros(128, dtype=np.float32)
        hi_input = np.zeros(256, dtype=np.float32)
        
        low_input[:len(low_band)] = low_band
        mid_input[:len(mid_band)] = mid_band  
        hi_input[:len(hi_band)] = hi_band
        
        print(f"  Input energies: Low={np.sum(low_input**2):.6f}, Mid={np.sum(mid_input**2):.6f}, Hi={np.sum(hi_input**2):.6f}")
        
        # Forward MDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_input, mid_input, hi_input, block_size_mode, channel=0, frame=frame_idx)
        
        print(f"  MDCT coeffs energy: {np.sum(specs**2):.6f}")
        
        # Inverse MDCT with CORRECT buffer sizes
        low_out = np.zeros(256, dtype=np.float32)   # Low: 256 samples  
        mid_out = np.zeros(256, dtype=np.float32)   # Mid: 256 samples
        hi_out = np.zeros(512, dtype=np.float32)    # High: 512 samples (CRITICAL FIX)
        
        mdct.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        print(f"  IMDCT energies: Low={np.sum(low_out**2):.6f}, Mid={np.sum(mid_out**2):.6f}, Hi={np.sum(hi_out**2):.6f}")
        
        # Combine bands back to time domain (simplified inverse of QMF split)
        # Take first 128 samples from low, middle 128 from mid  
        reconstructed_frame = np.zeros(frame_size, dtype=np.float32)
        reconstructed_frame[:frame_size//2] = low_out[:frame_size//2]
        reconstructed_frame[frame_size//2:] = mid_out[:frame_size//2]
        
        # Add contribution from high band (simplified)
        hi_contribution = hi_out[:frame_size] * 0.1  # Scale down high band contribution
        reconstructed_frame += hi_contribution
        
        reconstructed_frames.append(reconstructed_frame)
        
        print(f"  Frame reconstruction energy: {np.sum(reconstructed_frame**2):.6f}")
    
    # Overlap-add the reconstructed frames
    total_reconstructed = np.zeros(total_samples, dtype=np.float32)
    
    for frame_idx, frame_data in enumerate(reconstructed_frames):
        start_idx = frame_idx * overlap_size
        end_idx = start_idx + len(frame_data)
        
        if end_idx <= len(total_reconstructed):
            total_reconstructed[start_idx:end_idx] += frame_data
        else:
            available = len(total_reconstructed) - start_idx
            if available > 0:
                total_reconstructed[start_idx:start_idx + available] += frame_data[:available]
    
    # Analyze reconstruction quality
    print(f"\n=== Reconstruction Analysis ===")
    
    # Compare middle section (avoid edge effects)
    comparison_start = overlap_size
    comparison_end = len(test_signal) - overlap_size
    
    if comparison_end > comparison_start:
        original_section = test_signal[comparison_start:comparison_end]
        reconstructed_section = total_reconstructed[comparison_start:comparison_end]
        
        print(f"Comparing samples [{comparison_start}:{comparison_end}]")
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
            
            if snr_db > 40:
                print("✅ EXCELLENT reconstruction - TDAC working correctly!")
            elif snr_db > 20:
                print("✅ GOOD reconstruction - TDAC mostly working")
            elif snr_db > 10:
                print("⚠️  FAIR reconstruction - some TDAC issues remain")
            else:
                print("❌ POOR reconstruction - TDAC still broken")
                
            return snr_db
        else:
            print("⚠️  Cannot calculate SNR")
            return 0
    else:
        print("⚠️  Cannot compare - insufficient data")
        return 0

def test_direct_mdct_imdct():
    """Test direct MDCT->IMDCT with correct buffer sizes."""
    
    print(f"\n=== Direct MDCT->IMDCT Test ===")
    
    mdct = Atrac1MDCT()
    block_size_mode = BlockSizeMode(False, False, False)
    
    # Simple DC test
    low_input = np.ones(128, dtype=np.float32) * 0.5
    mid_input = np.zeros(128, dtype=np.float32)
    hi_input = np.zeros(256, dtype=np.float32)
    
    print("Input: DC signal on low band only")
    print(f"Input energy: {np.sum(low_input**2):.6f}")
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, mid_input, hi_input, block_size_mode, channel=0, frame=0)
    
    print(f"MDCT output energy: {np.sum(specs**2):.6f}")
    print(f"MDCT max coeff: {np.max(np.abs(specs)):.6f}")
    
    # Inverse MDCT with correct buffer sizes
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)  # CORRECT SIZE
    
    mdct.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=0)
    
    print(f"IMDCT output energies: Low={np.sum(low_out**2):.6f}, Mid={np.sum(mid_out**2):.6f}, Hi={np.sum(hi_out**2):.6f}")
    
    # Check low band reconstruction (where we put the input)
    low_region = low_out[32:224]  # Avoid edge effects
    low_mean = np.mean(low_region)
    low_std = np.std(low_region)
    
    print(f"Low band reconstruction: mean={low_mean:.4f}, std={low_std:.4f}")
    print(f"Input mean was: {np.mean(low_input):.4f}")
    
    amplitude_ratio = low_mean / np.mean(low_input)
    print(f"Amplitude ratio: {amplitude_ratio:.4f}")
    
    if low_std < 0.1 * abs(low_mean):
        print("✅ Low band reconstruction is reasonably constant")
    else:
        print("❌ Low band reconstruction has high variation")
    
    print(f"Scaling factor: {amplitude_ratio:.3f} (atracdenc expects ~0.25 for low band)")

if __name__ == "__main__":
    snr = test_tdac_with_correct_buffers()
    test_direct_mdct_imdct()
    
    print(f"\n=== Final Assessment ===")
    if snr > 20:
        print("🎉 BUFFER SIZE FIX RESOLVED TDAC ALIASING!")
    elif snr > 10:
        print("🤔 BUFFER SIZE FIX HELPED BUT ISSUES REMAIN")
    else:
        print("😞 BUFFER SIZE FIX DID NOT RESOLVE TDAC ALIASING")