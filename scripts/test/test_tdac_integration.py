#!/usr/bin/env python3
"""
Integration test to verify TDAC (Time Domain Alias Cancellation) is working correctly.
This test checks if encoder->decoder produces perfect reconstruction for a simple test signal.
"""

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.core.decoder import Atrac1Decoder

def test_perfect_reconstruction():
    """Test if encoder->decoder produces perfect reconstruction (ignoring quantization)."""
    
    # Create a simple test signal - sine wave at different frequencies
    num_frames = 4
    samples_per_frame = 512
    total_samples = num_frames * samples_per_frame
    
    # Generate test signal with frequencies that will reveal TDAC issues
    t = np.linspace(0, total_samples / 44100, total_samples, endpoint=False)
    test_signal = (
        0.3 * np.sin(2 * np.pi * 1000 * t) +  # 1kHz tone
        0.2 * np.sin(2 * np.pi * 4000 * t) +  # 4kHz tone
        0.1 * np.sin(2 * np.pi * 8000 * t)    # 8kHz tone  
    )
    
    # Ensure signal is in correct range
    test_signal = test_signal.astype(np.float32)
    
    encoder = Atrac1Encoder()
    decoder = Atrac1Decoder()
    
    encoded_frames = []
    
    # Encode frame by frame
    for frame_idx in range(num_frames):
        start_idx = frame_idx * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame_data = test_signal[start_idx:end_idx]
        
        # Encode mono frame
        compressed_frame = encoder._encode_single_channel(frame_data, 0, frame_idx)
        encoded_frames.append(compressed_frame)
    
    # Decode frame by frame  
    decoded_frames = []
    for frame_idx, compressed_frame in enumerate(encoded_frames):
        decoded_frame = decoder.decode_frame(compressed_frame)
        decoded_frames.append(decoded_frame)
    
    # Concatenate decoded frames
    reconstructed_signal = np.concatenate(decoded_frames)
    
    # Check reconstruction quality
    print(f"Original signal length: {len(test_signal)}")
    print(f"Reconstructed signal length: {len(reconstructed_signal)}")
    
    # Align signals (account for codec delay)
    min_len = min(len(test_signal), len(reconstructed_signal))
    orig_aligned = test_signal[:min_len]
    recon_aligned = reconstructed_signal[:min_len]
    
    # Calculate SNR (ignoring quantization, focusing on TDAC artifacts)
    mse = np.mean((orig_aligned - recon_aligned) ** 2)
    signal_power = np.mean(orig_aligned ** 2)
    
    if signal_power > 0:
        snr_db = 10 * np.log10(signal_power / (mse + 1e-10))
        print(f"SNR: {snr_db:.2f} dB")
        
        # TDAC artifacts typically show up as ~20-40dB degradation
        if snr_db < 20:
            print("⚠️  POSSIBLE TDAC ISSUE: SNR is very low, indicating potential aliasing")
        elif snr_db < 40:
            print("⚠️  POSSIBLE TDAC ISSUE: SNR suggests potential time-domain artifacts")
        else:
            print("✅ TDAC appears to be working correctly")
    
    # Look for frame boundary artifacts
    frame_boundary_errors = []
    for frame_idx in range(1, num_frames):
        boundary_idx = frame_idx * samples_per_frame
        if boundary_idx < min_len - 1:
            # Check for discontinuity at frame boundary
            boundary_error = abs(recon_aligned[boundary_idx] - recon_aligned[boundary_idx - 1])
            frame_boundary_errors.append(boundary_error)
    
    max_boundary_error = max(frame_boundary_errors) if frame_boundary_errors else 0
    avg_boundary_error = np.mean(frame_boundary_errors) if frame_boundary_errors else 0
    
    print(f"Max frame boundary error: {max_boundary_error:.6f}")
    print(f"Avg frame boundary error: {avg_boundary_error:.6f}")
    
    if max_boundary_error > 0.01:  # Threshold for obvious discontinuity
        print("⚠️  FRAME BOUNDARY ISSUE: Large discontinuities detected")
        print("    This indicates TDAC overlap-add may not be working correctly")
    
    return {
        'snr_db': snr_db if signal_power > 0 else float('-inf'),
        'max_boundary_error': max_boundary_error,
        'avg_boundary_error': avg_boundary_error,
        'reconstruction_successful': snr_db > 20 if signal_power > 0 else False
    }

if __name__ == "__main__":
    print("Testing TDAC Perfect Reconstruction...")
    results = test_perfect_reconstruction()
    print(f"\nTest Results: {results}")