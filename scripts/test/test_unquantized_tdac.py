#!/usr/bin/env python3
"""
Test TDAC without quantization to isolate the source of artifacts.
"""

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.core.decoder import Atrac1Decoder
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode
from pyatrac1.core.qmf import Atrac1SynthesisFilterBank

def test_unquantized_reconstruction():
    """Test encoder pipeline but skip quantization."""
    
    encoder = Atrac1Encoder()
    synthesis_filter = Atrac1SynthesisFilterBank()
    
    # Create test signal
    num_frames = 4
    samples_per_frame = 512
    total_samples = num_frames * samples_per_frame
    
    t = np.linspace(0, total_samples / 44100, total_samples, endpoint=False)
    test_signal = 0.1 * np.sin(2 * np.pi * 1000 * t)  # Simple 1kHz sine
    test_signal = test_signal.astype(np.float32)
    
    reconstructed_frames = []
    
    # CRITICAL: Create persistent buffers that maintain state across frames (like atracdenc)
    low_out = np.zeros(256 + 16, dtype=np.float32)
    mid_out = np.zeros(256 + 16, dtype=np.float32)
    hi_out = np.zeros(512 + 16, dtype=np.float32)
    
    # Also need persistent encoder buffers for MDCT overlap
    # The encoder maintains PcmBufLow/Mid/Hi that persist across frames
    
    for frame_idx in range(num_frames):
        start_idx = frame_idx * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame_data = test_signal[start_idx:end_idx]
        
        # Do QMF analysis - this fills encoder's persistent PcmBuf* arrays
        low, mid, hi = encoder.qmf_filter_bank_ch0.analysis(frame_data)
        
        # CRITICAL: Copy QMF output into encoder's persistent buffers
        # This is what the real encoder does - it copies QMF output into PcmBufLow/Mid/Hi
        encoder.mdct_processor.pcm_buf_low[0][:128] = low
        encoder.mdct_processor.pcm_buf_mid[0][:128] = mid
        encoder.mdct_processor.pcm_buf_hi[0][:256] = hi
        
        # Do MDCT using the persistent buffers
        specs = np.zeros(512, dtype=np.float32)
        # Use only long blocks for simplicity
        block_size_mode = BlockSizeMode(low_band_short=False, mid_band_short=False, high_band_short=False)
        encoder.mdct_processor.mdct(specs, 
                                   encoder.mdct_processor.pcm_buf_low[0], 
                                   encoder.mdct_processor.pcm_buf_mid[0], 
                                   encoder.mdct_processor.pcm_buf_hi[0], 
                                   block_size_mode, channel=0, frame=frame_idx)
        
        # Skip quantization - go directly to IMDCT with persistent buffers
        encoder.mdct_processor.imdct(specs, block_size_mode, low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        # QMF synthesis
        reconstructed_frame = synthesis_filter.synthesis(low_out[:128].tolist(), mid_out[:128].tolist(), hi_out[:256].tolist())
        reconstructed_frames.append(reconstructed_frame)
    
    # Concatenate results
    reconstructed_signal = np.concatenate(reconstructed_frames)
    
    # Align and compare
    min_len = min(len(test_signal), len(reconstructed_signal))
    orig_aligned = test_signal[:min_len]
    recon_aligned = reconstructed_signal[:min_len]
    
    mse = np.mean((orig_aligned - recon_aligned) ** 2)
    signal_power = np.mean(orig_aligned ** 2)
    
    if signal_power > 0:
        snr_db = 10 * np.log10(signal_power / (mse + 1e-10))
        print(f"Unquantized SNR: {snr_db:.2f} dB")
        
        if snr_db > 60:
            print("✅ Unquantized reconstruction is excellent")
        elif snr_db > 40:
            print("⚠️  Unquantized reconstruction has minor issues")
        else:
            print("❌ Unquantized reconstruction has major issues - TDAC problem likely")
    
    # Check frame boundaries
    frame_boundary_errors = []
    for frame_idx in range(1, num_frames):
        boundary_idx = frame_idx * samples_per_frame
        if boundary_idx < min_len - 1:
            boundary_error = abs(recon_aligned[boundary_idx] - recon_aligned[boundary_idx - 1])
            frame_boundary_errors.append(boundary_error)
    
    max_boundary_error = max(frame_boundary_errors) if frame_boundary_errors else 0
    print(f"Unquantized max boundary error: {max_boundary_error:.6f}")
    
    return snr_db if signal_power > 0 else float('-inf')

if __name__ == "__main__":
    print("Testing unquantized TDAC reconstruction...")
    snr = test_unquantized_reconstruction()