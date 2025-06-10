#!/usr/bin/env python3
"""
Test if TDAC aliasing is resolved with corrected MDCT/IMDCT transforms.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode
from pyatrac1.core.qmf import Atrac1AnalysisFilterBank, Atrac1SynthesisFilterBank

def test_tdac_aliasing_resolution():
    """Test TDAC aliasing with corrected MDCT/IMDCT transforms."""
    
    print("=== TDAC Aliasing Resolution Test ===")
    
    # Create test audio - sine wave that should reconstruct cleanly
    frame_size = 512
    freq = 1000  # 1kHz sine wave
    sample_rate = 44100
    
    # Generate 3 frames of audio for overlap testing
    total_samples = frame_size * 3
    t = np.arange(total_samples) / sample_rate
    test_audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    print(f"Test signal: {freq}Hz sine wave, {total_samples} samples")
    print(f"Input energy: {np.sum(test_audio**2):.6f}")
    
    # Initialize ATRAC1 components
    analysis_filter = Atrac1AnalysisFilterBank()
    synthesis_filter = Atrac1SynthesisFilterBank()
    mdct = Atrac1MDCT()
    
    # Use long blocks (typical for good quality)
    block_size_mode = BlockSizeMode(
        low_band_short=False,   # 128-point MDCT
        mid_band_short=False,   # 128-point MDCT  
        high_band_short=False   # 256-point MDCT
    )
    
    # Process multiple frames to test TDAC overlap
    output_frames = []
    
    for frame_idx in range(3):
        print(f"\nProcessing frame {frame_idx}:")
        
        # Extract frame
        start_idx = frame_idx * frame_size
        end_idx = start_idx + frame_size
        frame_audio = test_audio[start_idx:end_idx]
        
        # Create stereo (duplicate mono)
        frame_stereo = np.array([frame_audio, frame_audio])
        
        # QMF Analysis
        qmf_bands = analysis_filter.analysis(frame_stereo[0])
        low_band = np.array(qmf_bands[0], dtype=np.float32)     # Low band
        mid_band = np.array(qmf_bands[1], dtype=np.float32)     # Mid band  
        high_band = np.array(qmf_bands[2], dtype=np.float32)    # High band
        
        print(f"  QMF bands: Low={len(low_band)}, Mid={len(mid_band)}, High={len(high_band)}")
        
        # MDCT Analysis
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_band, mid_band, high_band, block_size_mode, channel=0, frame=frame_idx)
        
        print(f"  MDCT coefficients energy: {np.sum(specs**2):.6f}")
        print(f"  MDCT coefficients max: {np.max(np.abs(specs)):.6f}")
        
        # IMDCT Synthesis (adjust sizes to match QMF expectations)
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        high_out = np.zeros(256, dtype=np.float32)  # QMF expects 256, not 512
        
        mdct.imdct(specs, block_size_mode, low_out, mid_out, high_out, channel=0, frame=frame_idx)
        
        print(f"  IMDCT output energy: Low={np.sum(low_out**2):.6f}, Mid={np.sum(mid_out**2):.6f}, High={np.sum(high_out**2):.6f}")
        
        # QMF Synthesis
        output_mono = synthesis_filter.synthesis(low_out.tolist(), mid_out.tolist(), high_out.tolist())
        
        print(f"  QMF synthesis output energy: {np.sum(output_mono**2):.6f}")
        print(f"  Frame reconstruction max: {np.max(np.abs(output_mono)):.6f}")
        
        output_frames.append(output_mono)
    
    # Concatenate output frames
    reconstructed_audio = np.concatenate(output_frames)
    
    print(f"\n=== Overall Results ===")
    print(f"Input length: {len(test_audio)}")
    print(f"Output length: {len(reconstructed_audio)}")
    print(f"Input energy: {np.sum(test_audio**2):.6f}")
    print(f"Output energy: {np.sum(reconstructed_audio**2):.6f}")
    print(f"Energy ratio: {np.sum(reconstructed_audio**2) / np.sum(test_audio**2):.6f}")
    
    # Check reconstruction quality
    if len(reconstructed_audio) >= len(test_audio):
        # Compare same length sections
        comparison_length = min(len(test_audio), len(reconstructed_audio))
        input_section = test_audio[:comparison_length]
        output_section = reconstructed_audio[:comparison_length]
        
        # Calculate SNR
        error = output_section - input_section
        signal_power = np.sum(input_section**2)
        noise_power = np.sum(error**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            print(f"SNR: {snr_db:.2f} dB")
            
            if snr_db > 40:
                print("✅ EXCELLENT reconstruction (SNR > 40 dB)")
            elif snr_db > 20:
                print("✅ GOOD reconstruction (SNR > 20 dB)")  
            elif snr_db > 10:
                print("⚠️  FAIR reconstruction (SNR > 10 dB)")
            else:
                print("❌ POOR reconstruction (SNR < 10 dB)")
        else:
            print("✅ PERFECT reconstruction (zero error)")
    
    # Check for frame boundary artifacts
    print(f"\n=== Frame Boundary Analysis ===")
    frame_boundaries = [frame_size, frame_size * 2]
    
    for boundary in frame_boundaries:
        if boundary < len(reconstructed_audio):
            # Check samples around frame boundary
            window_size = 10
            start_idx = max(0, boundary - window_size)
            end_idx = min(len(reconstructed_audio), boundary + window_size)
            
            boundary_samples = reconstructed_audio[start_idx:end_idx]
            boundary_variation = np.std(boundary_samples)
            
            print(f"Frame boundary at sample {boundary}:")
            print(f"  Samples: {boundary_samples}")
            print(f"  Variation (std): {boundary_variation:.6f}")
            
            if boundary_variation < 0.1:
                print(f"  ✅ Smooth transition")
            else:
                print(f"  ⚠️  Potential discontinuity")

def test_impulse_response():
    """Test impulse response to detect aliasing artifacts."""
    
    print(f"\n=== Impulse Response Test ===")
    
    # Single impulse in the middle of a frame
    frame_size = 512
    impulse_frame = np.zeros(frame_size, dtype=np.float32)
    impulse_frame[frame_size // 2] = 1.0
    
    print(f"Testing impulse response...")
    
    # Process through full pipeline
    analysis_filter = Atrac1AnalysisFilterBank()
    synthesis_filter = Atrac1SynthesisFilterBank()
    mdct = Atrac1MDCT()
    
    block_size_mode = BlockSizeMode(False, False, False)  # Long blocks
    
    # QMF Analysis
    qmf_bands = analysis_filter.analysis(impulse_frame)
    low_band = np.array(qmf_bands[0], dtype=np.float32)
    mid_band = np.array(qmf_bands[1], dtype=np.float32)
    high_band = np.array(qmf_bands[2], dtype=np.float32)
    
    # MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_band, mid_band, high_band, block_size_mode)
    
    # IMDCT
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32) 
    high_out = np.zeros(256, dtype=np.float32)
    
    mdct.imdct(specs, block_size_mode, low_out, mid_out, high_out)
    
    # QMF Synthesis
    output_mono = synthesis_filter.synthesis(low_out.tolist(), mid_out.tolist(), high_out.tolist())
    
    # Analyze impulse response
    peak_idx = np.argmax(np.abs(output_mono))
    peak_val = output_mono[peak_idx]
    
    print(f"Input impulse at sample {frame_size // 2}, amplitude 1.0")
    print(f"Output peak at sample {peak_idx}, amplitude {peak_val:.6f}")
    print(f"Peak shift: {peak_idx - frame_size // 2} samples")
    
    # Check for ringing/aliasing artifacts
    # Look for significant values far from the main peak
    artifact_threshold = 0.1 * abs(peak_val)
    artifacts = []
    
    for i, val in enumerate(output_mono):
        if abs(val) > artifact_threshold and abs(i - peak_idx) > 50:
            artifacts.append((i, val))
    
    print(f"Artifacts (>{artifact_threshold:.3f}): {len(artifacts)}")
    if len(artifacts) > 0:
        print(f"  Artifact samples: {artifacts[:5]}")  # Show first 5
        print("  ⚠️  Possible aliasing artifacts detected")
    else:
        print("  ✅ No significant artifacts detected")

if __name__ == "__main__":
    test_tdac_aliasing_resolution()
    test_impulse_response()