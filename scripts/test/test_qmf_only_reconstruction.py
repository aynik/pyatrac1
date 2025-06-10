#!/usr/bin/env python3
"""
Test QMF Analysis + Synthesis only (bypassing MDCT/IMDCT) on pentagramm_4.wav
"""

import numpy as np
import wave
from pyatrac1.core.qmf import Atrac1AnalysisFilterBank, Atrac1SynthesisFilterBank

def read_wav(filename):
    """Read WAV file and return samples."""
    with wave.open(filename, 'rb') as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        
        print(f"Input WAV: {frames} frames, {sample_rate} Hz, {channels} channels, {sample_width*8}-bit")
        
        # Read raw audio data
        raw_data = wav.readframes(frames)
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit
            samples = np.frombuffer(raw_data, dtype=np.int16)
        elif sample_width == 4:  # 32-bit
            samples = np.frombuffer(raw_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert to float32 in range [-1, 1]
        if sample_width == 2:
            samples = samples.astype(np.float32) / 32767.0
        else:
            samples = samples.astype(np.float32) / 2147483647.0
        
        # Handle stereo by taking left channel only
        if channels == 2:
            samples = samples[::2]  # Take every other sample (left channel)
            print(f"Converted stereo to mono: {len(samples)} samples")
        
        return samples, sample_rate

def write_wav(filename, samples, sample_rate):
    """Write samples to WAV file."""
    # Convert float32 back to 16-bit PCM
    samples_16bit = np.clip(samples * 32767.0, -32767, 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(samples_16bit.tobytes())
    
    print(f"Wrote {len(samples)} samples to {filename}")

def test_qmf_only_reconstruction():
    """Test QMF Analysis → Synthesis only (no MDCT)"""
    
    print("=== QMF-Only Reconstruction Test ===")
    
    # Read input file
    input_samples, sample_rate = read_wav("pentagramm_4.wav")
    print(f"Loaded {len(input_samples)} samples at {sample_rate} Hz")
    
    # Initialize QMF components only
    qmf_analysis = Atrac1AnalysisFilterBank()
    qmf_synthesis = Atrac1SynthesisFilterBank()
    
    # Process in 512-sample frames (ATRAC1 frame size)
    frame_size = 512
    num_frames = len(input_samples) // frame_size
    
    print(f"Processing {num_frames} frames of {frame_size} samples each")
    print("Pipeline: QMF Analysis → QMF Synthesis (no MDCT/IMDCT)")
    
    # Initialize output buffer
    output_samples = np.zeros(len(input_samples), dtype=np.float32)
    
    # Process each frame
    for frame_idx in range(num_frames):
        start_idx = frame_idx * frame_size
        end_idx = start_idx + frame_size
        
        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}/{num_frames}")
        
        # Get input frame
        input_frame = input_samples[start_idx:end_idx]
        
        # QMF Analysis: 512 samples → 3 subbands (128, 128, 256 samples)
        qmf_low, qmf_mid, qmf_hi = qmf_analysis.analysis(input_frame.tolist(), frame_idx)
        
        # Convert to numpy arrays for inspection
        qmf_low = np.array(qmf_low, dtype=np.float32)
        qmf_mid = np.array(qmf_mid, dtype=np.float32)
        qmf_hi = np.array(qmf_hi, dtype=np.float32)
        
        # Log subband info for first frame
        if frame_idx == 0:
            print(f"  QMF Analysis output:")
            print(f"    Low band: {len(qmf_low)} samples, RMS={np.sqrt(np.mean(qmf_low**2)):.6f}")
            print(f"    Mid band: {len(qmf_mid)} samples, RMS={np.sqrt(np.mean(qmf_mid**2)):.6f}")
            print(f"    Hi band:  {len(qmf_hi)} samples, RMS={np.sqrt(np.mean(qmf_hi**2)):.6f}")
        
        # QMF Synthesis: 3 subbands → 512 samples
        # Pass subbands directly back to synthesis (perfect reconstruction test)
        recon_frame = qmf_synthesis.synthesis(qmf_low.tolist(), qmf_mid.tolist(), qmf_hi.tolist())
        
        # Convert to numpy array and store reconstructed frame
        recon_frame = np.array(recon_frame, dtype=np.float32)
        output_samples[start_idx:end_idx] = recon_frame
    
    # Save reconstructed audio
    write_wav("pentagramm_4_qmf_only.wav", output_samples, sample_rate)
    
    # Calculate reconstruction quality
    input_energy = np.sum(input_samples**2)
    output_energy = np.sum(output_samples**2)
    
    # Calculate SNR (limit to processed frames)
    processed_samples = num_frames * frame_size
    input_processed = input_samples[:processed_samples]
    output_processed = output_samples[:processed_samples]
    
    error = output_processed - input_processed
    error_energy = np.sum(error**2)
    signal_energy = np.sum(input_processed**2)
    
    if error_energy > 0:
        snr_db = 10 * np.log10(signal_energy / error_energy)
    else:
        snr_db = float('inf')
    
    print(f"\nQMF-Only Reconstruction Quality:")
    print(f"  Input energy: {input_energy:.6f}")
    print(f"  Output energy: {output_energy:.6f}")
    print(f"  Energy ratio: {output_energy/input_energy:.6f}")
    print(f"  SNR: {snr_db:.2f} dB")
    
    # RMS comparison
    input_rms = np.sqrt(np.mean(input_processed**2))
    output_rms = np.sqrt(np.mean(output_processed**2))
    print(f"  Input RMS: {input_rms:.6f}")
    print(f"  Output RMS: {output_rms:.6f}")
    print(f"  RMS ratio: {output_rms/input_rms:.6f}")
    
    # Signal level comparison
    input_max = np.max(np.abs(input_processed))
    output_max = np.max(np.abs(output_processed))
    print(f"  Input peak: {input_max:.6f}")
    print(f"  Output peak: {output_max:.6f}")
    print(f"  Peak ratio: {output_max/input_max:.6f}")
    
    # Quality assessment
    if snr_db > 40:
        print(f"  ✅ EXCELLENT: High quality QMF reconstruction!")
    elif snr_db > 20:
        print(f"  ✅ GOOD: Acceptable QMF reconstruction quality")
    elif snr_db > 10:
        print(f"  ⚠️  FAIR: Moderate QMF reconstruction quality")
    elif snr_db > 0:
        print(f"  ❌ POOR: Low QMF reconstruction quality")
    else:
        print(f"  ❌ VERY POOR: QMF reconstruction worse than input")
    
    # Detailed error analysis
    print(f"\nDetailed Error Analysis:")
    error_rms = np.sqrt(np.mean(error**2))
    error_max = np.max(np.abs(error))
    print(f"  Error RMS: {error_rms:.6f}")
    print(f"  Error peak: {error_max:.6f}")
    print(f"  Error ratio to signal: {error_rms/input_rms:.6f}")
    
    # Check for DC bias
    input_dc = np.mean(input_processed)
    output_dc = np.mean(output_processed)
    dc_error = output_dc - input_dc
    print(f"  Input DC: {input_dc:.6f}")
    print(f"  Output DC: {output_dc:.6f}")
    print(f"  DC error: {dc_error:.6f}")
    
    print(f"\nFiles created:")
    print(f"  Input: pentagramm_4.wav")
    print(f"  QMF-only output: pentagramm_4_qmf_only.wav")
    print(f"  Listen to both files to assess QMF filter bank quality.")
    
    return snr_db, output_energy/input_energy

def test_qmf_with_different_signals():
    """Test QMF with simpler signals to validate implementation."""
    
    print(f"\n=== QMF Test with Simple Signals ===")
    
    qmf_analysis = Atrac1AnalysisFilterBank()
    qmf_synthesis = Atrac1SynthesisFilterBank()
    
    # Test signals
    test_cases = [
        ("DC", np.ones(512, dtype=np.float32) * 0.5),
        ("1kHz sine", np.sin(2 * np.pi * 1000/44100 * np.arange(512)).astype(np.float32) * 0.5),
        ("Low freq (500Hz)", np.sin(2 * np.pi * 500/44100 * np.arange(512)).astype(np.float32) * 0.5),
        ("Mid freq (5kHz)", np.sin(2 * np.pi * 5000/44100 * np.arange(512)).astype(np.float32) * 0.5),
        ("High freq (15kHz)", np.sin(2 * np.pi * 15000/44100 * np.arange(512)).astype(np.float32) * 0.5),
    ]
    
    print("Testing QMF reconstruction with simple signals:")
    
    for name, signal in test_cases:
        # QMF Analysis → Synthesis
        qmf_low, qmf_mid, qmf_hi = qmf_analysis.analysis(signal.tolist(), 0)
        recon = qmf_synthesis.synthesis(qmf_low, qmf_mid, qmf_hi)
        recon = np.array(recon, dtype=np.float32)
        
        # Calculate SNR
        error = recon - signal
        error_energy = np.sum(error**2)
        signal_energy = np.sum(signal**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
        else:
            snr_db = float('inf')
        
        # Energy ratio
        recon_energy = np.sum(recon**2)
        energy_ratio = recon_energy / signal_energy if signal_energy > 0 else 0
        
        print(f"  {name:20s}: SNR={snr_db:6.2f} dB, energy ratio={energy_ratio:.3f}")

if __name__ == "__main__":
    try:
        # Test QMF-only reconstruction
        snr, energy_ratio = test_qmf_only_reconstruction()
        
        # Test with simple signals
        test_qmf_with_different_signals()
        
        print(f"\n=== QMF-ONLY TEST SUMMARY ===")
        print(f"QMF Analysis → Synthesis reconstruction:")
        print(f"  Final SNR: {snr:.2f} dB")
        print(f"  Energy preservation: {energy_ratio:.3f}")
        
        if snr > 20:
            print(f"  ✅ QMF filter bank is working well!")
            print(f"  The quality issues are likely in MDCT/IMDCT pipeline.")
        elif snr > 0:
            print(f"  ⚠️  QMF has some issues but basic functionality works.")
            print(f"  Need to investigate QMF filter coefficients or implementation.")
        else:
            print(f"  ❌ QMF filter bank has serious issues.")
            print(f"  This explains the poor full pipeline quality.")
        
    except Exception as e:
        print(f"Error during QMF-only test: {e}")
        import traceback
        traceback.print_exc()