#!/usr/bin/env python3
"""
Test full QMF + MDCT → IMDCT + QMF pipeline reconstruction on pentagramm_4.wav
"""

import numpy as np
import wave
import struct
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode
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

def test_full_pipeline_reconstruction():
    """Test full pipeline: QMF Analysis → MDCT → IMDCT → QMF Synthesis"""
    
    print("=== Full Pipeline Reconstruction Test ===")
    
    # Read input file
    input_samples, sample_rate = read_wav("pentagramm_4.wav")
    print(f"Loaded {len(input_samples)} samples at {sample_rate} Hz")
    
    # Initialize components
    qmf_analysis = Atrac1AnalysisFilterBank()
    qmf_synthesis = Atrac1SynthesisFilterBank()
    mdct = Atrac1MDCT()
    
    # Process in 512-sample frames (ATRAC1 frame size)
    frame_size = 512
    num_frames = len(input_samples) // frame_size
    
    print(f"Processing {num_frames} frames of {frame_size} samples each")
    
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
        
        # Convert to numpy arrays
        qmf_low = np.array(qmf_low, dtype=np.float32)
        qmf_mid = np.array(qmf_mid, dtype=np.float32)
        qmf_hi = np.array(qmf_hi, dtype=np.float32)
        
        # MDCT: Transform each subband to spectral coefficients
        specs = np.zeros(512, dtype=np.float32)  # Total coefficients (128+128+256)
        
        # Use all long blocks for simplicity (no transient detection)
        block_mode = BlockSizeMode(False, False, False)
        
        # Forward MDCT
        mdct.mdct(specs, qmf_low, qmf_mid, qmf_hi, block_mode, channel=0, frame=frame_idx)
        
        # At this point we would normally do:
        # 1. Psychoacoustic analysis
        # 2. Bit allocation 
        # 3. Quantization
        # But we're skipping quantization to test reconstruction quality
        
        # IMDCT: Spectral coefficients back to subbands
        imdct_low = np.zeros(256, dtype=np.float32)   # 128 + 16 overlap
        imdct_mid = np.zeros(256, dtype=np.float32)   # 128 + 16 overlap
        imdct_hi = np.zeros(512, dtype=np.float32)    # 256 + 16 overlap
        
        # Inverse MDCT
        mdct.imdct(specs, block_mode, imdct_low, imdct_mid, imdct_hi, channel=0, frame=frame_idx)
        
        # Extract main data (without overlap regions)
        recon_low = imdct_low[:128]
        recon_mid = imdct_mid[:128] 
        recon_hi = imdct_hi[:256]
        
        # QMF Synthesis: 3 subbands → 512 samples
        recon_frame = qmf_synthesis.synthesis(recon_low.tolist(), recon_mid.tolist(), recon_hi.tolist())
        
        # Convert to numpy array and store reconstructed frame
        recon_frame = np.array(recon_frame, dtype=np.float32)
        output_samples[start_idx:end_idx] = recon_frame
    
    # Save reconstructed audio
    write_wav("pentagramm_4_recon.wav", output_samples, sample_rate)
    
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
    
    print(f"\nReconstruction Quality Analysis:")
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
    
    if snr_db > 20:
        print(f"  ✅ EXCELLENT: High quality reconstruction!")
    elif snr_db > 10:
        print(f"  ✅ GOOD: Acceptable reconstruction quality")
    elif snr_db > 5:
        print(f"  ⚠️  FAIR: Moderate reconstruction quality")
    else:
        print(f"  ❌ POOR: Low reconstruction quality")
    
    print(f"\nFiles created:")
    print(f"  Input: pentagramm_4.wav")
    print(f"  Output: pentagramm_4_recon.wav")
    print(f"  You can listen to both files to compare quality.")
    
    return snr_db, output_energy/input_energy

if __name__ == "__main__":
    try:
        snr, energy_ratio = test_full_pipeline_reconstruction()
        
        print(f"\n=== PIPELINE TEST SUMMARY ===")
        print(f"Successfully reconstructed pentagramm_4.wav without quantization")
        print(f"Final SNR: {snr:.2f} dB")
        print(f"Energy preservation: {energy_ratio:.3f}")
        print(f"This demonstrates the quality of our MDCT/IMDCT implementation")
        print(f"on real audio content before quantization losses.")
        
    except ImportError as e:
        print(f"Error: Missing QMF implementation: {e}")
        print(f"Please ensure QMF classes are available in pyatrac1.core.qmf")
    except Exception as e:
        print(f"Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()