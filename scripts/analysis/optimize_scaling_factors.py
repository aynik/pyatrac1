#!/usr/bin/env python3
"""
Optimize scaling factors for better TDAC performance.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def analyze_current_scaling():
    """Analyze current scaling factors across different signals."""
    
    print("=== Current Scaling Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # Test with different signal types
    test_signals = [
        ("DC 0.5", np.ones(128, dtype=np.float32) * 0.5),
        ("DC 1.0", np.ones(128, dtype=np.float32) * 1.0),
        ("Ramp", np.linspace(0, 1, 128, dtype=np.float32)),
        ("Sine 1Hz", np.sin(2 * np.pi * np.arange(128) / 128).astype(np.float32)),
        ("Sine 4Hz", np.sin(2 * np.pi * 4 * np.arange(128) / 128).astype(np.float32)),
        ("Impulse", np.concatenate([np.zeros(64, dtype=np.float32), [1.0], np.zeros(63, dtype=np.float32)])),
    ]
    
    scaling_results = []
    
    for name, signal in test_signals:
        print(f"\n{name}:")
        
        input_energy = np.sum(signal**2)
        input_mean = np.mean(signal)
        input_std = np.std(signal)
        
        # MDCT->IMDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, signal, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Analyze useful output region
        useful_output = low_out[32:144]  # 112 samples
        
        output_energy = np.sum(useful_output**2)
        output_mean = np.mean(useful_output)
        output_std = np.std(useful_output)
        
        # Calculate scaling factors
        energy_scaling = np.sqrt(output_energy / input_energy) if input_energy > 0 else 0
        amplitude_scaling = output_mean / input_mean if input_mean != 0 else np.nan
        
        print(f"  Input: energy={input_energy:.6f}, mean={input_mean:.6f}, std={input_std:.6f}")
        print(f"  Output: energy={output_energy:.6f}, mean={output_mean:.6f}, std={output_std:.6f}")
        print(f"  Energy scaling: {energy_scaling:.6f}")
        print(f"  Amplitude scaling: {amplitude_scaling:.6f}")
        
        scaling_results.append({
            'name': name,
            'energy_scaling': energy_scaling,
            'amplitude_scaling': amplitude_scaling,
            'input_energy': input_energy,
            'output_energy': output_energy
        })
    
    # Analyze scaling consistency
    print(f"\n=== Scaling Consistency Analysis ===")
    
    energy_scalings = [r['energy_scaling'] for r in scaling_results if not np.isnan(r['energy_scaling'])]
    amplitude_scalings = [r['amplitude_scaling'] for r in scaling_results if not np.isnan(r['amplitude_scaling'])]
    
    if energy_scalings:
        energy_mean = np.mean(energy_scalings)
        energy_std = np.std(energy_scalings)
        print(f"Energy scaling: mean={energy_mean:.6f}, std={energy_std:.6f}")
        
        if energy_std < 0.05:
            print("  ✅ Consistent energy scaling")
        else:
            print("  ⚠️  Variable energy scaling")
    
    if amplitude_scalings:
        amplitude_mean = np.mean(amplitude_scalings)
        amplitude_std = np.std(amplitude_scalings)
        print(f"Amplitude scaling: mean={amplitude_mean:.6f}, std={amplitude_std:.6f}")
        
        if amplitude_std < 0.05:
            print("  ✅ Consistent amplitude scaling")
        else:
            print("  ⚠️  Variable amplitude scaling")
    
    return scaling_results

def test_optimal_reconstruction_strategy():
    """Test different reconstruction strategies for best SNR."""
    
    print(f"\n=== Optimal Reconstruction Strategy ===")
    
    mdct = Atrac1MDCT()
    
    # Create test signal with known properties
    freq = 1000
    sample_rate = 44100
    frame_size = 128
    
    # Generate overlapping frames like real TDAC
    num_frames = 3
    overlap_size = 64
    total_samples = (num_frames - 1) * overlap_size + frame_size
    
    t = np.arange(total_samples) / sample_rate
    test_signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    print(f"Test signal: {freq}Hz sine, {total_samples} samples")
    print(f"Overlap strategy: {overlap_size} samples")
    
    # Process overlapping frames
    frame_outputs = []
    
    for frame_idx in range(num_frames):
        start_idx = frame_idx * overlap_size
        end_idx = start_idx + frame_size
        
        if end_idx <= len(test_signal):
            frame_input = test_signal[start_idx:end_idx]
        else:
            frame_input = np.zeros(frame_size, dtype=np.float32)
            available = len(test_signal) - start_idx
            if available > 0:
                frame_input[:available] = test_signal[start_idx:start_idx + available]
        
        # MDCT->IMDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, frame_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=frame_idx)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=frame_idx)
        
        frame_outputs.append(low_out)
        
        print(f"Frame {frame_idx}: energy={np.sum(low_out**2):.6f}")
    
    # Test different reconstruction strategies
    strategies = [
        ("Strategy 1: Simple overlap-add", reconstruct_simple_overlap),
        ("Strategy 2: TDAC windowed", reconstruct_tdac_windowed),
        ("Strategy 3: Middle-only", reconstruct_middle_only),
    ]
    
    best_snr = -float('inf')
    best_strategy = None
    
    for strategy_name, strategy_func in strategies:
        print(f"\n{strategy_name}:")
        
        reconstructed = strategy_func(frame_outputs, overlap_size)
        
        # Compare with original
        comparison_length = min(len(test_signal), len(reconstructed))
        original_section = test_signal[:comparison_length]
        reconstructed_section = reconstructed[:comparison_length]
        
        # Calculate SNR
        error = reconstructed_section - original_section
        error_energy = np.sum(error**2)
        signal_energy = np.sum(original_section**2)
        
        if signal_energy > 0 and error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
            energy_ratio = np.sum(reconstructed_section**2) / signal_energy
            
            print(f"  SNR: {snr_db:.2f} dB")
            print(f"  Energy ratio: {energy_ratio:.6f}")
            
            if snr_db > best_snr:
                best_snr = snr_db
                best_strategy = strategy_name
        else:
            print(f"  Cannot calculate SNR")
    
    print(f"\nBest strategy: {best_strategy} with SNR {best_snr:.2f} dB")
    return best_snr

def reconstruct_simple_overlap(frame_outputs, overlap_size):
    """Simple overlap-add reconstruction."""
    
    total_length = len(frame_outputs[0]) + (len(frame_outputs) - 1) * overlap_size
    reconstructed = np.zeros(total_length, dtype=np.float32)
    
    for i, frame_output in enumerate(frame_outputs):
        start_pos = i * overlap_size
        end_pos = start_pos + len(frame_output)
        
        if end_pos <= len(reconstructed):
            reconstructed[start_pos:end_pos] += frame_output
        else:
            available = len(reconstructed) - start_pos
            reconstructed[start_pos:start_pos + available] += frame_output[:available]
    
    return reconstructed

def reconstruct_tdac_windowed(frame_outputs, overlap_size):
    """TDAC windowed reconstruction using middle regions."""
    
    # Use middle regions from each frame
    frame_middles = []
    for frame_output in frame_outputs:
        middle = frame_output[32:144]  # Core middle region (112 samples)
        frame_middles.append(middle)
    
    # Overlap-add the middle regions
    total_length = len(frame_middles[0]) + (len(frame_middles) - 1) * overlap_size
    reconstructed = np.zeros(total_length, dtype=np.float32)
    
    for i, middle in enumerate(frame_middles):
        start_pos = i * overlap_size
        end_pos = start_pos + len(middle)
        
        if end_pos <= len(reconstructed):
            reconstructed[start_pos:end_pos] += middle
        else:
            available = len(reconstructed) - start_pos
            reconstructed[start_pos:start_pos + available] += middle[:available]
    
    return reconstructed

def reconstruct_middle_only(frame_outputs, overlap_size):
    """Reconstruction using only non-overlapping middle portions."""
    
    # Use only the core part of each middle region to avoid overlap artifacts
    frame_cores = []
    for frame_output in frame_outputs:
        core = frame_output[48:128]  # Avoid edge effects (80 samples)
        frame_cores.append(core)
    
    # Concatenate without overlap
    reconstructed = np.concatenate(frame_cores)
    
    return reconstructed

def test_energy_compensation():
    """Test if energy compensation can improve SNR."""
    
    print(f"\n=== Energy Compensation Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test with known signal
    test_input = np.sin(2 * np.pi * np.arange(128) / 32).astype(np.float32)
    input_energy = np.sum(test_input**2)
    
    print(f"Input energy: {input_energy:.6f}")
    
    # MDCT->IMDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    useful_output = low_out[32:160]  # Match input length
    output_energy = np.sum(useful_output**2)
    
    print(f"Output energy: {output_energy:.6f}")
    print(f"Energy ratio: {output_energy / input_energy:.6f}")
    
    # Test energy compensation
    compensation_factor = np.sqrt(input_energy / output_energy)
    compensated_output = useful_output * compensation_factor
    
    print(f"Compensation factor: {compensation_factor:.6f}")
    print(f"Compensated energy: {np.sum(compensated_output**2):.6f}")
    
    # Calculate SNR with and without compensation
    error_orig = test_input - useful_output
    error_comp = test_input - compensated_output
    
    snr_orig = 10 * np.log10(input_energy / np.sum(error_orig**2)) if np.sum(error_orig**2) > 0 else float('inf')
    snr_comp = 10 * np.log10(input_energy / np.sum(error_comp**2)) if np.sum(error_comp**2) > 0 else float('inf')
    
    print(f"SNR without compensation: {snr_orig:.2f} dB")
    print(f"SNR with compensation: {snr_comp:.2f} dB")
    print(f"SNR improvement: {snr_comp - snr_orig:.2f} dB")
    
    if snr_comp > snr_orig:
        print("✅ Energy compensation improves SNR")
    else:
        print("❌ Energy compensation doesn't help")
    
    return compensation_factor, snr_comp - snr_orig

if __name__ == "__main__":
    scaling_results = analyze_current_scaling()
    best_snr = test_optimal_reconstruction_strategy()
    compensation_factor, snr_improvement = test_energy_compensation()
    
    print(f"\n=== Optimization Summary ===")
    print(f"Best reconstruction SNR: {best_snr:.2f} dB")
    print(f"Energy compensation improvement: {snr_improvement:.2f} dB")
    print(f"Potential optimized SNR: {best_snr + snr_improvement:.2f} dB")
    
    if best_snr + snr_improvement > 10:
        print("🎉 EXCELLENT TDAC PERFORMANCE ACHIEVABLE!")
    elif best_snr + snr_improvement > 5:
        print("✅ Good TDAC performance achievable")
    else:
        print("⚠️  More optimization needed")