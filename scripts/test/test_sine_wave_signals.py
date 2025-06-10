#!/usr/bin/env python3
"""
Test with sine waves at different frequencies as suggested in evaluation.txt
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_sine_waves_by_subband():
    """Test sine waves designed for each QMF subband."""
    
    print("=== Sine Wave Tests by Subband ===")
    
    mdct = Atrac1MDCT()
    
    # Define sine waves for different subbands
    # LOW: 0-5.5 kHz, MID: 5.5-11 kHz, HIGH: 11-22 kHz (assuming 44.1 kHz sample rate)
    
    test_signals = [
        ("DC", np.ones(128, dtype=np.float32) * 0.5),
        ("Low freq (1 kHz)", np.sin(2 * np.pi * 1000/44100 * np.arange(128)).astype(np.float32) * 0.5),
        ("Low freq (3 kHz)", np.sin(2 * np.pi * 3000/44100 * np.arange(128)).astype(np.float32) * 0.5),
        ("Mid freq (7 kHz)", np.sin(2 * np.pi * 7000/44100 * np.arange(128)).astype(np.float32) * 0.5),
        ("High freq (15 kHz)", np.sin(2 * np.pi * 15000/44100 * np.arange(128)).astype(np.float32) * 0.5),
        ("Nyquist/2 (11 kHz)", np.sin(2 * np.pi * 11000/44100 * np.arange(128)).astype(np.float32) * 0.5),
    ]
    
    print("Testing different frequency sine waves:")
    print("(Note: This tests MDCT+IMDCT, not full QMF analysis/synthesis)")
    
    results = []
    
    for name, signal in test_signals:
        input_energy = np.sum(signal**2)
        input_rms = np.sqrt(np.mean(signal**2))
        
        # Full MDCT+IMDCT pipeline
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, signal, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Analyze reconstruction
        final_output = low_out[32:160]  # Account for +32 QMF delay
        output_energy = np.sum(final_output**2)
        output_rms = np.sqrt(np.mean(final_output**2))
        
        # Calculate SNR
        if len(final_output) == len(signal):
            error = final_output - signal
            error_energy = np.sum(error**2)
            
            if error_energy > 0:
                snr_db = 10 * np.log10(input_energy / error_energy)
            else:
                snr_db = float('inf')
        else:
            snr_db = None
        
        # Energy preservation
        energy_ratio = output_energy / input_energy if input_energy > 0 else 0
        rms_ratio = output_rms / input_rms if input_rms > 0 else 0
        
        results.append((name, snr_db, energy_ratio, rms_ratio))
        
        print(f"  {name:20s}: SNR={snr_db:6.2f} dB, energy ratio={energy_ratio:.3f}, RMS ratio={rms_ratio:.3f}")
    
    return results

def test_frequency_sweep():
    """Test a frequency sweep to see response across spectrum."""
    
    print(f"\n=== Frequency Sweep Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test different frequencies
    frequencies = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
    
    print("Frequency response test:")
    
    for freq in frequencies:
        # Generate sine wave
        signal = np.sin(2 * np.pi * freq/44100 * np.arange(128)).astype(np.float32) * 0.5
        input_rms = np.sqrt(np.mean(signal**2))
        
        # Process
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, signal, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        final_output = low_out[32:160]
        output_rms = np.sqrt(np.mean(final_output**2))
        
        # Calculate gain
        gain_db = 20 * np.log10(output_rms / input_rms) if input_rms > 0 else -100
        
        print(f"  {freq:5d} Hz: input RMS={input_rms:.4f}, output RMS={output_rms:.4f}, gain={gain_db:6.2f} dB")

def test_mdct_block_types():
    """Test different MDCT block types (long vs short)."""
    
    print(f"\n=== MDCT Block Type Test ===")
    
    mdct = Atrac1MDCT()
    
    # Test signal
    signal = np.sin(2 * np.pi * 1000/44100 * np.arange(128)).astype(np.float32) * 0.5
    input_energy = np.sum(signal**2)
    
    block_modes = [
        ("All long blocks", BlockSizeMode(False, False, False)),
        ("All short blocks", BlockSizeMode(True, True, True)),
        ("Mixed: low short", BlockSizeMode(True, False, False)),
        ("Mixed: mid short", BlockSizeMode(False, True, False)),
        ("Mixed: high short", BlockSizeMode(False, False, True)),
    ]
    
    print("Testing different block size modes:")
    
    for name, mode in block_modes:
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, signal, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  mode, channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, mode, low_out, mid_out, hi_out, channel=0, frame=0)
        
        final_output = low_out[32:160]
        output_energy = np.sum(final_output**2)
        
        # Calculate SNR
        error = final_output - signal
        error_energy = np.sum(error**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(input_energy / error_energy)
        else:
            snr_db = float('inf')
        
        energy_ratio = output_energy / input_energy
        
        print(f"  {name:20s}: SNR={snr_db:6.2f} dB, energy ratio={energy_ratio:.3f}")

def analyze_dc_vs_sine_performance():
    """Compare DC vs sine wave performance to understand the difference."""
    
    print(f"\n=== DC vs Sine Wave Performance Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # Test signals
    test_cases = [
        ("DC 0.5", np.ones(128, dtype=np.float32) * 0.5),
        ("Sine 1kHz", np.sin(2 * np.pi * 1000/44100 * np.arange(128)).astype(np.float32) * 0.5),
        ("Sine 3kHz", np.sin(2 * np.pi * 3000/44100 * np.arange(128)).astype(np.float32) * 0.5),
    ]
    
    print("Detailed analysis of DC vs sine waves:")
    
    for name, signal in test_cases:
        print(f"\n{name}:")
        
        # Input analysis
        input_mean = np.mean(signal)
        input_std = np.std(signal)
        input_energy = np.sum(signal**2)
        
        print(f"  Input:  mean={input_mean:.6f}, std={input_std:.6f}, energy={input_energy:.6f}")
        
        # Process
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, signal, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        # Check spectral distribution
        low_coeffs = specs[:128]
        dc_coeff = low_coeffs[0]
        ac_energy = np.sum(low_coeffs[1:]**2)
        total_coeff_energy = np.sum(low_coeffs**2)
        
        print(f"  Coeffs: DC={dc_coeff:.6f}, AC energy={ac_energy:.6f}, total={total_coeff_energy:.6f}")
        
        # IMDCT
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Output analysis
        final_output = low_out[32:160]
        output_mean = np.mean(final_output)
        output_std = np.std(final_output)
        output_energy = np.sum(final_output**2)
        
        print(f"  Output: mean={output_mean:.6f}, std={output_std:.6f}, energy={output_energy:.6f}")
        
        # SNR calculation
        error = final_output - signal
        error_energy = np.sum(error**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(input_energy / error_energy)
        else:
            snr_db = float('inf')
        
        print(f"  SNR: {snr_db:.2f} dB")
        
        # Why does this signal perform this way?
        if "DC" in name:
            print(f"  Analysis: DC should have zero AC coefficients, minimal variation")
        else:
            print(f"  Analysis: Sine wave should spread across multiple coefficients")

if __name__ == "__main__":
    sine_results = test_sine_waves_by_subband()
    test_frequency_sweep()
    test_mdct_block_types()
    analyze_dc_vs_sine_performance()
    
    print(f"\n=== SINE WAVE TEST SUMMARY ===")
    print("Key findings:")
    
    # Find best and worst performing signals
    valid_results = [(name, snr) for name, snr, _, _ in sine_results if snr is not None and snr != float('inf')]
    if valid_results:
        best = max(valid_results, key=lambda x: x[1])
        worst = min(valid_results, key=lambda x: x[1])
        
        print(f"Best performing: {best[0]} with {best[1]:.2f} dB SNR")
        print(f"Worst performing: {worst[0]} with {worst[1]:.2f} dB SNR")
    
    print("This confirms frequency-dependent behavior.")
    print("Next: Compare with atracdenc reference output for these same signals.")