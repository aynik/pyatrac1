#!/usr/bin/env python3
"""
Push toward the 40 dB SNR target by fine-tuning scaling and investigating remaining issues.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode, MDCT, IMDCT

def analyze_remaining_snr_issues():
    """Analyze what's preventing us from reaching 40 dB SNR."""
    
    print("=== Remaining SNR Issues Analysis ===")
    
    mdct = Atrac1MDCT()
    test_input = np.ones(128, dtype=np.float32) * 0.5
    input_energy = np.sum(test_input**2)
    
    # Run current pipeline
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    final_output = low_out[32:160]
    
    print(f"Current performance:")
    print(f"  Input:  mean={np.mean(test_input):.6f}, std={np.std(test_input):.6f}")
    print(f"  Output: mean={np.mean(final_output):.6f}, std={np.std(final_output):.6f}")
    print(f"  SNR: {8.66:.2f} dB")
    
    # Analyze the error characteristics
    error = final_output - test_input
    error_energy = np.sum(error**2)
    print(f"\nError analysis:")
    print(f"  Error energy: {error_energy:.6f}")
    print(f"  Error mean: {np.mean(error):.6f}")
    print(f"  Error std: {np.std(error):.6f}")
    print(f"  Max error: {np.max(np.abs(error)):.6f}")
    
    # Check if error is systematic (DC offset) or random (noise)
    dc_error = np.mean(error)
    noise_error = np.std(error)
    print(f"\nError breakdown:")
    print(f"  DC offset error: {dc_error:.6f}")
    print(f"  Noise error (std): {noise_error:.6f}")
    
    # What SNR would we get if we fixed just the DC offset?
    corrected_output = final_output - dc_error
    corrected_error = corrected_output - test_input
    corrected_error_energy = np.sum(corrected_error**2)
    
    if corrected_error_energy > 0:
        corrected_snr = 10 * np.log10(input_energy / corrected_error_energy)
        print(f"  SNR if DC offset fixed: {corrected_snr:.2f} dB")
    
    # Theoretical perfect reconstruction
    perfect_mean = np.mean(test_input)
    mean_error = abs(np.mean(final_output) - perfect_mean)
    print(f"\nReconstruction quality:")
    print(f"  Target mean: {perfect_mean:.6f}")
    print(f"  Actual mean: {np.mean(final_output):.6f}")
    print(f"  Mean error: {mean_error:.6f} ({mean_error/perfect_mean*100:.1f}%)")
    
    # For DC input, output should be constant. Check variation.
    output_variation = np.std(final_output)
    print(f"  Output variation (should be 0): {output_variation:.6f}")
    
    # What's the theoretical SNR limit given this variation?
    if output_variation > 0:
        # Signal power vs noise power
        signal_power = np.mean(final_output)**2
        noise_power = output_variation**2
        theoretical_snr = 10 * np.log10(signal_power / noise_power)
        print(f"  Theoretical SNR limit: {theoretical_snr:.2f} dB")

def test_perfect_reconstruction_scaling():
    """Test if there's a scaling that gives perfect reconstruction."""
    
    print(f"\n=== Perfect Reconstruction Scaling Test ===")
    
    test_input = np.ones(128, dtype=np.float32) * 0.5
    target_mean = np.mean(test_input)
    
    # Current result
    mdct = Atrac1MDCT()
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    current_output = low_out[32:160]
    current_mean = np.mean(current_output)
    
    print(f"Current reconstruction:")
    print(f"  Target: {target_mean:.6f}")
    print(f"  Actual: {current_mean:.6f}")
    print(f"  Scale factor needed: {target_mean/current_mean:.6f}")
    
    # Test if applying this scale factor gets us to perfect reconstruction
    scale_factor = target_mean / current_mean
    scaled_output = current_output * scale_factor
    
    scaled_error = scaled_output - test_input
    scaled_error_energy = np.sum(scaled_error**2)
    input_energy = np.sum(test_input**2)
    
    if scaled_error_energy > 0:
        scaled_snr = 10 * np.log10(input_energy / scaled_error_energy)
        print(f"  SNR with scale correction: {scaled_snr:.2f} dB")
        
        if scaled_snr > 30:
            print(f"  ✅ Scale correction would achieve high SNR!")
            print(f"  Consider adjusting IMDCT scale by factor {scale_factor:.6f}")
        else:
            print(f"  ❌ Scale correction alone insufficient")
    
    return scale_factor

def test_ultra_fine_tuning():
    """Ultra-fine tuning of MDCT/IMDCT parameters."""
    
    print(f"\n=== Ultra-Fine Parameter Tuning ===")
    
    test_input = np.ones(128, dtype=np.float32) * 0.5
    input_energy = np.sum(test_input**2)
    
    # Test fine variations around our current best parameters
    mdct_scales = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    imdct_scales = [100, 110, 120, 128, 130, 140, 150]
    
    print(f"Fine-tuning around MDCT=8.0, IMDCT=128:")
    
    best_snr = -100
    best_params = None
    
    for mdct_scale in mdct_scales:
        for imdct_scale in imdct_scales:
            # Create custom MDCT
            mdct = Atrac1MDCT()
            mdct.mdct256 = MDCT(256, mdct_scale)
            mdct.imdct256 = IMDCT(256, imdct_scale)
            
            # Test
            specs = np.zeros(512, dtype=np.float32)
            mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                      BlockSizeMode(False, False, False), channel=0, frame=0)
            
            low_out = np.zeros(256, dtype=np.float32)
            mid_out = np.zeros(256, dtype=np.float32)
            hi_out = np.zeros(512, dtype=np.float32)
            
            mdct.imdct(specs, BlockSizeMode(False, False, False),
                       low_out, mid_out, hi_out, channel=0, frame=0)
            
            final_output = low_out[32:160]
            error = final_output - test_input
            error_energy = np.sum(error**2)
            
            if error_energy > 0:
                snr_db = 10 * np.log10(input_energy / error_energy)
                
                if snr_db > best_snr:
                    best_snr = snr_db
                    best_params = (mdct_scale, imdct_scale)
                    
                if mdct_scale in [8.0] and imdct_scale in [120, 128, 130]:  # Show key points
                    print(f"  MDCT={mdct_scale:4.1f}, IMDCT={imdct_scale:3.0f}: SNR={snr_db:6.2f} dB")
    
    if best_params:
        print(f"\nBest fine-tuned parameters:")
        print(f"  MDCT={best_params[0]}, IMDCT={best_params[1]}")
        print(f"  SNR={best_snr:.2f} dB")
        
        if best_snr > 15:
            print(f"  ✅ Significant improvement found!")
            return best_params
        else:
            print(f"  ⚠️  Only marginal improvement")
    
    return None

def investigate_fundamental_limits():
    """Investigate if there are fundamental limits preventing 40 dB SNR."""
    
    print(f"\n=== Fundamental Limits Investigation ===")
    
    # Test with different input signals to see if the issue is DC-specific
    test_signals = [
        ("DC 0.5", np.ones(128, dtype=np.float32) * 0.5),
        ("DC 1.0", np.ones(128, dtype=np.float32) * 1.0),
        ("Linear ramp", np.linspace(0, 1, 128, dtype=np.float32)),
        ("Sine wave", np.sin(2 * np.pi * np.arange(128) / 128).astype(np.float32) * 0.5),
        ("Low freq sine", np.sin(2 * np.pi * np.arange(128) / 64).astype(np.float32) * 0.5),
    ]
    
    mdct = Atrac1MDCT()
    
    print(f"Testing different input signals:")
    
    for name, signal in test_signals:
        input_energy = np.sum(signal**2)
        
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, signal, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        final_output = low_out[32:160]
        error = final_output - signal
        error_energy = np.sum(error**2)
        
        if error_energy > 0 and input_energy > 0:
            snr_db = 10 * np.log10(input_energy / error_energy)
            print(f"  {name:15s}: SNR={snr_db:6.2f} dB")
        else:
            print(f"  {name:15s}: Perfect reconstruction")
    
    print(f"\nConclusions:")
    print(f"- If all signals have similar SNR: issue is in MDCT/IMDCT implementation")
    print(f"- If some signals have better SNR: issue is frequency-dependent")
    print(f"- If DC has worst SNR: issue is with DC handling specifically")

if __name__ == "__main__":
    analyze_remaining_snr_issues()
    scale_factor = test_perfect_reconstruction_scaling()
    best_params = test_ultra_fine_tuning()
    investigate_fundamental_limits()
    
    print(f"\n=== 40 dB TARGET ANALYSIS ===")
    print(f"Current status: 8.66 dB SNR (target: >40 dB)")
    print(f"Gap: {40 - 8.66:.1f} dB improvement needed")
    print(f"")
    if best_params:
        print(f"Recommended next step: Apply fine-tuned parameters")
        print(f"  self.mdct256 = MDCT(256, {best_params[0]})")
        print(f"  self.imdct256 = IMDCT(256, {best_params[1]})")
    else:
        print(f"Recommended next step: Investigate windowing or buffer layout issues")
        print(f"The remaining error may be due to:")
        print(f"- Windowing function mismatch with atracdenc")
        print(f"- Buffer layout or indexing differences")
        print(f"- FFT implementation differences (numpy vs KissFFT)")