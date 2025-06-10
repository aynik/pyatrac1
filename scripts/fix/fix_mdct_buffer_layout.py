#!/usr/bin/env python3
"""
Fix the MDCT buffer layout to match atracdenc expectations.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, MDCT

def test_mdct_buffer_expectations():
    """Test what buffer layout MDCT expects for proper energy transfer."""
    
    print("=== MDCT Buffer Layout Expectations ===")
    
    mdct = Atrac1MDCT()
    
    # Test different buffer layouts with the same energy
    test_energy = 32.0
    test_dc_level = 0.5
    
    layouts = [
        ("Centered [64:192]", lambda: create_centered_buffer(test_dc_level)),
        ("Front-loaded [0:128]", lambda: create_front_buffer(test_dc_level)),
        ("Back-loaded [128:256]", lambda: create_back_buffer(test_dc_level)),
        ("Our current [80:208]", lambda: create_current_buffer(test_dc_level)),
        ("atracdenc style [0:256] full", lambda: create_full_buffer(test_dc_level)),
    ]
    
    print(f"Testing different input buffer layouts:")
    print(f"Target: DC level {test_dc_level}, energy {test_energy}")
    
    best_layout = None
    best_energy = 0
    
    for name, buffer_func in layouts:
        test_buffer = buffer_func()
        buffer_energy = np.sum(test_buffer**2)
        
        # Test with MDCT
        mdct_output = mdct.mdct256(test_buffer)
        output_energy = np.sum(mdct_output**2)
        dc_coeff = mdct_output[0]
        
        print(f"\n{name}:")
        print(f"  Input energy: {buffer_energy:.6f}")
        print(f"  MDCT output energy: {output_energy:.6f}")
        print(f"  DC coefficient: {dc_coeff:.6f}")
        print(f"  Energy transfer efficiency: {output_energy/buffer_energy:.6f}")
        
        if output_energy > best_energy:
            best_energy = output_energy
            best_layout = name
    
    print(f"\nBest layout: {best_layout} with energy {best_energy:.6f}")

def create_centered_buffer(dc_level):
    """Center the data in the buffer [64:192]."""
    buf = np.zeros(256, dtype=np.float32)
    buf[64:192] = dc_level
    return buf

def create_front_buffer(dc_level):
    """Put data at the front [0:128]."""
    buf = np.zeros(256, dtype=np.float32)
    buf[0:128] = dc_level
    return buf

def create_back_buffer(dc_level):
    """Put data at the back [128:256]."""
    buf = np.zeros(256, dtype=np.float32)
    buf[128:256] = dc_level
    return buf

def create_current_buffer(dc_level):
    """Our current layout [80:208] with windowing simulation."""
    buf = np.zeros(256, dtype=np.float32)
    # Simulate our windowing: zeros [0:80], data [80:208], zeros [208:256]
    buf[80:208] = dc_level
    return buf

def create_full_buffer(dc_level):
    """Full buffer layout like atracdenc might expect."""
    buf = np.zeros(256, dtype=np.float32)
    buf[:] = dc_level
    return buf

def test_proper_windowing_layout():
    """Test the proper windowing layout that atracdenc expects."""
    
    print(f"\n=== Proper Windowing Layout Test ===")
    
    mdct = Atrac1MDCT()
    
    # What atracdenc probably expects:
    # - Previous frame overlap in first half
    # - Current frame data in second half
    # - Windowing applied at the boundary
    
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    # Proper MDCT input layout (similar to standard MDCT literature)
    proper_mdct_input = np.zeros(256, dtype=np.float32)
    
    # Option 1: Previous frame in [0:128], current frame in [128:256]
    proper_mdct_input[0:128] = 0.0  # Previous frame (zeros for first frame)
    proper_mdct_input[128:256] = test_input  # Current frame
    
    print(f"Option 1 - Standard layout [prev|curr]:")
    print(f"  [0:128]: {np.sum(proper_mdct_input[0:128]**2):.6f} energy (previous)")
    print(f"  [128:256]: {np.sum(proper_mdct_input[128:256]**2):.6f} energy (current)")
    
    result1 = mdct.mdct256(proper_mdct_input)
    print(f"  MDCT output energy: {np.sum(result1**2):.6f}")
    print(f"  DC coefficient: {result1[0]:.6f}")
    
    # Option 2: Try centered approach
    proper_mdct_input2 = np.zeros(256, dtype=np.float32)
    proper_mdct_input2[64:192] = test_input  # Centered
    
    print(f"\nOption 2 - Centered layout:")
    print(f"  [64:192]: {np.sum(proper_mdct_input2[64:192]**2):.6f} energy")
    
    result2 = mdct.mdct256(proper_mdct_input2)
    print(f"  MDCT output energy: {np.sum(result2**2):.6f}")
    print(f"  DC coefficient: {result2[0]:.6f}")
    
    # Option 3: What if we need to apply windowing first?
    print(f"\nOption 3 - Pre-windowed input:")
    windowed_input = np.zeros(256, dtype=np.float32)
    windowed_input[128:256] = test_input
    
    # Apply sine windowing at boundaries (like atracdenc would)
    # Window the transition between previous (zeros) and current (0.5)
    window = np.array([np.sin((i + 0.5) * np.pi / 64.0) for i in range(32)], dtype=np.float32)
    
    # Apply windowing at 128-32=96 to 128+32=160
    for i in range(32):
        # Crossfade from previous to current
        windowed_input[128 - 32 + i] = windowed_input[128 - 32 + i] * (1 - window[i])  # Previous frame
        windowed_input[128 + i] = windowed_input[128 + i] * window[i]  # Current frame
    
    result3 = mdct.mdct256(windowed_input)
    print(f"  MDCT output energy: {np.sum(result3**2):.6f}")
    print(f"  DC coefficient: {result3[0]:.6f}")

def compare_with_fft_based_approach():
    """Compare with a simple FFT-based approach to verify our MDCT is working."""
    
    print(f"\n=== FFT-based Verification ===")
    
    # Simple test: pure DC should give strong DC coefficient
    test_signal = np.ones(256, dtype=np.float32) * 0.5
    
    # Direct FFT (for comparison)
    fft_result = np.fft.fft(test_signal)
    print(f"Direct FFT:")
    print(f"  Input energy: {np.sum(test_signal**2):.6f}")
    print(f"  FFT output energy: {np.sum(np.abs(fft_result)**2):.6f}")
    print(f"  DC component: {fft_result[0]:.6f}")
    
    # Our MDCT
    mdct = Atrac1MDCT()
    mdct_result = mdct.mdct256(test_signal)
    print(f"\nOur MDCT:")
    print(f"  MDCT output energy: {np.sum(mdct_result**2):.6f}")
    print(f"  DC coefficient: {mdct_result[0]:.6f}")
    
    # The energy ratio should be reasonable
    energy_ratio = np.sum(mdct_result**2) / np.sum(test_signal**2)
    print(f"  Energy ratio: {energy_ratio:.6f}")
    
    if energy_ratio > 0.01:  # Reasonable energy transfer
        print(f"  ✅ MDCT is working correctly with proper input")
    else:
        print(f"  ❌ MDCT has very low energy transfer")

if __name__ == "__main__":
    test_mdct_buffer_expectations()
    test_proper_windowing_layout()
    compare_with_fft_based_approach()
    
    print(f"\n=== BUFFER LAYOUT ANALYSIS ===")
    print("The issue is likely:")
    print("1. Our tmp buffer layout doesn't match MDCT expectations")
    print("2. MDCT expects specific data placement (e.g., [0:128] and [128:256])")
    print("3. Our windowing places data in [80:208] which may not be optimal")
    print("4. Need to adjust win_start parameter or buffer preparation")