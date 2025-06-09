#!/usr/bin/env python3

import numpy as np
import tempfile
import os

# Import PyATRAC1 modules
from pyatrac1.core import encoder
from pyatrac1.core.bitstream import TBitStream

def create_simple_test_signal():
    """Create simple test signal for debugging"""
    sample_rate = 44100
    duration = 0.25  # Just enough for a few frames
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = np.sin(2 * np.pi * 1000 * t) * 0.8  # 1kHz sine wave
    return signal.astype(np.float32)

def analyze_frame_bits():
    """Analyze exact bit layout of PyATRAC1 frame"""
    # Create test signal and encode one frame
    test_signal = create_simple_test_signal()
    
    # Create encoder
    test_encoder = encoder.Atrac1Encoder()
    
    # Encode just one frame (512 samples)
    frame_samples = test_signal[:512]
    if len(frame_samples) < 512:
        frame_samples = np.pad(frame_samples, (0, 512 - len(frame_samples)))
    
    encoded_frame = test_encoder.encode_frame(frame_samples)
    
    print(f"Encoded frame size: {len(encoded_frame)} bytes (expected: 212)")
    
    # Parse the frame bit by bit
    reader = TBitStream(encoded_frame)
    
    print("\n=== BITSTREAM ANALYSIS ===")
    print(f"Total bits available: {len(encoded_frame) * 8}")
    
    # Read control bits
    print("\n--- Control Data ---")
    bsm_low = reader.read_bits(2)
    bsm_mid = reader.read_bits(2) 
    bsm_high = reader.read_bits(2)
    reserved1 = reader.read_bits(2)
    
    print(f"BSM Low: {bsm_low}")
    print(f"BSM Mid: {bsm_mid}")
    print(f"BSM High: {bsm_high}")
    print(f"Reserved 1: {reserved1}")
    
    bfu_amount_idx = reader.read_bits(3)
    reserved2 = reader.read_bits(5)  # Should be 5 bits, not 3
    
    print(f"BFU Amount Index: {bfu_amount_idx}")
    print(f"Reserved 2: {reserved2}")
    
    # Calculate active BFUs 
    BFU_AMOUNT_TAB = [52, 47, 36, 29, 23, 18, 14, 10]
    num_active_bfus = BFU_AMOUNT_TAB[bfu_amount_idx]
    print(f"Active BFUs: {num_active_bfus}")
    
    current_pos = reader.byte_position * 8 + reader.bit_position
    print(f"Position after control data: {current_pos} bits")
    
    # Read word lengths
    print("\n--- Word Lengths ---")
    word_lengths = []
    for i in range(num_active_bfus):
        wl = reader.read_bits(4)
        word_lengths.append(wl)
        if i < 10:  # Show first 10
            print(f"WL[{i}]: {wl}")
    
    current_pos = reader.byte_position * 8 + reader.bit_position
    print(f"Position after word lengths: {current_pos} bits")
    
    # Read scale factors
    print("\n--- Scale Factors ---")
    scale_factors = []
    for i in range(num_active_bfus):
        sf = reader.read_bits(6)
        scale_factors.append(sf)
        if i < 10:  # Show first 10
            print(f"SF[{i}]: {sf}")
    
    current_pos = reader.byte_position * 8 + reader.bit_position
    print(f"Position after scale factors: {current_pos} bits")
    
    # Calculate mantissa bits 
    header_bits = 8 + 3 + 5  # BSM (8) + BFU amount (3) + reserved (5) = 16 bits total
    wl_bits = num_active_bfus * 4
    sf_bits = num_active_bfus * 6
    total_header_bits = header_bits + wl_bits + sf_bits
    
    print(f"\n--- Bit Budget ---")
    print(f"Header bits: {header_bits}")
    print(f"Word length bits: {wl_bits}")  
    print(f"Scale factor bits: {sf_bits}")
    print(f"Total header bits: {total_header_bits}")
    
    total_frame_bits = 212 * 8
    mantissa_bits_available = total_frame_bits - total_header_bits
    print(f"Total frame bits: {total_frame_bits}")
    print(f"Mantissa bits available: {mantissa_bits_available}")
    
    # Calculate expected mantissa bits
    expected_mantissa_bits = 0
    for i, wl in enumerate(word_lengths):
        if wl > 0:
            # Get number of spectral coefficients for this BFU
            # This is approximate - actual mapping depends on block size modes
            if i < 8:  # Low band
                num_specs = 8
            elif i < 16:  # Mid band  
                num_specs = 8
            else:  # High band
                num_specs = 16 if i < 32 else 8
            
            expected_mantissa_bits += wl * num_specs
    
    print(f"Expected mantissa bits (approx): {expected_mantissa_bits}")
    
    # Try to read remaining bits
    remaining_bits = total_frame_bits - current_pos
    print(f"Remaining bits to read: {remaining_bits}")
    
    if remaining_bits > 0:
        print(f"\n--- Reading remaining {remaining_bits} bits ---")
        # Read in chunks of 8 bits for readability
        for i in range(min(10, remaining_bits // 8)):  # Show first 10 bytes
            try:
                byte_val = reader.read_bits(8)
                print(f"Byte {i}: {byte_val:02x} ({byte_val:08b})")
            except:
                print(f"Failed to read byte {i}")
                break

if __name__ == "__main__":
    analyze_frame_bits()