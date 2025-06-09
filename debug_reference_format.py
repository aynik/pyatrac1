#!/usr/bin/env python3

import sys

def analyze_reference_aea():
    """Analyze the reference AEA file from atracdenc to understand the expected format"""
    
    with open('/tmp/reference_atracdenc.aea', 'rb') as f:
        data = f.read()
    
    print(f"Total file size: {len(data)} bytes")
    
    # Skip AEA header (2048 bytes)
    header = data[:2048]
    frames_data = data[2048:]
    
    print(f"Header size: {len(header)} bytes")
    print(f"Frames data size: {len(frames_data)} bytes")
    
    # Calculate number of frames
    frame_size = 212
    num_frames = len(frames_data) // frame_size
    print(f"Number of frames: {num_frames}")
    
    # Analyze the first actual frame (skip dummy frames)
    # atracdenc typically puts 5 dummy frames first
    if num_frames > 5:
        actual_frame_start = 5 * frame_size
        first_actual_frame = frames_data[actual_frame_start:actual_frame_start + frame_size]
        
        print(f"\n=== FIRST ACTUAL FRAME (Frame {5}) ===")
        print(f"Frame size: {len(first_actual_frame)} bytes")
        
        # Analyze bit by bit
        bits = []
        for byte in first_actual_frame:
            for bit in range(7, -1, -1):
                bits.append((byte >> bit) & 1)
        
        print(f"Total bits: {len(bits)}")
        
        # Parse according to atracdenc dequantizer expectations
        bit_pos = 0
        
        print("\n--- BSM Parsing (TBlockSizeMod::Parse reads first) ---")
        
        # TBlockSizeMod::Parse reads first 8 bits
        bsm_low_raw = (bits[0] << 1) | bits[1]
        bsm_mid_raw = (bits[2] << 1) | bits[3]
        bsm_high_raw = (bits[4] << 1) | bits[5]
        bsm_reserved = (bits[6] << 1) | bits[7]
        
        bsm_low_actual = 2 - bsm_low_raw
        bsm_mid_actual = 2 - bsm_mid_raw
        bsm_high_actual = 3 - bsm_high_raw
        
        print(f"BSM Raw: Low={bsm_low_raw}, Mid={bsm_mid_raw}, High={bsm_high_raw}, Reserved={bsm_reserved}")
        print(f"BSM Actual: Low={bsm_low_actual}, Mid={bsm_mid_actual}, High={bsm_high_actual}")
        
        bit_pos = 8  # After BSM parsing
        
        print("\n--- Dequantizer reads from position 8 onwards ---")
        
        # Now dequantizer reads BFU amount from bit position 8
        bfu_amount_idx = 0
        for i in range(3):
            bfu_amount_idx = (bfu_amount_idx << 1) | bits[bit_pos]
            bit_pos += 1
        
        print(f"BFU Amount Index (from pos 8): {bfu_amount_idx}")
        
        # Skip next 5 bits (2 + 3 reserved)
        reserved_bits = 0
        for i in range(5):
            reserved_bits = (reserved_bits << 1) | bits[bit_pos]
            bit_pos += 1
        
        print(f"Reserved bits (5): {reserved_bits:05b}")
        
        BFU_AMOUNT_TAB = [52, 47, 36, 29, 23, 18, 14, 10]
        num_active_bfus = BFU_AMOUNT_TAB[bfu_amount_idx]
        print(f"Active BFUs: {num_active_bfus}")
        
        print(f"Position after BFU control: {bit_pos} bits")
        
        # Read word lengths
        print(f"\n--- Word Lengths ({num_active_bfus} BFUs) ---")
        word_lengths = []
        for i in range(num_active_bfus):
            wl = 0
            for j in range(4):
                wl = (wl << 1) | bits[bit_pos]
                bit_pos += 1
            word_lengths.append(wl)
            if i < 10:
                print(f"WL[{i}]: {wl}")
        
        print(f"Position after word lengths: {bit_pos} bits")
        
        # Read scale factors
        print(f"\n--- Scale Factors ({num_active_bfus} BFUs) ---")
        scale_factors = []
        for i in range(num_active_bfus):
            sf = 0
            for j in range(6):
                sf = (sf << 1) | bits[bit_pos]
                bit_pos += 1
            scale_factors.append(sf)
            if i < 10:
                print(f"SF[{i}]: {sf}")
        
        print(f"Position after scale factors: {bit_pos} bits")
        
        # Show remaining bits
        remaining_bits = len(bits) - bit_pos
        print(f"Remaining mantissa bits: {remaining_bits}")
        
        # Show first few bytes of raw frame data for comparison
        print(f"\n--- Raw Frame Bytes (first 20) ---")
        for i in range(min(20, len(first_actual_frame))):
            print(f"Byte {i}: 0x{first_actual_frame[i]:02x} ({first_actual_frame[i]:08b})")
        
        # Now let's also check what the BSM values should be by analyzing position 0
        print(f"\n--- Analysis of BSM Values (TBlockSizeMod::Parse reads from pos 0) ---")
        # If TBlockSizeMod::Parse reads from position 0, let's see what's there
        bsm_bits = bits[:8]  # First 8 bits
        bsm_low = (bsm_bits[0] << 1) | bsm_bits[1]
        bsm_mid = (bsm_bits[2] << 1) | bsm_bits[3]
        bsm_high = (bsm_bits[4] << 1) | bsm_bits[5]
        bsm_reserved = (bsm_bits[6] << 1) | bsm_bits[7]
        
        print(f"If first 8 bits are BSM: Low={bsm_low}, Mid={bsm_mid}, High={bsm_high}, Reserved={bsm_reserved}")
        
        # But wait, the dequantizer reads BFU amount from position 0 too!
        # This suggests the format is NOT: BSM + BFU + data
        # Instead it might be: BFU + reserved + BSM + WL + SF + mantissas
        # OR there's some other arrangement
        
        print(f"\n--- Alternative interpretation: BFU amount at position 0 ---")
        print(f"First 3 bits as BFU amount: {bfu_amount_idx}")
        print(f"This means BSM values must be stored elsewhere or the format is different")

if __name__ == "__main__":
    analyze_reference_aea()