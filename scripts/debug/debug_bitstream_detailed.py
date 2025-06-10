#!/usr/bin/env python3

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder

# Create test signal
audio_data = np.random.randn(512).astype(np.float32) * 0.1

# Create encoder
encoder = Atrac1Encoder()

# Encode just the frame (no AEA wrapper)
frame_bytes = encoder.encode_frame(audio_data)

print("=== DETAILED BITSTREAM ANALYSIS ===")
print(f"Frame size: {len(frame_bytes)} bytes")

# Show first 4 bytes in detail
first_bytes = frame_bytes[:4]
print(f"First 4 bytes: {' '.join([f'{b:02x}' for b in first_bytes])}")

# Convert to binary and analyze bit by bit
all_bits = []
for byte in first_bytes:
    for i in range(7, -1, -1):  # MSB first
        all_bits.append((byte >> i) & 1)

print(f"First 32 bits: {''.join(map(str, all_bits))}")

# Parse according to our expected format
print("\n=== OUR FORMAT (what PyATRAC1 should write) ===")
bit_pos = 0
bfu_amount = (all_bits[0] << 2) | (all_bits[1] << 1) | all_bits[2]
bit_pos += 3
print(f"BFU amount (bits 0-2): {bfu_amount}")

bsm_low = (all_bits[bit_pos] << 1) | all_bits[bit_pos+1]
bit_pos += 2
print(f"BSM low (bits 3-4): {bsm_low}")

bsm_mid = (all_bits[bit_pos] << 1) | all_bits[bit_pos+1]
bit_pos += 2
print(f"BSM mid (bits 5-6): {bsm_mid}")

bsm_high = (all_bits[bit_pos] << 1) | all_bits[bit_pos+1]
bit_pos += 2
print(f"BSM high (bits 7-8): {bsm_high}")

print(f"PyATRAC1 BSM: [{bsm_low}, {bsm_mid}, {bsm_high}]")

# Parse according to what atracdenc actually reads
print("\n=== ATRACDENC FORMAT (what it actually reads) ===")
# Based on debug output, atracdenc reads [3,2,0] from this data
# Let's see what bit positions would give those values

# Try different bit alignments to see where [3,2,0] comes from
for offset in range(16):
    try:
        test_bfu = (all_bits[offset] << 2) | (all_bits[offset+1] << 1) | all_bits[offset+2]
        test_bsm_low = (all_bits[offset+3] << 1) | all_bits[offset+4]
        test_bsm_mid = (all_bits[offset+5] << 1) | all_bits[offset+6]
        test_bsm_high = (all_bits[offset+7] << 1) | all_bits[offset+8]
        
        if test_bsm_low == 3 and test_bsm_mid == 2 and test_bsm_high == 0:
            print(f"FOUND MATCH at offset {offset}:")
            print(f"  BFU amount: {test_bfu}")
            print(f"  BSM: [{test_bsm_low}, {test_bsm_mid}, {test_bsm_high}]")
            print(f"  This means atracdenc reads with {offset} bit offset!")
    except IndexError:
        break

print(f"\n=== COMPARISON ===")
print(f"Expected BSM: [0, 0, 0]")
print(f"atracdenc reads: [3, 2, 0]") 
print(f"This suggests a bit alignment or bit order issue")