#!/usr/bin/env python3

import numpy as np
import tempfile
import os
from pyatrac1.core.encoder import Atrac1Encoder

# Create a test signal (exactly 512 samples for one frame)
samples = 512
audio_data = np.random.randn(samples).astype(np.float32) * 0.1

# Create encoder
encoder = Atrac1Encoder()

# Encode one frame
frame_bytes = encoder.encode_frame(audio_data)

# Parse the frame to get frame_data
from pyatrac1.core.bitstream import Atrac1BitstreamReader
from pyatrac1.core.codec_data import Atrac1CodecData

codec_data = Atrac1CodecData()
bitstream_reader = Atrac1BitstreamReader(codec_data)
frame_data = bitstream_reader.read_frame(frame_bytes)

# Calculate bit usage
header_bits = 2 + 2 + 2 + 2 + 3 + 2 + 3  # 16 bits total
num_bfus = frame_data.num_active_bfus
wl_bits = num_bfus * 4
sf_bits = num_bfus * 6

# Count mantissa bits
mantissa_bits = 0
for i, word_length in enumerate(frame_data.word_lengths):
    if word_length > 0:
        bfu_size = len(frame_data.quantized_mantissas[i])
        mantissa_bits += word_length * bfu_size

total_bits = header_bits + wl_bits + sf_bits + mantissa_bits
total_bytes = (total_bits + 7) // 8

print(f"Header bits: {header_bits}")
print(f"Number of BFUs: {num_bfus}")  
print(f"Word length bits: {wl_bits}")
print(f"Scale factor bits: {sf_bits}")
print(f"Mantissa bits: {mantissa_bits}")
print(f"Total bits: {total_bits}")
print(f"Total bytes: {total_bytes}")
print(f"Target: 212 bytes = 1696 bits")
print(f"Overflow: {total_bits - 1696} bits = {total_bytes - 212} bytes")