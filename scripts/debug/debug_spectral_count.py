#!/usr/bin/env python3

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.core.bitstream import Atrac1BitstreamReader
from pyatrac1.core.codec_data import Atrac1CodecData

# Create test signal (512 samples for 1 frame)
audio_data = np.random.randn(512).astype(np.float32) * 0.1

# Create encoder
encoder = Atrac1Encoder()

# Encode frame
frame_bytes = encoder.encode_frame(audio_data)

# Parse the frame back to see what was actually written
codec_data = Atrac1CodecData()
bitstream_reader = Atrac1BitstreamReader(codec_data)
frame_data = bitstream_reader.read_frame(frame_bytes)

print("Spectral coefficient analysis:")
print(f"Number of BFUs: {frame_data.num_active_bfus}")
print(f"BFU amount index: {frame_data.bfu_amount_idx}")

total_specs = 0
for i, mantissas in enumerate(frame_data.quantized_mantissas):
    spec_count = len(mantissas)
    total_specs += spec_count
    word_len = frame_data.word_lengths[i] if i < len(frame_data.word_lengths) else 0
    print(f"BFU {i:2d}: {spec_count:3d} coefficients, word_length={word_len}")

print(f"\nTotal spectral coefficients: {total_specs}")
print(f"Expected for ATRAC1: 512 coefficients")
print(f"Match: {total_specs == 512}")

print(f"\natracdenc MDCT expects exactly 512 float coefficients")
print(f"Buffer overflow occurs when trying to access beyond coefficient 512")

# Check if any BFUs have unusual coefficient counts
expected_bfu_sizes = codec_data.bfu_amount_tab  # This should have the BFU sizes
print(f"\nBFU amount table: {expected_bfu_sizes}")

# Calculate expected total based on BFU distribution
if hasattr(codec_data, 'bfu_table') or hasattr(codec_data, 'bfu_specs'):
    print("Need to check BFU size distribution...")