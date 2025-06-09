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
print(f"Frame size: {len(frame_bytes)} bytes")

# Parse the frame back to see what was actually written
codec_data = Atrac1CodecData()
bitstream_reader = Atrac1BitstreamReader(codec_data)
frame_data = bitstream_reader.read_frame(frame_bytes)

print(f"Encoder set: chosen_bfu_amount_idx = 7 (should be 52 BFUs)")
print(f"Bitstream contains: bfu_amount_idx = {frame_data.bfu_amount_idx}")
print(f"Calculated BFUs: {frame_data.num_active_bfus}")
print(f"BFU amount table: {codec_data.bfu_amount_tab}")

# Check if the index matches
expected_bfus = codec_data.bfu_amount_tab[7]  # What encoder intended
actual_bfus = frame_data.num_active_bfus      # What was written/read
print(f"Expected BFUs (index 7): {expected_bfus}")
print(f"Actual BFUs from bitstream: {actual_bfus}")
print(f"Match: {expected_bfus == actual_bfus}")