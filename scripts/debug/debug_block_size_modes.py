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

print("Block Size Modes written by PyATRAC1:")
print(f"Low band BSM:  {frame_data.bsm_low}")
print(f"Mid band BSM:  {frame_data.bsm_mid}")  
print(f"High band BSM: {frame_data.bsm_high}")

print("\natracdenc expectations (from source code):")
print("BSM values should be:")
print("- Low/Mid bands: 0-2 (where 0=short windows, 2=long windows)")
print("- High band: 0-3 (where 0=short windows, 3=long windows)")
print("- LogCount calculation: Low/Mid = 2-BSM, High = 3-BSM")

print(f"\nCalculated LogCounts from PyATRAC1 BSMs:")
log_count_low = 2 - frame_data.bsm_low if frame_data.bsm_low <= 2 else "INVALID"
log_count_mid = 2 - frame_data.bsm_mid if frame_data.bsm_mid <= 2 else "INVALID"  
log_count_high = 3 - frame_data.bsm_high if frame_data.bsm_high <= 3 else "INVALID"

print(f"Low band LogCount:  {log_count_low}")
print(f"Mid band LogCount:  {log_count_mid}")
print(f"High band LogCount: {log_count_high}")

print(f"\nMDCT blocks per band:")
if isinstance(log_count_low, int):
    print(f"Low band:  2^{log_count_low} = {2**log_count_low} blocks")
if isinstance(log_count_mid, int):
    print(f"Mid band:  2^{log_count_mid} = {2**log_count_mid} blocks")  
if isinstance(log_count_high, int):
    print(f"High band: 2^{log_count_high} = {2**log_count_high} blocks")