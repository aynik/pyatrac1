#!/usr/bin/env python3

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.aea.aea_writer import AeaWriter
from pyatrac1.aea.aea_reader import AeaReader
from pyatrac1.core.bitstream import Atrac1BitstreamReader
from pyatrac1.core.codec_data import Atrac1CodecData

# Create test signal (512 samples for 1 frame)
audio_data = np.random.randn(512).astype(np.float32) * 0.1

# Create encoder
encoder = Atrac1Encoder()

# Create AEA file and check what frames it contains
test_file = "debug_frames.aea"
with AeaWriter(test_file, channel_count=1, title="Debug") as writer:
    frame_bytes = encoder.encode_frame(audio_data)
    print(f"Generated frame size: {len(frame_bytes)} bytes")
    writer.write_frame(frame_bytes)

print(f"Created AEA file: {test_file}")

# Read back and analyze each frame
import struct

with open(test_file, 'rb') as f:
    # Skip header
    f.seek(2048)
    
    frame_num = 0
    codec_data = Atrac1CodecData()
    bitstream_reader = Atrac1BitstreamReader(codec_data)
    
    while True:
        frame_data = f.read(212)
        if len(frame_data) != 212:
            print(f"End of frames - read {len(frame_data)} bytes")
            break
            
        frame_num += 1
        print(f"\n=== FRAME {frame_num} ANALYSIS ===")
        print(f"Frame size: {len(frame_data)} bytes")
        
        try:
            frame_obj = bitstream_reader.read_frame(frame_data)
            print(f"BSM values: [{frame_obj.bsm_low}, {frame_obj.bsm_mid}, {frame_obj.bsm_high}]")
            print(f"BFU amount idx: {frame_obj.bfu_amount_idx}")
            print(f"Number of BFUs: {frame_obj.num_active_bfus}")
            
            # Check first few bytes as hex
            hex_bytes = ' '.join([f'{b:02x}' for b in frame_data[:8]])
            print(f"First 8 bytes (hex): {hex_bytes}")
            
        except Exception as e:
            print(f"Error parsing frame: {e}")
            # Show raw bytes for debugging
            hex_bytes = ' '.join([f'{b:02x}' for b in frame_data[:16]])
            print(f"First 16 bytes (hex): {hex_bytes}")

print(f"\nTotal frames found: {frame_num}")