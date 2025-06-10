#!/usr/bin/env python3

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.aea.aea_writer import AeaWriter
from pyatrac1.aea.aea_reader import AeaReader

# Read the same test signal that atracdenc uses
import soundfile as sf
audio_data, sr = sf.read('test_input.wav')
# Take first 512 samples for 1 frame
audio_data = audio_data[:512].astype(np.float32)

# Create encoder
encoder = Atrac1Encoder()

# Create test AEA file
test_file = "test_metadata.aea"
with AeaWriter(test_file, channel_count=1, title="TestTitle") as writer:
    # Write one frame
    for n in range(16):
        frame_bytes = encoder.encode_frame(audio_data)
        writer.write_frame(frame_bytes)

print(f"Created AEA file: {test_file}")

# Read back the metadata
with AeaReader(test_file) as reader:
    print(f"Title: '{reader.metadata.title}'")
    print(f"Total frames: {reader.metadata.total_frames}")
    print(f"Channel count: {reader.metadata.channel_count}")
    
    # Calculate file size info
    file_size = reader._get_file_size()
    expected_frames = (file_size - 2048) // (212 * reader.metadata.channel_count)
    print(f"File size: {file_size} bytes")
    print(f"Expected frames: {expected_frames}")
    print(f"Metadata frames match: {reader.metadata.total_frames == expected_frames}")
