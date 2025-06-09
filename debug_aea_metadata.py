#!/usr/bin/env python3

import sys
from pyatrac1.aea.aea_reader import AeaReader

if len(sys.argv) != 2:
    print("Usage: python debug_aea_metadata.py <aea_file>")
    sys.exit(1)

aea_file = sys.argv[1]

try:
    with AeaReader(aea_file) as reader:
        print(f"AEA file: {aea_file}")
        print(f"Title: '{reader.metadata.title}'")
        print(f"Total frames: {reader.metadata.total_frames}")
        print(f"Channel count: {reader.metadata.channel_count}")
        print(f"File size: {reader._get_file_size()} bytes")
        
        # Calculate expected frame count
        expected_frames = (reader._get_file_size() - 2048) // (212 * reader.metadata.channel_count)
        print(f"Expected frames: {expected_frames}")
        
except Exception as e:
    print(f"Error reading AEA file: {e}")