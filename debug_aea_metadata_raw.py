#!/usr/bin/env python3

import struct

aea_file = "debug_frames.aea"

with open(aea_file, 'rb') as f:
    # Read AEA header
    header = f.read(2048)
    
    # Check channel count (offset 264)
    channel_count = header[264]
    print(f"Channel count (byte 264): {channel_count}")
    
    # Check for frame count in header (need to find the right offset)
    # AEA format typically stores frame count somewhere in the header
    
    # Check various potential frame count locations
    for offset in [260, 261, 262, 263, 264, 265, 266, 267, 268]:
        value = struct.unpack('<I', header[offset:offset+4])[0] if offset + 4 <= len(header) else 0
        print(f"Offset {offset}: {value} (0x{value:08x})")
    
    # Check file size calculation
    f.seek(0, 2)  # Go to end
    file_size = f.tell()
    print(f"\nFile size: {file_size} bytes")
    print(f"Header size: 2048 bytes")
    print(f"Frame data size: {file_size - 2048} bytes")
    print(f"Number of 212-byte frames: {(file_size - 2048) // 212}")
    
    # Manual calculation like atracdenc
    nChannels = header[264] if header[264] else 1
    manual_calc = (file_size - 2048) // 212 // nChannels - 5
    print(f"atracdenc calculation: ({file_size} - 2048) / 212 / {nChannels} - 5 = {manual_calc}")
    
    # The issue might be that we need MORE frames, not fewer
    print(f"Expected by atracdenc: {manual_calc + 5} frames")
    print(f"Should write dummy frames: {5} frames minimum")