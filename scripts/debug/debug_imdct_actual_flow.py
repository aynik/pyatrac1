#!/usr/bin/env python3
"""
Debug script to trace the actual IMDCT flow during decoding
and identify where the middle section becomes zeros.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyatrac1.core.mdct import *
from pyatrac1.core.decoder import Atrac1Decoder
from pyatrac1.aea.aea_reader import AeaReader
from pyatrac1.common.debug_logger import debug_logger

def debug_imdct_actual_flow():
    """
    Debug the actual IMDCT flow during decoding to find where zeros appear.
    """
    
    # Create test AEA file with PyATRAC1 encoder
    print("=== Creating Test AEA File ===")
    
    # Generate simple test audio
    sample_rate = 44100
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Simple sine wave
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Create a simple AEA file using our encoder
    from pyatrac1.core.encoder import Atrac1Encoder
    from pyatrac1.aea.aea_writer import AeaWriter
    
    encoder = Atrac1Encoder()
    
    # Encode one frame
    frame_samples = test_audio[:512]  # One frame
    if len(frame_samples) < 512:
        frame_samples = np.pad(frame_samples, (0, 512 - len(frame_samples)))
    
    print(f"Input audio range: [{np.min(frame_samples):.6f}, {np.max(frame_samples):.6f}]")
    
    # Encode the frame
    frame_bytes = encoder.encode_frame(frame_samples)
    print(f"Encoded frame size: {len(frame_bytes)} bytes")
    
    # Write to AEA file
    with open("debug_test.aea", "wb") as f:
        # Write AEA header
        aea_writer = AeaWriter(f, channel_count=1, title="Debug")
        aea_writer.write_frame(frame_bytes)
        aea_writer.close()
    
    print("=== Decoding Test AEA File ===")
    
    # Now decode it and trace the IMDCT flow
    decoder = Atrac1Decoder()
    
    # Patch the IMDCT function to add debug logging
    original_imdct = decoder.mdct.imdct
    
    def debug_imdct(specs, mode, low, mid, hi, channel=0, frame=0):
        print(f"\n--- IMDCT Debug for Channel {channel}, Frame {frame} ---")
        
        # Log input specs
        print(f"Input specs range: [{np.min(specs):.6f}, {np.max(specs):.6f}]")
        print(f"Input specs non-zero: {np.count_nonzero(specs)}/512")
        
        # Check each band's specs
        pos = 0
        for band in range(3):
            buf_sz = 256 if band == 2 else 128
            band_specs = specs[pos:pos + buf_sz]
            band_name = ["LOW", "MID", "HIGH"][band]
            print(f"{band_name} band specs [{pos}:{pos + buf_sz}]:")
            print(f"  - Range: [{np.min(band_specs):.6f}, {np.max(band_specs):.6f}]")
            print(f"  - Non-zero: {np.count_nonzero(band_specs)}/{buf_sz}")
            print(f"  - First 8: {band_specs[:8]}")
            
            # Test IMDCT on this band
            if band == 0:  # LOW band
                raw_imdct = decoder.mdct.imdct256(band_specs)
            elif band == 1:  # MID band
                raw_imdct = decoder.mdct.imdct256(band_specs)
            else:  # HIGH band
                raw_imdct = decoder.mdct.imdct512(band_specs)
            
            print(f"  - Raw IMDCT range: [{np.min(raw_imdct):.6f}, {np.max(raw_imdct):.6f}]")
            
            # Check middle section
            middle_start = len(raw_imdct) // 4
            middle_length = len(raw_imdct) // 2
            middle_section = raw_imdct[middle_start:middle_start + middle_length]
            print(f"  - Middle section range: [{np.min(middle_section):.6f}, {np.max(middle_section):.6f}]")
            print(f"  - Middle section first 8: {middle_section[:8]}")
            
            pos += buf_sz
        
        # Call original IMDCT
        result = original_imdct(specs, mode, low, mid, hi, channel, frame)
        
        # Check outputs
        print(f"Output LOW range: [{np.min(low):.6f}, {np.max(low):.6f}]")
        print(f"Output MID range: [{np.min(mid):.6f}, {np.max(mid):.6f}]")
        print(f"Output HIGH range: [{np.min(hi):.6f}, {np.max(hi):.6f}]")
        
        # Check middle sections of outputs
        print(f"LOW middle [32:144]: {low[32:40]}")
        print(f"MID middle [32:144]: {mid[32:40]}")
        print(f"HIGH middle [32:272]: {hi[32:40]}")
        
        return result
    
    # Patch the decoder
    decoder.mdct.imdct = debug_imdct
    
    # Decode the file
    try:
        with open("debug_test.aea", "rb") as f:
            aea_reader = AeaReader(f)
            decoded_audio = decoder.decode_stream(aea_reader)
            
            print(f"\n=== Final Decoded Audio ===")
            print(f"Decoded audio shape: {decoded_audio.shape}")
            print(f"Decoded audio range: [{np.min(decoded_audio):.6f}, {np.max(decoded_audio):.6f}]")
            print(f"Decoded audio first 16: {decoded_audio[:16]}")
            
    except Exception as e:
        print(f"Error during decoding: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    if os.path.exists("debug_test.aea"):
        os.remove("debug_test.aea")

if __name__ == "__main__":
    debug_imdct_actual_flow()