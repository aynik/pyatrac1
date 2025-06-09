#!/usr/bin/env python3
"""
Test script to verify PyATRAC1 debug logging functionality.
Creates a simple sine wave and encodes one frame to test logging output.
"""

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.common.debug_logger import enable_debug_logging
from pyatrac1.common.constants import NUM_SAMPLES

def main():
    # Enable debug logging
    enable_debug_logging("test_pytrac_debug.log")
    print("Debug logging enabled. Output will be written to: test_pytrac_debug.log")
    
    # Create a simple test signal (sine wave)
    sample_rate = 44100
    duration = NUM_SAMPLES / sample_rate  # Duration for one frame
    frequency = 440  # 440 Hz sine wave (A4 note)
    
    t = np.linspace(0, duration, NUM_SAMPLES, endpoint=False)
    test_signal = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"Generated test signal: {NUM_SAMPLES} samples, {frequency}Hz sine wave")
    print(f"Signal range: [{np.min(test_signal):.6f}, {np.max(test_signal):.6f}]")
    
    # Create encoder and encode one frame
    encoder = Atrac1Encoder()
    
    print("Encoding frame 0...")
    encoded_frame = encoder.encode_frame(test_signal)
    
    print(f"Encoded frame size: {len(encoded_frame)} bytes")
    print(f"Expected frame size: 212 bytes")
    
    # Encode a second frame to test frame counter
    print("Encoding frame 1...")
    encoded_frame2 = encoder.encode_frame(test_signal * 0.8)  # Slightly different amplitude
    
    print(f"Second encoded frame size: {len(encoded_frame2)} bytes")
    
    print("\nLogging test completed!")
    print("Check test_pytrac_debug.log for detailed processing logs.")
    print("\nExample log analysis commands:")
    print("  grep 'QMF_OUTPUT.*CH0.*FR000.*BAND_LOW' test_pytrac_debug.log")
    print("  grep 'MDCT_OUTPUT.*CH0.*FR001' test_pytrac_debug.log")
    print("  grep 'BITSTREAM_OUTPUT' test_pytrac_debug.log")

if __name__ == "__main__":
    main()