#!/usr/bin/env python3
"""
Quick cross-validation test to get started.
Run this to test PyATRAC1 against any reference ATRAC1 files you can find.
"""

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.core.decoder import Atrac1Decoder

def create_test_signal(duration_samples=512, frequency=1000, sample_rate=44100):
    """Create a test sine wave signal."""
    t = np.arange(duration_samples, dtype=np.float32) / sample_rate
    return 0.1 * np.sin(2 * np.pi * frequency * t)

def test_encode_decode_quality():
    """Test encode-decode quality with known signal."""
    print("Testing PyATRAC1 encode-decode quality...")
    
    # Create test signal
    original = create_test_signal()
    
    # Encode and decode
    encoder = Atrac1Encoder()
    decoder = Atrac1Decoder()
    
    encoded = encoder.encode_frame(original)
    decoded = decoder.decode_frame(encoded)
    
    # Calculate SNR
    noise = decoded - original
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
        print(f"SNR: {snr_db:.2f} dB")
        
        if snr_db > 20:  # Basic quality threshold
            print("✅ Basic quality test PASSED")
        else:
            print("❌ Basic quality test FAILED")
    else:
        print("⚠️  Perfect reconstruction (no noise)")
    
    return encoded, decoded

def validate_frame_format(encoded_frame):
    """Validate basic frame format."""
    print("Validating frame format...")
    
    # Check frame size
    if len(encoded_frame) == 212:
        print("✅ Frame size correct (212 bytes)")
    else:
        print(f"❌ Frame size incorrect: {len(encoded_frame)} bytes")
    
    # Check for non-zero content (basic sanity check)
    if any(b != 0 for b in encoded_frame):
        print("✅ Frame contains data")
    else:
        print("⚠️  Frame is all zeros")

if __name__ == "__main__":
    print("PyATRAC1 Quick Validation Test")
    print("=" * 40)
    
    try:
        encoded, decoded = test_encode_decode_quality()
        validate_frame_format(encoded)
        
        print("\nNext steps:")
        print("1. Find reference ATRAC1 files (.aea or .at1)")
        print("2. Test decoding with PyATRAC1")
        print("3. Compare output quality")
        print("4. Set up full cross-codec validation")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("This suggests there may be issues with the codec implementation.")