#!/usr/bin/env python3
"""
Test script to verify atracdenc debug logging functionality.
Creates a simple test WAV file and encodes it with both PyATRAC1 and atracdenc.
"""

import numpy as np
import wave
import os
import subprocess
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.common.debug_logger import enable_debug_logging
from pyatrac1.common.constants import NUM_SAMPLES

def create_test_wav(filename, duration_frames=2):
    """Create a simple test WAV file with a sine wave."""
    sample_rate = 44100
    frequency = 440  # A4 note
    
    # Generate enough samples for the specified number of frames
    total_samples = NUM_SAMPLES * duration_frames
    t = np.linspace(0, total_samples / sample_rate, total_samples, endpoint=False)
    
    # Create a sine wave with slight amplitude modulation for more interesting signal
    amplitude = 0.5 * (1.0 + 0.1 * np.sin(2 * np.pi * 10 * t))  # 10 Hz modulation
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    signal_16bit = (signal * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_16bit.tobytes())
    
    print(f"Created test WAV: {filename} ({duration_frames} frames, {len(signal_16bit)} samples)")

def main():
    test_wav = "test_input.wav"
    pytrac_aea = "test_pytrac.aea"
    atracdenc_aea = "test_atracdenc.aea"
    
    # Clean up any existing files
    for f in [test_wav, pytrac_aea, atracdenc_aea, "pytrac_debug.log", "atracdenc_debug.log"]:
        if os.path.exists(f):
            os.remove(f)
    
    # Create test WAV file (2 frames for comparison)
    create_test_wav(test_wav, duration_frames=2)
    
    print("\n=== Testing PyATRAC1 Encoding with Logging ===")
    
    # Test PyATRAC1 encoding with debug logging
    enable_debug_logging("pytrac_debug.log")
    
    # Read WAV file and encode with PyATRAC1
    with wave.open(test_wav, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    encoder = Atrac1Encoder()
    
    # Encode frame by frame to test logging
    frame_size = NUM_SAMPLES
    encoded_frames = []
    
    for i in range(0, len(audio_data), frame_size):
        frame_data = audio_data[i:i+frame_size]
        if len(frame_data) == frame_size:
            encoded_frame = encoder.encode_frame(frame_data)
            encoded_frames.append(encoded_frame)
            print(f"PyATRAC1: Encoded frame {i//frame_size}, size: {len(encoded_frame)} bytes")
    
    # Write encoded data (simplified, without proper AEA header)
    with open(pytrac_aea, 'wb') as f:
        for frame in encoded_frames:
            f.write(frame)
    
    print(f"PyATRAC1 encoding completed. Log: pytrac_debug.log")
    
    print("\n=== Testing atracdenc Encoding with Logging ===")
    
    # Test atracdenc encoding with debug logging (output to stderr, redirect to file)
    try:
        cmd = [
            "/Users/pablo/Projects/pytrac/atracdenc/build/src/atracdenc",
            "-e", "atrac1", "-i", test_wav, "-o", atracdenc_aea
        ]
        
        # Run atracdenc and capture both stdout and stderr
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Save atracdenc debug output
        with open("atracdenc_debug.log", "w") as f:
            f.write("# atracdenc Debug Log\\n")
            f.write(f"# Command: {' '.join(cmd)}\\n")
            f.write("# STDERR:\\n")
            f.write(result.stderr)
            f.write("\\n# STDOUT:\\n")
            f.write(result.stdout)
        
        if result.returncode == 0:
            print(f"atracdenc encoding completed successfully. Log: atracdenc_debug.log")
            print(f"Output file: {atracdenc_aea} ({os.path.getsize(atracdenc_aea)} bytes)")
        else:
            print(f"atracdenc encoding failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error running atracdenc: {e}")
    
    print("\\n=== Log Analysis ===")
    
    # Check log files
    for log_file in ["pytrac_debug.log", "atracdenc_debug.log"]:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
            print(f"{log_file}: {len(lines)} lines")
            
            # Show first few log entries
            if log_file == "pytrac_debug.log":
                pcm_entries = [l for l in lines if "PCM_INPUT" in l]
                qmf_entries = [l for l in lines if "QMF_OUTPUT" in l] 
                mdct_entries = [l for l in lines if "MDCT_OUTPUT" in l]
                print(f"  - PCM_INPUT entries: {len(pcm_entries)}")
                print(f"  - QMF_OUTPUT entries: {len(qmf_entries)}")
                print(f"  - MDCT_OUTPUT entries: {len(mdct_entries)}")
            elif log_file == "atracdenc_debug.log":
                pcm_entries = [l for l in lines if "PCM_INPUT" in l]
                qmf_entries = [l for l in lines if "QMF_OUTPUT" in l]
                mdct_entries = [l for l in lines if "MDCT_OUTPUT" in l] 
                print(f"  - PCM_INPUT entries: {len(pcm_entries)}")
                print(f"  - QMF_OUTPUT entries: {len(qmf_entries)}")
                print(f"  - MDCT_OUTPUT entries: {len(mdct_entries)}")
        else:
            print(f"{log_file}: Not found")
    
    print("\\nTest completed! Check log files for detailed signal processing data.")
    print("\\nNext steps:")
    print("  1. Compare logs: grep 'QMF_OUTPUT.*CH0.*FR000.*LOW' *.log")
    print("  2. Analyze divergence: python debug_log_analyzer.py")

if __name__ == "__main__":
    main()