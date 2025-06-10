#!/usr/bin/env python3

"""
Debug script to understand the initialization state of atracdenc PCM buffers
and how they generate the -0.372549 values in Frame 0.
"""

import numpy as np
import os
import subprocess
import tempfile

def run_atracdenc_with_debug():
    """Run atracdenc with detailed debug logging to capture initialization state."""
    
    # Generate a very simple test signal - single sine wave
    sample_rate = 44100
    duration = 0.1  # Very short to focus on first frame
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 1000
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    
    # Write WAV file manually (simple format)
    import struct
    import wave
    
    with wave.open(tmp_wav_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Convert to 16-bit integers
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    
    # Run atracdenc with debug logging
    with tempfile.NamedTemporaryFile(suffix='.aea', delete=False) as tmp_aea:
        tmp_aea_path = tmp_aea.name
    
    try:
        # Build atracdenc if needed
        build_dir = "/Users/pablo/Projects/pytrac/atracdenc/build"
        if not os.path.exists(os.path.join(build_dir, "src/atracdenc")):
            print("Building atracdenc...")
            subprocess.run(["make", "-j4"], cwd=build_dir, check=True)
        
        # Run encoding with debug output
        cmd = [
            os.path.join(build_dir, "src/atracdenc"),
            "-e", "atrac1", "-i", tmp_wav_path, "-o", tmp_aea_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=build_dir)
        
        print("STDOUT:")
        print(result.stdout)
        
        print("STDERR:")
        print(result.stderr)
        
        if result.returncode != 0:
            print(f"atracdenc failed with return code {result.returncode}")
            return None
            
        return tmp_aea_path
    
    finally:
        # Clean up temporary WAV file
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)

def analyze_aea_structure(aea_path):
    """Analyze the structure of the generated AEA file."""
    if not aea_path or not os.path.exists(aea_path):
        print("No AEA file to analyze")
        return
    
    file_size = os.path.getsize(aea_path)
    print(f"AEA file size: {file_size} bytes")
    
    # AEA structure: 2048-byte header + frames
    header_size = 2048
    frame_size = 212
    
    if file_size < header_size:
        print("File too small to contain AEA header")
        return
    
    data_size = file_size - header_size
    num_frames = data_size // frame_size
    
    print(f"Header size: {header_size}")
    print(f"Data size: {data_size}")
    print(f"Frame size: {frame_size}")
    print(f"Number of frames: {num_frames}")
    
    # Read and analyze header
    with open(aea_path, 'rb') as f:
        header = f.read(header_size)
        
        # AEA header format
        magic = header[:4]
        title = header[4:20].decode('utf-8', errors='ignore').rstrip('\x00')
        num_frames_header = int.from_bytes(header[260:264], 'little')
        channels = header[264]
        
        print(f"Magic: {magic.hex()}")
        print(f"Title: '{title}'")
        print(f"Frames in header: {num_frames_header}")
        print(f"Channels: {channels}")
        
        # Read first few frames to see if they're dummy frames
        for i in range(min(5, num_frames)):
            frame_data = f.read(frame_size)
            if len(frame_data) < frame_size:
                break
                
            # Check if frame is all zeros (dummy frame)
            is_zero = all(b == 0 for b in frame_data)
            non_zero_count = sum(1 for b in frame_data if b != 0)
            
            print(f"Frame {i}: {'zero' if is_zero else f'{non_zero_count} non-zero bytes'}")
            
            if i == 0 and not is_zero:
                # Show first few bytes of first frame
                print(f"  First 16 bytes: {frame_data[:16].hex()}")

def test_imdct_initialization():
    """Test what happens when IMDCT is called with zero spectral coefficients."""
    print("\n=== Testing IMDCT Initialization ===")
    
    # This simulates what might happen when atracdenc processes dummy frames
    from scipy.fft import idct
    
    # Test IMDCT with zero coefficients (what dummy frames would produce)
    coeffs = np.zeros(256)  # Mid/Low band size
    imdct_result = idct(coeffs, type=4, norm='ortho')
    
    print(f"IMDCT of zero coeffs: all zeros = {np.allclose(imdct_result, 0)}")
    
    # But the interesting part is the windowing and overlap-add
    # Let's simulate the atracdenc windowing process
    
    # Sine window as calculated by atracdenc
    sine_window = np.array([
        np.sin((i + 0.5) * (np.pi / (2.0 * 32.0))) for i in range(32)
    ])
    
    print(f"Sine window range: [{sine_window[0]:.6f}, {sine_window[-1]:.6f}]")
    
    # The key insight: -0.372549 must come from somewhere in the overlap-add process
    # Let's see if it matches any specific computation
    
    target = -0.372549
    print(f"Target value: {target}")
    
    # Check various combinations
    print("Testing potential sources:")
    
    # Maybe it's from a previous IMDCT result that got windowed
    test_val = 1 / np.sqrt(2)
    windowed = -test_val * sine_window[15]  # Mid-range window value
    print(f"-1/sqrt(2) * SineWindow[15] = {windowed:.6f}")
    
    windowed = -test_val * sine_window[20]
    print(f"-1/sqrt(2) * SineWindow[20] = {windowed:.6f}")
    
    # Maybe it's a normalized DCT coefficient
    test_val = np.sqrt(2/256)  # Normalization factor
    windowed = -test_val * 0.5  
    print(f"-sqrt(2/256) * 0.5 = {windowed:.6f}")
    
    # Try to match the exact value
    closest_match = None
    closest_diff = float('inf')
    
    for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        for window_idx in range(len(sine_window)):
            test_val = -scale * sine_window[window_idx]
            diff = abs(test_val - target)
            if diff < closest_diff:
                closest_diff = diff
                closest_match = (scale, window_idx, test_val)
    
    if closest_match:
        scale, idx, val = closest_match
        print(f"Closest match: -{scale} * SineWindow[{idx}] = {val:.6f} (diff: {closest_diff:.6f})")

if __name__ == "__main__":
    print("=== Debugging atracdenc Initialization State ===")
    
    # Run atracdenc and analyze output
    aea_path = run_atracdenc_with_debug()
    if aea_path:
        analyze_aea_structure(aea_path)
        # Clean up
        os.unlink(aea_path)
    
    # Test theoretical IMDCT initialization
    test_imdct_initialization()