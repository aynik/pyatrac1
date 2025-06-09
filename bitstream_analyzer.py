#!/usr/bin/env python3
"""
Bitstream analysis tool to compare PyATRAC1 vs atracdenc frame structures.
"""

import sys
import os
sys.path.insert(0, '/Users/pablo/Projects/pytrac')

import numpy as np
from pyatrac1.aea.aea_reader import AeaReader
from pyatrac1.core.bitstream import Atrac1BitstreamReader
from pyatrac1.core.codec_data import Atrac1CodecData
import subprocess
import tempfile

def hex_dump(data, bytes_per_line=16, prefix=""):
    """Create a hex dump of binary data."""
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
        lines.append(f"{prefix}{i:04x}: {hex_part:<{bytes_per_line*3-1}} |{ascii_part}|")
    return '\n'.join(lines)

def analyze_aea_frame_structure(aea_file_path, frame_num=0):
    """Analyze the structure of a specific frame in an AEA file."""
    print(f"\n🔍 Analyzing frame {frame_num} in {os.path.basename(aea_file_path)}")
    print("=" * 80)
    
    # First, analyze the AEA file structure
    print(f"📋 AEA File Structure:")
    file_size = os.path.getsize(aea_file_path)
    print(f"  Total file size: {file_size} bytes")
    print(f"  Expected metadata size: 2048 bytes")
    print(f"  Remaining for frames: {file_size - 2048} bytes")
    print(f"  Expected frame count: {(file_size - 2048) // 212} frames")
    
    try:
        with AeaReader(aea_file_path) as reader:
            print(f"📄 AEA Metadata:")
            if reader.metadata:
                print(f"  Channel count: {reader.metadata.channel_count}")
                print(f"  Total frames: {reader.metadata.total_frames}")
                print(f"  Title: '{reader.metadata.title}'")
                total_samples = reader.metadata.total_frames * 512
                duration = total_samples / 44100  # Assuming 44.1kHz
                print(f"  Duration: {duration:.3f}s (estimated)")
            else:
                print(f"  ❌ No metadata found")
            
            frames_analyzed = 0
            for frame_data in reader.frames():
                if frames_analyzed == frame_num:
                    print(f"📦 Frame {frame_num} raw data ({len(frame_data)} bytes):")
                    print(hex_dump(frame_data, prefix="  "))
                    
                    # Try to parse with PyATRAC1 bitstream reader
                    print(f"\n🔧 PyATRAC1 Bitstream Analysis:")
                    try:
                        codec_data = Atrac1CodecData()
                        bitstream_reader = Atrac1BitstreamReader(codec_data)
                        frame_obj = bitstream_reader.read_frame(frame_data)
                        
                        print(f"  ✅ Successfully parsed frame:")
                        print(f"    BSM: low={frame_obj.bsm_low}, mid={frame_obj.bsm_mid}, high={frame_obj.bsm_high}")
                        print(f"    BFU amount idx: {frame_obj.bfu_amount_idx}")
                        print(f"    Num active BFUs: {frame_obj.num_active_bfus}")
                        print(f"    Word lengths: {frame_obj.word_lengths[:10]}..." if len(frame_obj.word_lengths) > 10 else f"    Word lengths: {frame_obj.word_lengths}")
                        print(f"    Scale factors: {frame_obj.scale_factor_indices[:10]}..." if len(frame_obj.scale_factor_indices) > 10 else f"    Scale factors: {frame_obj.scale_factor_indices}")
                        print(f"    Mantissa blocks: {len(frame_obj.quantized_mantissas)}")
                        
                        # Detailed bit analysis
                        print(f"\n📊 Detailed Bit Analysis:")
                        total_bits = len(frame_data) * 8
                        print(f"  Total bits available: {total_bits}")
                        
                        # Calculate header bits
                        header_bits = 2 + 2 + 2 + 3  # BSM + BFU amount
                        wl_bits = frame_obj.num_active_bfus * 4  # Word lengths
                        sf_bits = frame_obj.num_active_bfus * 6  # Scale factors
                        
                        print(f"  Header control bits: {header_bits}")
                        print(f"  Word length bits: {wl_bits}")
                        print(f"  Scale factor bits: {sf_bits}")
                        
                        mantissa_bits = 0
                        for i, mantissas in enumerate(frame_obj.quantized_mantissas):
                            if i < len(frame_obj.word_lengths):
                                wl = frame_obj.word_lengths[i]
                                mantissa_bits += len(mantissas) * wl
                        
                        print(f"  Mantissa bits used: {mantissa_bits}")
                        print(f"  Total accounted bits: {header_bits + wl_bits + sf_bits + mantissa_bits}")
                        print(f"  Remaining bits: {total_bits - (header_bits + wl_bits + sf_bits + mantissa_bits)}")
                        
                    except Exception as e:
                        print(f"  ❌ Failed to parse: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    return frame_data
                    
                frames_analyzed += 1
                    
    except Exception as e:
        print(f"❌ Error reading AEA file: {e}")
        return None

def compare_implementations(test_wav_path):
    """Compare PyATRAC1 and atracdenc frame structures."""
    print(f"\n🔄 Comparing implementations for {os.path.basename(test_wav_path)}")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        pyatrac1_aea = os.path.join(temp_dir, "pyatrac1.aea")
        reference_aea = os.path.join(temp_dir, "reference.aea") 
        
        # Encode with PyATRAC1
        print("🐍 Encoding with PyATRAC1...")
        cmd = [
            "python", "atrac1_cli.py", "-m", "encode", 
            "-i", test_wav_path, "-o", pyatrac1_aea
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/pablo/Projects/pytrac")
        if result.returncode != 0:
            print(f"❌ PyATRAC1 encoding failed: {result.stderr}")
            return
        
        # Encode with atracdenc
        print("🔧 Encoding with atracdenc...")
        cmd = [
            "/Users/pablo/Projects/pytrac/atracdenc/build/src/atracdenc", 
            "-e", "atrac1", "-i", test_wav_path, "-o", reference_aea
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ atracdenc encoding failed: {result.stderr}")
            return
        
        print(f"\n📈 File sizes:")
        pyatrac1_size = os.path.getsize(pyatrac1_aea)
        reference_size = os.path.getsize(reference_aea)
        print(f"  PyATRAC1: {pyatrac1_size} bytes")
        print(f"  atracdenc: {reference_size} bytes")
        print(f"  Difference: {pyatrac1_size - reference_size} bytes")
        
        # Analyze first few frames from both
        for frame_idx in range(min(3, (pyatrac1_size - 2048) // 212)):
            print(f"\n" + "="*50 + f" FRAME {frame_idx} " + "="*50)
            
            print(f"\n🐍 PyATRAC1 Frame {frame_idx}:")
            pyatrac1_frame = analyze_aea_frame_structure(pyatrac1_aea, frame_idx)
            
            print(f"\n🔧 atracdenc Frame {frame_idx}:")
            reference_frame = analyze_aea_frame_structure(reference_aea, frame_idx)
            
            # Compare frame data byte-by-byte
            if pyatrac1_frame and reference_frame:
                print(f"\n🔍 Byte-by-byte comparison:")
                if len(pyatrac1_frame) != len(reference_frame):
                    print(f"  ❌ Frame size mismatch: PyATRAC1={len(pyatrac1_frame)}, atracdenc={len(reference_frame)}")
                    continue
                
                differences = []
                for i, (p_byte, r_byte) in enumerate(zip(pyatrac1_frame, reference_frame)):
                    if p_byte != r_byte:
                        differences.append((i, p_byte, r_byte))
                
                if differences:
                    print(f"  ❌ Found {len(differences)} byte differences:")
                    for i, (offset, p_byte, r_byte) in enumerate(differences[:20]):  # Show first 20
                        print(f"    Offset {offset:3d}: PyATRAC1=0x{p_byte:02x} atracdenc=0x{r_byte:02x}")
                    if len(differences) > 20:
                        print(f"    ... and {len(differences) - 20} more differences")
                else:
                    print(f"  ✅ Frames are identical!")

def create_minimal_test_signal():
    """Create a minimal test signal for debugging."""
    print("🎵 Creating minimal test signal...")
    
    # Create a very simple signal - just a few sine wave cycles
    sample_rate = 44100
    duration = 512 / sample_rate  # Exactly one frame
    t = np.linspace(0, duration, 512, False)
    
    # Simple 1kHz sine wave with small amplitude
    signal = 0.1 * np.sin(2 * np.pi * 1000 * t)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        test_wav = f.name
    
    # Write WAV file
    import wave
    with wave.open(test_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Convert to 16-bit integers
        signal_int = (signal * 32767).astype(np.int16)
        wav_file.writeframes(signal_int.tobytes())
    
    print(f"✅ Created test signal: {test_wav}")
    return test_wav

def main():
    print("🔍 ATRAC1 Bitstream Analysis Tool")
    print("=" * 80)
    
    # Create minimal test case
    test_wav = create_minimal_test_signal()
    
    try:
        # Compare implementations
        compare_implementations(test_wav)
        
    finally:
        # Clean up
        if os.path.exists(test_wav):
            os.unlink(test_wav)

if __name__ == "__main__":
    main()