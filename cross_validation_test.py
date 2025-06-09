#!/usr/bin/env python3
"""
Cross-codec validation test for PyATRAC1.
Tests PyATRAC1 against the reference atracdenc implementation.
"""

import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
import sys
import shutil

# Add PyATRAC1 to path
sys.path.insert(0, '/Users/pablo/Projects/pytrac')

from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.core.decoder import Atrac1Decoder
from pyatrac1.aea.aea_writer import AeaWriter
from pyatrac1.aea.aea_reader import AeaReader

class CrossValidationTester:
    def __init__(self):
        self.atracdenc_path = "/Users/pablo/Projects/pytrac/atracdenc/build/src/atracdenc"
        self.test_files = [
            "/Volumes/Work/Music/B2 - Shkoon (2019)/01 - B2 - Shkoon.flac",
            "/Volumes/Work/Music/The Hissing of Summer Lawns - Joni Mitchell (1987)/01 - In France They Kiss on Main Street - Joni Mitchell.flac",
        ]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="atrac1_cross_validation_"))
        print(f"Working in temporary directory: {self.temp_dir}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_test_wav(self, duration_seconds=5, sample_rate=44100):
        """Create a test WAV file with known content."""
        wav_path = self.temp_dir / "test_signal.wav"
        
        # Create a test signal: 1kHz sine wave
        t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), False)
        signal = 0.1 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        
        # Write as WAV
        sf.write(wav_path, signal, sample_rate)
        print(f"Created test WAV: {wav_path}")
        return wav_path

    def convert_flac_to_wav(self, flac_path, max_duration=10):
        """Convert FLAC to WAV for testing (truncate to avoid large files)."""
        wav_path = self.temp_dir / f"{Path(flac_path).stem}_truncated.wav"
        
        try:
            # Read FLAC file
            data, samplerate = sf.read(flac_path)
            
            # Truncate to max_duration seconds
            max_samples = int(max_duration * samplerate)
            if len(data) > max_samples:
                data = data[:max_samples]
            
            # Convert to mono if stereo (for simplicity)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Ensure 44.1kHz sample rate
            if samplerate != 44100:
                print(f"Warning: Converting from {samplerate}Hz to 44100Hz")
                # Simple resampling (not ideal but sufficient for testing)
                new_length = int(len(data) * 44100 / samplerate)
                data = np.interp(np.linspace(0, len(data), new_length), np.arange(len(data)), data)
                samplerate = 44100
            
            # Write as WAV
            sf.write(wav_path, data.astype(np.float32), samplerate)
            print(f"Converted FLAC to WAV: {wav_path}")
            return wav_path
            
        except Exception as e:
            print(f"Failed to convert {flac_path}: {e}")
            return None

    def encode_with_atracdenc(self, wav_path):
        """Encode WAV to ATRAC1 using reference atracdenc."""
        aea_path = self.temp_dir / f"{wav_path.stem}_reference.aea"
        
        cmd = [
            self.atracdenc_path,
            "-e", "atrac1",
            "-i", str(wav_path),
            "-o", str(aea_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"Reference encoding successful: {aea_path}")
                return aea_path
            else:
                print(f"Reference encoding failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("Reference encoding timed out")
            return None
        except Exception as e:
            print(f"Reference encoding error: {e}")
            return None

    def decode_with_atracdenc(self, aea_path):
        """Decode ATRAC1 to WAV using reference atracdenc."""
        wav_path = self.temp_dir / f"{aea_path.stem}_decoded_reference.wav"
        
        cmd = [
            self.atracdenc_path,
            "-d",
            "-i", str(aea_path),
            "-o", str(wav_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"Reference decoding successful: {wav_path}")
                return wav_path
            else:
                print(f"Reference decoding failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("Reference decoding timed out")
            return None
        except Exception as e:
            print(f"Reference decoding error: {e}")
            return None

    def encode_with_pyatrac1(self, wav_path):
        """Encode WAV to ATRAC1 using PyATRAC1."""
        aea_path = self.temp_dir / f"{wav_path.stem}_pyatrac1.aea"
        
        try:
            # Read WAV file
            data, samplerate = sf.read(wav_path)
            if samplerate != 44100:
                print(f"Warning: Expected 44100Hz, got {samplerate}Hz")
            
            # Ensure mono and correct format
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)
            
            # Encode using PyATRAC1
            encoder = Atrac1Encoder()
            
            # Process in 512-sample frames
            frame_size = 512
            num_frames = len(data) // frame_size
            
            if num_frames == 0:
                print("Audio too short for encoding")
                return None
            
            with AeaWriter(str(aea_path), channel_count=1, title=wav_path.stem[:15]) as writer:
                for i in range(num_frames):
                    frame_start = i * frame_size
                    frame_end = frame_start + frame_size
                    frame_data = data[frame_start:frame_end]
                    
                    encoded_frame = encoder.encode_frame(frame_data)
                    writer.write_frame(encoded_frame)
            
            print(f"PyATRAC1 encoding successful: {aea_path}")
            return aea_path
            
        except Exception as e:
            print(f"PyATRAC1 encoding failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def decode_with_pyatrac1(self, aea_path):
        """Decode ATRAC1 to WAV using PyATRAC1."""
        wav_path = self.temp_dir / f"{aea_path.stem}_decoded_pyatrac1.wav"
        
        try:
            decoder = Atrac1Decoder()
            decoded_frames = []
            
            with AeaReader(str(aea_path)) as reader:
                metadata = reader.get_metadata()
                print(f"Decoding {metadata.total_frames} frames")
                
                for frame_bytes in reader.frames():
                    decoded_frame = decoder.decode_frame(frame_bytes)
                    decoded_frames.append(decoded_frame)
            
            if decoded_frames:
                # Concatenate all frames
                decoded_audio = np.concatenate(decoded_frames)
                
                # Write as WAV
                sf.write(wav_path, decoded_audio, 44100)
                print(f"PyATRAC1 decoding successful: {wav_path}")
                return wav_path
            else:
                print("No frames decoded")
                return None
                
        except Exception as e:
            print(f"PyATRAC1 decoding failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_audio_quality(self, original_path, decoded_path):
        """Calculate audio quality metrics."""
        try:
            # Read both files
            original, sr1 = sf.read(original_path)
            decoded, sr2 = sf.read(decoded_path)
            
            # Ensure same sample rate
            if sr1 != sr2:
                print(f"Sample rate mismatch: {sr1} vs {sr2}")
                return None
            
            # Ensure mono
            if len(original.shape) > 1:
                original = np.mean(original, axis=1)
            if len(decoded.shape) > 1:
                decoded = np.mean(decoded, axis=1)
            
            # Truncate to same length
            min_len = min(len(original), len(decoded))
            original = original[:min_len]
            decoded = decoded[:min_len]
            
            # Calculate SNR
            noise = decoded - original
            signal_power = np.mean(original ** 2)
            noise_power = np.mean(noise ** 2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = float('inf')
            
            # Calculate RMS error
            rms_error = np.sqrt(noise_power)
            
            return {
                'snr_db': snr_db,
                'rms_error': rms_error,
                'signal_power': signal_power,
                'noise_power': noise_power
            }
            
        except Exception as e:
            print(f"Quality calculation failed: {e}")
            return None

    def test_encode_decode_round_trip(self, wav_path):
        """Test encode-decode round trip with PyATRAC1."""
        print(f"\n🔄 Testing PyATRAC1 encode-decode round trip with {wav_path.name}")
        
        # Encode with PyATRAC1
        aea_path = self.encode_with_pyatrac1(wav_path)
        if not aea_path:
            return False
        
        # Decode with PyATRAC1
        decoded_path = self.decode_with_pyatrac1(aea_path)
        if not decoded_path:
            return False
        
        # Calculate quality
        quality = self.calculate_audio_quality(wav_path, decoded_path)
        if quality:
            print(f"✅ Round trip SNR: {quality['snr_db']:.2f} dB")
            print(f"   RMS Error: {quality['rms_error']:.6f}")
            return quality['snr_db'] > 20  # Basic quality threshold
        
        return False

    def test_cross_compatibility(self, wav_path):
        """Test cross-compatibility between PyATRAC1 and reference."""
        print(f"\n🔀 Testing cross-compatibility with {wav_path.name}")
        
        # Test 1: Reference encode -> PyATRAC1 decode
        print("  Testing: Reference encoder -> PyATRAC1 decoder")
        ref_aea = self.encode_with_atracdenc(wav_path)
        if ref_aea:
            py_decoded = self.decode_with_pyatrac1(ref_aea)
            if py_decoded:
                quality = self.calculate_audio_quality(wav_path, py_decoded)
                if quality:
                    print(f"    ✅ SNR: {quality['snr_db']:.2f} dB")
                else:
                    print("    ❌ Quality calculation failed")
            else:
                print("    ❌ PyATRAC1 decoding failed")
        else:
            print("    ❌ Reference encoding failed")
        
        # Test 2: PyATRAC1 encode -> Reference decode
        print("  Testing: PyATRAC1 encoder -> Reference decoder")
        py_aea = self.encode_with_pyatrac1(wav_path)
        if py_aea:
            ref_decoded = self.decode_with_atracdenc(py_aea)
            if ref_decoded:
                quality = self.calculate_audio_quality(wav_path, ref_decoded)
                if quality:
                    print(f"    ✅ SNR: {quality['snr_db']:.2f} dB")
                else:
                    print("    ❌ Quality calculation failed")
            else:
                print("    ❌ Reference decoding failed")
        else:
            print("    ❌ PyATRAC1 encoding failed")

    def run_validation(self):
        """Run the complete cross-validation test suite."""
        print("🎵 PyATRAC1 Cross-Codec Validation Test")
        print("=" * 50)
        
        try:
            # Test 1: Simple synthetic signal
            print("\n📊 Test 1: Synthetic 1kHz sine wave")
            test_wav = self.create_test_wav()
            self.test_encode_decode_round_trip(test_wav)
            self.test_cross_compatibility(test_wav)
            
            # Test 2: Real music (truncated)
            print("\n🎵 Test 2: Real music samples")
            for flac_path in self.test_files[:2]:  # Test first 2 files only
                if os.path.exists(flac_path):
                    wav_path = self.convert_flac_to_wav(flac_path)
                    if wav_path:
                        self.test_encode_decode_round_trip(wav_path)
                        self.test_cross_compatibility(wav_path)
                else:
                    print(f"⚠️  File not found: {flac_path}")
            
            print(f"\n🔍 All test files saved in: {self.temp_dir}")
            print("✅ Cross-validation tests completed!")
            
        except KeyboardInterrupt:
            print("\n⚠️  Tests interrupted by user")
        except Exception as e:
            print(f"\n❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Don't auto-cleanup so we can inspect results
            print(f"\n📁 Test files preserved in: {self.temp_dir}")
            print("   Run 'rm -rf {self.temp_dir}' to clean up manually")

def main():
    # Check dependencies
    try:
        import soundfile
    except ImportError:
        print("❌ soundfile library required. Install with: pip install soundfile")
        sys.exit(1)
    
    tester = CrossValidationTester()
    
    # Check if atracdenc is available
    if not os.path.exists(tester.atracdenc_path):
        print(f"❌ atracdenc not found at {tester.atracdenc_path}")
        sys.exit(1)
    
    tester.run_validation()

if __name__ == "__main__":
    main()