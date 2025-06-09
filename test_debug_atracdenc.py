#!/usr/bin/env python3
import wave
import numpy as np
import tempfile
import os
import subprocess

# Create a simple test WAV
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    test_wav = f.name

with wave.open(test_wav, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(44100)
    
    # 512 samples = exactly one frame
    signal = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(512) / 44100)
    signal_int = (signal * 32767).astype(np.int16)
    wav_file.writeframes(signal_int.tobytes())

print(f'Created test WAV: {test_wav}')

# Encode with PyATRAC1
result = subprocess.run(['python', 'atrac1_cli.py', '-m', 'encode', '-i', test_wav, '-o', 'debug_test.aea'], capture_output=True, text=True)
print(f'PyATRAC1 encoding: {result.returncode}')
if result.returncode != 0:
    print(f'Error: {result.stderr}')
else:
    print(f'Success: {result.stdout}')

if os.path.exists('debug_test.aea'):
    print(f'Created AEA file: {os.path.getsize("debug_test.aea")} bytes')
    
    # Try to decode with atracdenc (debug version)
    result = subprocess.run(['./atracdenc/build/src/atracdenc', '-d', '-i', 'debug_test.aea', '-o', 'debug_output.wav'], capture_output=True, text=True)
    print(f'atracdenc decoding result: {result.returncode}')
    print(f'Debug output: {result.stdout}')
    print(f'Debug errors: {result.stderr}')
else:
    print('No AEA file created')

# Cleanup
os.unlink(test_wav)
if os.path.exists('debug_test.aea'):
    os.unlink('debug_test.aea')
if os.path.exists('debug_output.wav'):
    os.unlink('debug_output.wav')