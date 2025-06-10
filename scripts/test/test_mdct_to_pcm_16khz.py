#!/usr/bin/env python3
"""
MDCT to PCM transform test with 16kHz samples and comprehensive logging.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_mdct_to_pcm_16khz():
    """Test MDCT to PCM transform with 16kHz samples and detailed logging."""
    
    print("=== MDCT to PCM Transform Test (16kHz) ===")
    
    # Ensure all operations use np.float32
    mdct = Atrac1MDCT()
    
    # Generate 16kHz test signal
    sample_rate = 16000
    duration = 128 / sample_rate  # 128 samples duration
    freq = 1000  # 1kHz tone
    
    t = np.arange(128, dtype=np.float32) / sample_rate
    test_signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    print(f"Test signal: {freq}Hz sine wave")
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Duration: {duration:.6f}s")
    print(f"Signal dtype: {test_signal.dtype}")
    print(f"Signal shape: {test_signal.shape}")
    print(f"Signal energy: {np.sum(test_signal**2):.6f}")
    print(f"Signal range: [{np.min(test_signal):.6f}, {np.max(test_signal):.6f}]")
    print(f"First 8 samples: {test_signal[:8]}")
    print(f"Last 8 samples: {test_signal[-8:]}")
    
    # Prepare inputs (ensure np.float32)
    low_input = test_signal.astype(np.float32)
    mid_input = np.zeros(128, dtype=np.float32)
    hi_input = np.zeros(256, dtype=np.float32)
    
    print(f"\nInput verification:")
    print(f"  low_input dtype: {low_input.dtype}, shape: {low_input.shape}")
    print(f"  mid_input dtype: {mid_input.dtype}, shape: {mid_input.shape}")
    print(f"  hi_input dtype: {hi_input.dtype}, shape: {hi_input.shape}")
    
    # Log initial buffer states
    print(f"\nInitial buffer states:")
    print(f"  pcm_buf_low[0] dtype: {mdct.pcm_buf_low[0].dtype}, shape: {mdct.pcm_buf_low[0].shape}")
    print(f"  pcm_buf_low[0] first 8: {mdct.pcm_buf_low[0][:8]}")
    print(f"  pcm_buf_low[0] last 8: {mdct.pcm_buf_low[0][-8:]}")
    
    # MDCT Transform
    print(f"\n=== FORWARD MDCT ===")
    
    specs = np.zeros(512, dtype=np.float32)
    print(f"specs dtype: {specs.dtype}, shape: {specs.shape}")
    
    # Call MDCT with detailed state logging
    mdct.mdct(specs, low_input, mid_input, hi_input, 
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    # Log MDCT results
    low_coeffs = specs[:128]
    mid_coeffs = specs[128:256]
    hi_coeffs = specs[256:512]
    
    print(f"MDCT coefficients analysis:")
    print(f"  Low coeffs dtype: {low_coeffs.dtype}, shape: {low_coeffs.shape}")
    print(f"  Low coeffs energy: {np.sum(low_coeffs**2):.6f}")
    print(f"  Low coeffs max: {np.max(np.abs(low_coeffs)):.6f}")
    print(f"  Low coeffs first 8: {low_coeffs[:8]}")
    print(f"  Low coeffs DC: {low_coeffs[0]:.6f}")
    print(f"  Mid coeffs energy: {np.sum(mid_coeffs**2):.6f}")
    print(f"  Hi coeffs energy: {np.sum(hi_coeffs**2):.6f}")
    
    # Log buffer states after MDCT
    print(f"\nBuffer states after MDCT:")
    print(f"  pcm_buf_low[0] first 8: {mdct.pcm_buf_low[0][:8]}")
    print(f"  pcm_buf_low[0][120:136]: {mdct.pcm_buf_low[0][120:136]}")  # Input + overlap region
    print(f"  tmp_buffers[0][0] dtype: {mdct.tmp_buffers[0][0].dtype}")
    print(f"  tmp_buffers[0][0] first 8: {mdct.tmp_buffers[0][0][:8]}")
    
    # IMDCT Transform
    print(f"\n=== INVERSE MDCT ===")
    
    # Prepare output buffers (ensure np.float32)
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    print(f"Output buffer verification:")
    print(f"  low_out dtype: {low_out.dtype}, shape: {low_out.shape}")
    print(f"  mid_out dtype: {mid_out.dtype}, shape: {mid_out.shape}")
    print(f"  hi_out dtype: {hi_out.dtype}, shape: {hi_out.shape}")
    
    # Call IMDCT with detailed state logging
    mdct.imdct(specs, BlockSizeMode(False, False, False), 
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    # Log IMDCT results
    print(f"IMDCT output analysis:")
    print(f"  Low output energy: {np.sum(low_out**2):.6f}")
    print(f"  Low output max: {np.max(np.abs(low_out)):.6f}")
    print(f"  Low output first 8: {low_out[:8]}")
    print(f"  Low output last 8: {low_out[-8:]}")
    
    # Critical intermediate analysis
    print(f"\n=== CRITICAL BUFFER ANALYSIS ===")
    
    # Manual IMDCT to see raw output vs our extraction
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"Raw IMDCT analysis:")
    print(f"  Raw IMDCT dtype: {raw_imdct.dtype}, shape: {raw_imdct.shape}")
    print(f"  Raw IMDCT energy: {np.sum(raw_imdct**2):.6f}")
    
    # Verify our extraction logic
    inv_len = len(raw_imdct)  # 256
    print(f"  inv_len: {inv_len}")
    
    # Before fix: inv[64:192]
    old_extraction = raw_imdct[64:192]
    # After fix: inv[96:224]
    new_extraction = raw_imdct[96:224]
    
    print(f"  Old extraction [64:192]: energy={np.sum(old_extraction**2):.6f}, first 8: {old_extraction[:8]}")
    print(f"  New extraction [96:224]: energy={np.sum(new_extraction**2):.6f}, first 8: {new_extraction[:8]}")
    
    # Compare with atracdenc formula: inv[i + inv.size()/4]
    atracdenc_start = inv_len // 4  # inv.size()/4 = 64
    print(f"  atracdenc formula start: {atracdenc_start}")
    print(f"  atracdenc would extract: inv[{atracdenc_start}:{atracdenc_start + 128}]")
    
    # This means our NEW extraction [96:224] is actually WRONG!
    # atracdenc extracts [64:192], not [96:224]
    correct_extraction = raw_imdct[atracdenc_start:atracdenc_start + 128]
    print(f"  Correct atracdenc extraction: energy={np.sum(correct_extraction**2):.6f}, first 8: {correct_extraction[:8]}")
    
    # Reconstruction quality analysis
    print(f"\n=== RECONSTRUCTION QUALITY ===")
    
    # Extract useful region for comparison
    useful_output = low_out[32:160]  # 128 samples to match input
    
    print(f"Input vs Output comparison:")
    print(f"  Input energy: {np.sum(test_signal**2):.6f}")
    print(f"  Useful output energy: {np.sum(useful_output**2):.6f}")
    print(f"  Energy ratio: {np.sum(useful_output**2) / np.sum(test_signal**2):.6f}")
    
    # Calculate error and SNR
    if len(useful_output) == len(test_signal):
        error = useful_output - test_signal
        error_energy = np.sum(error**2)
        signal_energy = np.sum(test_signal**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
            print(f"  SNR: {snr_db:.2f} dB")
            
            if snr_db > 40:
                print("  ✅ EXCELLENT reconstruction")
            elif snr_db > 20:
                print("  ✅ Good reconstruction")
            elif snr_db > 5:
                print("  ⚠️  Fair reconstruction")
            else:
                print("  ❌ Poor reconstruction")
        else:
            print("  Perfect reconstruction (zero error)")
    
    # Correlation analysis
    if len(useful_output) == len(test_signal):
        correlation = np.corrcoef(test_signal, useful_output)[0, 1]
        print(f"  Correlation: {correlation:.6f}")
    
    return snr_db if 'snr_db' in locals() else 0

def dump_intermediate_states():
    """Dump all intermediate buffer states for comparison with atracdenc."""
    
    print(f"\n=== INTERMEDIATE STATE DUMP ===")
    
    mdct = Atrac1MDCT()
    
    # Simple test case for easier comparison
    test_input = np.ones(128, dtype=np.float32) * 0.5  # DC signal
    
    print(f"Test input: DC signal (0.5)")
    print(f"Input dtype: {test_input.dtype}")
    
    # Before MDCT
    print(f"\nBEFORE MDCT:")
    print(f"pcm_buf_low[0] shape: {mdct.pcm_buf_low[0].shape}")
    print(f"pcm_buf_low[0] first 16: {mdct.pcm_buf_low[0][:16]}")
    print(f"pcm_buf_low[0] last 16: {mdct.pcm_buf_low[0][-16:]}")
    
    # MDCT call
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    # After MDCT
    print(f"\nAFTER MDCT:")
    print(f"pcm_buf_low[0] first 16: {mdct.pcm_buf_low[0][:16]}")
    print(f"pcm_buf_low[0][120:144]: {mdct.pcm_buf_low[0][120:144]}")  # Input + overlap
    print(f"tmp_buffers[0][0] first 16: {mdct.tmp_buffers[0][0][:16]}")
    print(f"tmp_buffers[0][0][48:64]: {mdct.tmp_buffers[0][0][48:64]}")  # win_start region
    print(f"specs[:16]: {specs[:16]}")
    
    # Manual IMDCT for raw output
    raw_imdct = mdct.imdct256(specs[:128])
    print(f"\nRAW IMDCT OUTPUT:")
    print(f"raw_imdct dtype: {raw_imdct.dtype}, shape: {raw_imdct.shape}")
    print(f"raw_imdct first 16: {raw_imdct[:16]}")
    print(f"raw_imdct[64:80]: {raw_imdct[64:80]}")  # atracdenc extraction start
    print(f"raw_imdct[96:112]: {raw_imdct[96:112]}")  # our extraction start
    print(f"raw_imdct last 16: {raw_imdct[-16:]}")
    
    # IMDCT call
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32) 
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    # After IMDCT
    print(f"\nAFTER IMDCT:")
    print(f"low_out first 16: {low_out[:16]}")
    print(f"low_out[32:48]: {low_out[32:48]}")  # Middle region start
    print(f"low_out[224:240]: {low_out[224:240]}")  # Tail region
    print(f"low_out last 16: {low_out[-16:]}")

if __name__ == "__main__":
    snr = test_mdct_to_pcm_16khz()
    dump_intermediate_states()
    
    print(f"\n=== CRITICAL FINDING ===")
    print(f"Our +32 offset fix may be WRONG!")
    print(f"atracdenc uses inv[i + inv.size()/4] = inv[64:192]")
    print(f"We changed to inv[96:224] which doesn't match!")
    print(f"This could explain the poor SNR performance.")