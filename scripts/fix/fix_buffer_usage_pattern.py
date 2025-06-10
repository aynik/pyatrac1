#!/usr/bin/env python3
"""
Fix the buffer usage pattern - use correct parts of atracdenc extraction.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def analyze_correct_buffer_usage():
    """Analyze how atracdenc extraction should be used for overlap vs middle."""
    
    print("=== Correct Buffer Usage Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # Test with DC signal to see extraction pattern clearly
    test_input = np.ones(128, dtype=np.float32) * 1.0
    
    # Get MDCT coefficients
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    # Get raw IMDCT and atracdenc extraction
    raw_imdct = mdct.imdct256(specs[:128])
    atracdenc_extraction = raw_imdct[64:192]  # What atracdenc extracts
    
    print(f"atracdenc extraction [64:192] pattern:")
    print(f"  Total energy: {np.sum(atracdenc_extraction**2):.6f}")
    
    # Analyze in 16-sample chunks to understand the pattern
    chunks = []
    for i in range(0, 128, 16):
        chunk = atracdenc_extraction[i:i+16]
        chunks.append(chunk)
        print(f"  Chunk {i//16} [{i}:{i+16}]: energy={np.sum(chunk**2):.6f}, mean={np.mean(chunk):.6f}, max={np.max(np.abs(chunk)):.6f}")
        print(f"    Values: {chunk[:4]} ...")
    
    # The pattern should show us:
    # - Which chunks are near-zero (for overlap/windowing)
    # - Which chunks have meaningful reconstruction data
    
    print(f"\n=== Current Implementation Analysis ===")
    
    # What we currently do:
    # 1. Use inv_buf[0:16] for vector_fmul_window (overlap)
    # 2. Use inv_buf[16:128] for middle copy
    
    current_overlap_data = atracdenc_extraction[:16]    # First chunk
    current_middle_data = atracdenc_extraction[16:128]  # Rest of extraction
    
    print(f"Current overlap data [0:16]: energy={np.sum(current_overlap_data**2):.6f}")
    print(f"Current middle data [16:128]: energy={np.sum(current_middle_data**2):.6f}")
    
    # Test what happens if we use different splits
    print(f"\n=== Alternative Buffer Usage Patterns ===")
    
    alternatives = [
        ("Current [0:16] + [16:128]", atracdenc_extraction[:16], atracdenc_extraction[16:128]),
        ("Alt1 [112:128] + [0:112]", atracdenc_extraction[112:128], atracdenc_extraction[:112]),
        ("Alt2 [0:16] + [48:128]", atracdenc_extraction[:16], atracdenc_extraction[48:128]),
        ("Alt3 [96:112] + [16:96]", atracdenc_extraction[96:112], atracdenc_extraction[16:96]),
    ]
    
    for name, overlap_part, middle_part in alternatives:
        overlap_energy = np.sum(overlap_part**2)
        middle_energy = np.sum(middle_part**2)
        
        print(f"{name}:")
        print(f"  Overlap energy: {overlap_energy:.6f}")
        print(f"  Middle energy: {middle_energy:.6f}")
        print(f"  Overlap max: {np.max(np.abs(overlap_part)):.6f}")
        print(f"  Middle max: {np.max(np.abs(middle_part)):.6f}")
        
        # For good TDAC, we want:
        # - Overlap data with reasonable values (not near-zero)
        # - Middle data with strong reconstruction values
        
        if np.max(np.abs(overlap_part)) > 0.01 and np.max(np.abs(middle_part)) > 0.1:
            print(f"  ✅ Both overlap and middle have meaningful data")
        elif np.max(np.abs(overlap_part)) < 0.01:
            print(f"  ❌ Overlap data is too small")
        elif np.max(np.abs(middle_part)) < 0.1:
            print(f"  ❌ Middle data is too small")
        else:
            print(f"  ⚠️  Unclear data distribution")

def test_theoretical_perfect_reconstruction():
    """Test what perfect reconstruction should look like."""
    
    print(f"\n=== Theoretical Perfect Reconstruction ===")
    
    mdct = Atrac1MDCT()
    
    # For a lossless transform, we should achieve very high SNR
    print("Testing perfect reconstruction expectations:")
    
    # Simple test cases
    test_cases = [
        ("DC 1.0", np.ones(128, dtype=np.float32)),
        ("DC 0.5", np.ones(128, dtype=np.float32) * 0.5),
        ("Impulse", np.concatenate([np.zeros(64, dtype=np.float32), [1.0], np.zeros(63, dtype=np.float32)])),
    ]
    
    for name, test_input in test_cases:
        print(f"\n{name}:")
        
        input_energy = np.sum(test_input**2)
        
        # MDCT->IMDCT round trip
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        # Extract useful reconstruction
        useful_output = low_out[32:160]  # 128 samples to match input
        
        if len(useful_output) == len(test_input):
            # Calculate error and SNR
            error = useful_output - test_input
            error_energy = np.sum(error**2)
            
            if error_energy > 0:
                snr_db = 10 * np.log10(input_energy / error_energy)
                
                print(f"  SNR: {snr_db:.2f} dB")
                
                if snr_db > 40:
                    print(f"  ✅ Excellent (theoretical expectation)")
                elif snr_db > 20:
                    print(f"  ✅ Good")
                elif snr_db > 5:
                    print(f"  ⚠️  Fair")
                else:
                    print(f"  ❌ Poor (indicates fundamental issues)")
                
                # Energy analysis
                output_energy = np.sum(useful_output**2)
                energy_ratio = output_energy / input_energy
                print(f"  Energy ratio: {energy_ratio:.6f}")
                
                # Correlation
                correlation = np.corrcoef(test_input, useful_output)[0, 1]
                print(f"  Correlation: {correlation:.6f}")
            else:
                print(f"  Perfect reconstruction (zero error)")

def debug_atracdenc_buffer_behavior():
    """Debug exactly how atracdenc uses its buffer extraction."""
    
    print(f"\n=== atracdenc Buffer Behavior Debug ===")
    
    # From atracdenc source analysis:
    # 1. Extract inv[i + inv.size()/4] into invBuf[start+i]
    # 2. Use &invBuf[start] for vector_fmul_window (first 16 samples)
    # 3. Use &invBuf[16] for middle copy (memcpy)
    # 4. Use invBuf[bufSz - 16 + j] for tail update
    
    print("atracdenc buffer usage pattern:")
    print("1. invBuf[0:16] -> vector_fmul_window overlap")
    print("2. invBuf[16:128] -> middle copy (memcpy)")
    print("3. invBuf[112:128] -> tail for next frame")
    
    # This means:
    # - The first 16 samples of extraction are SUPPOSED to be used for overlap
    # - Even if they're near-zero, that might be correct for first frame (prev_buf = zeros)
    
    print("\nFor first frame (prev_buf = zeros):")
    print("- vector_fmul_window with zero prev_buf and near-zero inv_buf gives near-zero output")
    print("- This might be CORRECT behavior!")
    
    print("\nThe real test is multi-frame TDAC where prev_buf has data from previous frame")
    
    # Let's test this theory
    mdct = Atrac1MDCT()
    
    # Frame 1: Initialize
    frame1_input = np.ones(128, dtype=np.float32) * 0.5
    
    specs1 = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs1, frame1_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out1 = np.zeros(256, dtype=np.float32)
    mid_out1 = np.zeros(256, dtype=np.float32)
    hi_out1 = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs1, BlockSizeMode(False, False, False),
               low_out1, mid_out1, hi_out1, channel=0, frame=0)
    
    print(f"\nFrame 1 results:")
    print(f"  Overlap region [0:32]: max={np.max(np.abs(low_out1[:32])):.6f}")
    print(f"  Middle region [32:144]: mean={np.mean(low_out1[32:144]):.6f}")
    print(f"  Tail region [240:256]: {low_out1[240:256][:4]} ...")
    
    # Frame 2: Should use Frame 1's tail
    frame2_input = np.ones(128, dtype=np.float32) * 0.8
    
    specs2 = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs2, frame2_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=1)
    
    low_out2 = np.zeros(256, dtype=np.float32)
    mid_out2 = np.zeros(256, dtype=np.float32)
    hi_out2 = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs2, BlockSizeMode(False, False, False),
               low_out2, mid_out2, hi_out2, channel=0, frame=1)
    
    print(f"\nFrame 2 results:")
    print(f"  Overlap region [0:32]: max={np.max(np.abs(low_out2[:32])):.6f}")
    print(f"  Middle region [32:144]: mean={np.mean(low_out2[32:144]):.6f}")
    
    # Frame 2 should have meaningful overlap because prev_buf has data from Frame 1
    if np.max(np.abs(low_out2[:32])) > 0.01:
        print(f"  ✅ Frame 2 has meaningful overlap (multi-frame TDAC working)")
    else:
        print(f"  ❌ Frame 2 still has tiny overlap (TDAC not working)")

if __name__ == "__main__":
    analyze_correct_buffer_usage()
    test_theoretical_perfect_reconstruction()
    debug_atracdenc_buffer_behavior()
    
    print(f"\n=== CONCLUSION ===")
    print("The issue may not be buffer extraction, but how we handle the extracted data.")
    print("atracdenc extracts [64:192] and this contains the right pattern:")
    print("- Near-zero values at start (correct for first frame overlap)")
    print("- Strong values at end (correct for reconstruction)")
    print("The problem might be elsewhere in our implementation.")