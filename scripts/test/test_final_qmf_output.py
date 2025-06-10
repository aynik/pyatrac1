#!/usr/bin/env python3
"""
Test the final QMF output buffers instead of raw IMDCT to see if +32 shift is already handled.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def test_final_qmf_output_vs_raw_imdct():
    """Compare final QMF output vs raw IMDCT to see if +32 shift is already corrected."""
    
    print("=== Final QMF Output vs Raw IMDCT Comparison ===")
    
    mdct = Atrac1MDCT()
    
    # Test with DC signal
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    print(f"Input: DC signal = 0.5")
    print(f"Input energy: {np.sum(test_input**2):.6f}")
    
    # Forward MDCT
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"MDCT DC coefficient: {low_coeffs[0]:.6f}")
    
    # 1. RAW IMDCT (what we've been debugging)
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"\n1. RAW IMDCT analysis:")
    print(f"   Shape: {raw_imdct.shape}")
    print(f"   Energy: {np.sum(raw_imdct**2):.6f}")
    print(f"   atracdenc region [64:192] energy: {np.sum(raw_imdct[64:192]**2):.6f}")
    print(f"   Our suspected region [96:224] energy: {np.sum(raw_imdct[96:224]**2):.6f}")
    
    # 2. FINAL QMF OUTPUT (what actually matters)
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    print(f"\n2. FINAL QMF OUTPUT analysis:")
    print(f"   low_out shape: {low_out.shape}")
    print(f"   low_out energy: {np.sum(low_out**2):.6f}")
    
    # According to atracdenc: main reconstruction is at dstBuf + 32
    # So useful output should be low_out[32:32+128] for long blocks
    useful_output = low_out[32:160]  # 128 samples starting at offset 32
    print(f"   Useful output [32:160] energy: {np.sum(useful_output**2):.6f}")
    print(f"   Useful output [32:160] mean: {np.mean(useful_output):.6f}")
    print(f"   Useful output [32:160] std: {np.std(useful_output):.6f}")
    
    # Calculate SNR with useful output
    if len(useful_output) == len(test_input):
        error = useful_output - test_input
        error_energy = np.sum(error**2)
        signal_energy = np.sum(test_input**2)
        
        if error_energy > 0:
            snr_db = 10 * np.log10(signal_energy / error_energy)
            print(f"   FINAL QMF SNR: {snr_db:.2f} dB")
            
            if snr_db > 40:
                print(f"   ✅ EXCELLENT (target achieved!)")
            elif snr_db > 20:
                print(f"   ✅ Good")
            elif snr_db > 5:
                print(f"   ⚠️  Fair")
            else:
                print(f"   ❌ Poor")
        else:
            print(f"   Perfect reconstruction!")

def test_different_extraction_regions():
    """Test different extraction regions from final QMF output."""
    
    print(f"\n=== Different Extraction Regions Test ===")
    
    mdct = Atrac1MDCT()
    
    # DC test
    test_input = np.ones(128, dtype=np.float32) * 0.5
    
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, test_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    # Test different extraction regions
    extraction_regions = [
        ("atracdenc style [32:160]", low_out[32:160]),  # dstBuf + 32, length 128
        ("No offset [0:128]", low_out[0:128]),
        ("Middle [64:192]", low_out[64:192]),
        ("Late [96:224]", low_out[96:224]),
    ]
    
    print(f"Testing different extraction regions from final QMF output:")
    
    best_snr = -100
    best_region = None
    
    for name, extracted in extraction_regions:
        if len(extracted) == len(test_input):
            error = extracted - test_input
            error_energy = np.sum(error**2)
            signal_energy = np.sum(test_input**2)
            
            if error_energy > 0:
                snr_db = 10 * np.log10(signal_energy / error_energy)
                print(f"  {name}: SNR = {snr_db:.2f} dB")
                
                if snr_db > best_snr:
                    best_snr = snr_db
                    best_region = name
                    
                # Check reconstruction quality
                region_mean = np.mean(extracted)
                region_std = np.std(extracted)
                print(f"      mean={region_mean:.6f}, std={region_std:.6f}")
                
                # For DC 0.5, we expect mean around 0.5 (or scaled version)
                if abs(region_mean - 0.5) < 0.05:
                    print(f"      ✅ Good DC reconstruction!")
                elif abs(region_mean - 0.125) < 0.05:  # 4x scaling
                    print(f"      ✅ Good DC reconstruction (4x scaling)!")
            else:
                print(f"  {name}: Perfect reconstruction!")
                best_snr = float('inf')
                best_region = name
    
    print(f"\nBest extraction region: {best_region} with SNR {best_snr:.2f} dB")

def test_multi_frame_tdac():
    """Test multi-frame TDAC with final QMF output."""
    
    print(f"\n=== Multi-Frame TDAC Test (Final QMF) ===")
    
    mdct = Atrac1MDCT()
    
    # Frame 1
    frame1_input = np.ones(128, dtype=np.float32) * 0.5
    
    specs1 = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs1, frame1_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_out1 = np.zeros(256, dtype=np.float32)
    mid_out1 = np.zeros(256, dtype=np.float32)
    hi_out1 = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs1, BlockSizeMode(False, False, False),
               low_out1, mid_out1, hi_out1, channel=0, frame=0)
    
    print(f"Frame 1 final QMF output:")
    print(f"  Overlap region [0:32] max: {np.max(np.abs(low_out1[:32])):.6f}")
    print(f"  Main region [32:160] mean: {np.mean(low_out1[32:160]):.6f}")
    
    # Frame 2
    frame2_input = np.ones(128, dtype=np.float32) * 0.8
    
    specs2 = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs2, frame2_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=1)
    
    low_out2 = np.zeros(256, dtype=np.float32)
    mid_out2 = np.zeros(256, dtype=np.float32)
    hi_out2 = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs2, BlockSizeMode(False, False, False),
               low_out2, mid_out2, hi_out2, channel=0, frame=1)
    
    print(f"Frame 2 final QMF output:")
    print(f"  Overlap region [0:32] max: {np.max(np.abs(low_out2[:32])):.6f}")
    print(f"  Main region [32:160] mean: {np.mean(low_out2[32:160]):.6f}")
    
    if np.max(np.abs(low_out2[:32])) > 0.01:
        print(f"  ✅ Frame 2 has meaningful overlap (TDAC working)")
    else:
        print(f"  ❌ Frame 2 still has tiny overlap (TDAC not working)")

if __name__ == "__main__":
    test_final_qmf_output_vs_raw_imdct()
    test_different_extraction_regions()
    test_multi_frame_tdac()
    
    print(f"\n=== CRITICAL REALIZATION ===")
    print("If final QMF output [32:160] gives good reconstruction,")
    print("then our +32 shift issue was in the wrong analysis target!")
    print("We should have been testing final QMF output, not raw IMDCT.")