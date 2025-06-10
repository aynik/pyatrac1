#!/usr/bin/env python3
"""
Debug why middle regions aren't constant for DC signals.
"""

import numpy as np
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode

def analyze_middle_region_variation():
    """Analyze what causes middle region variation in DC reconstruction."""
    
    print("=== Middle Region Variation Analysis ===")
    
    mdct = Atrac1MDCT()
    
    # Test with perfect DC
    dc_level = 1.0
    low_input = np.ones(128, dtype=np.float32) * dc_level
    
    print(f"Input: constant DC level {dc_level}")
    
    # Step-by-step analysis
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    print(f"\nMDCT coefficients analysis:")
    print(f"  DC coeff (specs[0]): {low_coeffs[0]:.6f}")
    print(f"  Non-DC coeffs energy: {np.sum(low_coeffs[1:]**2):.6f}")
    print(f"  Total energy: {np.sum(low_coeffs**2):.6f}")
    
    # Raw IMDCT analysis
    raw_imdct = mdct.imdct256(low_coeffs)
    print(f"\nRaw IMDCT analysis:")
    print(f"  Total energy: {np.sum(raw_imdct**2):.6f}")
    
    # Analyze each quarter for DC distribution
    quarters = [
        ("Q1 [0:64]", raw_imdct[:64]),
        ("Q2 [64:128]", raw_imdct[64:128]),
        ("Q3 [128:192]", raw_imdct[128:192]),
        ("Q4 [192:256]", raw_imdct[192:256])
    ]
    
    for name, quarter in quarters:
        mean_val = np.mean(quarter)
        std_val = np.std(quarter)
        print(f"  {name}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    # Our extraction [96:224] analysis
    our_extraction = raw_imdct[96:224]
    print(f"\nOur extraction [96:224] analysis:")
    print(f"  Length: {len(our_extraction)}")
    print(f"  Energy: {np.sum(our_extraction**2):.6f}")
    
    # Break down extraction into regions
    overlap_data = our_extraction[:16]  # For vector_fmul_window
    middle_copy = our_extraction[16:128]  # For middle copy (112 samples)
    tail_data = our_extraction[112:]    # For next frame tail
    
    print(f"  Overlap data [0:16]: mean={np.mean(overlap_data):.6f}, std={np.std(overlap_data):.6f}")
    print(f"  Middle copy [16:128]: mean={np.mean(middle_copy):.6f}, std={np.std(middle_copy):.6f}")
    print(f"  Tail data [112:128]: mean={np.mean(tail_data):.6f}, std={np.std(tail_data):.6f}")
    
    # Full IMDCT pipeline
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    print(f"\nFull IMDCT pipeline analysis:")
    
    # Analyze each region
    overlap_start = low_out[:32]        # vector_fmul_window output
    middle_region = low_out[32:144]     # Middle copy region (112 samples)
    middle_extended = low_out[32:224]   # Extended middle for analysis
    overlap_end = low_out[224:]         # End overlap region
    
    print(f"  Overlap start [0:32]: mean={np.mean(overlap_start):.6f}, std={np.std(overlap_start):.6f}")
    print(f"  Middle region [32:144]: mean={np.mean(middle_region):.6f}, std={np.std(middle_region):.6f}")
    print(f"  Middle extended [32:224]: mean={np.mean(middle_extended):.6f}, std={np.std(middle_extended):.6f}")
    print(f"  Overlap end [224:256]: mean={np.mean(overlap_end):.6f}, std={np.std(overlap_end):.6f}")
    
    # Expected vs actual for DC
    expected_dc_response = dc_level * 0.25  # Expected 25% scaling
    actual_middle_mean = np.mean(middle_region)
    scaling_factor = actual_middle_mean / dc_level
    
    print(f"\nDC reconstruction analysis:")
    print(f"  Expected output: {expected_dc_response:.6f}")
    print(f"  Actual middle mean: {actual_middle_mean:.6f}")
    print(f"  Scaling factor: {scaling_factor:.6f}")
    print(f"  Scaling error: {abs(scaling_factor - 0.25):.6f}")
    
    # Analyze variation sources
    middle_variation = np.std(middle_region)
    relative_variation = middle_variation / abs(actual_middle_mean) if actual_middle_mean != 0 else float('inf')
    
    print(f"  Middle variation (std): {middle_variation:.6f}")
    print(f"  Relative variation: {relative_variation:.4f} ({relative_variation*100:.1f}%)")
    
    if relative_variation < 0.1:
        print("  ✅ Middle region is reasonably constant")
    elif relative_variation < 0.3:
        print("  ⚠️  Middle region has moderate variation")
    else:
        print("  ❌ Middle region varies too much")
    
    return middle_region, our_extraction

def test_different_dc_levels():
    """Test how different DC levels affect middle region variation."""
    
    print(f"\n=== Different DC Levels Test ===")
    
    mdct = Atrac1MDCT()
    
    dc_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    print("Testing middle region consistency across DC levels:")
    
    scaling_factors = []
    variations = []
    
    for dc_level in dc_levels:
        low_input = np.ones(128, dtype=np.float32) * dc_level
        
        # MDCT->IMDCT
        specs = np.zeros(512, dtype=np.float32)
        mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
                  BlockSizeMode(False, False, False), channel=0, frame=0)
        
        low_out = np.zeros(256, dtype=np.float32)
        mid_out = np.zeros(256, dtype=np.float32)
        hi_out = np.zeros(512, dtype=np.float32)
        
        mdct.imdct(specs, BlockSizeMode(False, False, False),
                   low_out, mid_out, hi_out, channel=0, frame=0)
        
        middle_region = low_out[32:144]
        
        mean_val = np.mean(middle_region)
        std_val = np.std(middle_region)
        scaling_factor = mean_val / dc_level
        relative_variation = std_val / abs(mean_val) if mean_val != 0 else float('inf')
        
        scaling_factors.append(scaling_factor)
        variations.append(relative_variation)
        
        print(f"  DC {dc_level:4.2f}: mean={mean_val:.6f}, scaling={scaling_factor:.6f}, variation={relative_variation:.4f}")
    
    # Check consistency
    scaling_std = np.std(scaling_factors)
    variation_mean = np.mean(variations)
    
    print(f"\nConsistency analysis:")
    print(f"  Scaling factor std: {scaling_std:.6f}")
    print(f"  Average variation: {variation_mean:.4f}")
    
    if scaling_std < 0.01:
        print("  ✅ Scaling is consistent across levels")
    else:
        print("  ⚠️  Scaling varies across levels")
    
    if variation_mean < 0.1:
        print("  ✅ Variation is low across levels")
    else:
        print("  ❌ High variation across levels")

def debug_windowing_effects():
    """Debug if windowing is causing middle region variation."""
    
    print(f"\n=== Windowing Effects Debug ===")
    
    mdct = Atrac1MDCT()
    
    # Compare DC with and without windowing effects
    dc_level = 1.0
    low_input = np.ones(128, dtype=np.float32) * dc_level
    
    print(f"Analyzing windowing impact on DC level {dc_level}")
    
    # Get raw MDCT coefficients
    specs = np.zeros(512, dtype=np.float32)
    mdct.mdct(specs, low_input, np.zeros(128, dtype=np.float32), np.zeros(256, dtype=np.float32),
              BlockSizeMode(False, False, False), channel=0, frame=0)
    
    low_coeffs = specs[:128]
    
    # Manual IMDCT to see raw reconstruction
    raw_imdct = mdct.imdct256(low_coeffs)
    
    # Our extraction for middle copy
    extraction = raw_imdct[96:224]
    middle_copy_source = extraction[16:128]  # What gets copied to middle
    
    print(f"Raw IMDCT middle copy source analysis:")
    print(f"  Source mean: {np.mean(middle_copy_source):.6f}")
    print(f"  Source std: {np.std(middle_copy_source):.6f}")
    print(f"  Source variation: {np.std(middle_copy_source)/abs(np.mean(middle_copy_source)):.4f}")
    
    # Full pipeline
    low_out = np.zeros(256, dtype=np.float32)
    mid_out = np.zeros(256, dtype=np.float32)
    hi_out = np.zeros(512, dtype=np.float32)
    
    mdct.imdct(specs, BlockSizeMode(False, False, False),
               low_out, mid_out, hi_out, channel=0, frame=0)
    
    middle_result = low_out[32:144]
    
    print(f"Pipeline middle result analysis:")
    print(f"  Result mean: {np.mean(middle_result):.6f}")
    print(f"  Result std: {np.std(middle_result):.6f}")
    print(f"  Result variation: {np.std(middle_result)/abs(np.mean(middle_result)):.4f}")
    
    # Compare source vs result
    if len(middle_copy_source) == len(middle_result):
        copy_error = np.mean(np.abs(middle_copy_source - middle_result))
        print(f"  Copy error: {copy_error:.8f}")
        
        if copy_error < 1e-6:
            print("  ✅ Middle copy is exact")
        else:
            print("  ⚠️  Middle copy has errors")
    
    # The variation is likely coming from raw IMDCT itself
    print(f"\nConclusion:")
    print(f"  Middle region variation comes from raw IMDCT output")
    print(f"  This suggests the issue is in MDCT->IMDCT perfect reconstruction")

if __name__ == "__main__":
    middle_region, extraction = analyze_middle_region_variation()
    test_different_dc_levels()
    debug_windowing_effects()
    
    print(f"\n=== Summary ===")
    print("Middle region variation analysis complete.")
    print("Check individual test results to identify the root cause.")