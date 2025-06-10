#!/usr/bin/env python3

"""
Debug script to calculate SinCos values from atracdenc MDCT implementation
and check if -0.372549 matches any of the computed values.
"""

import math
import numpy as np

def calc_sincos_atracdenc(n, scale=1.0):
    """
    Replicate the CalcSinCos function from atracdenc MDCT implementation.
    
    From mdct.cpp:
    const float alpha = 2.0 * M_PI / (8.0 * n);
    const float omiga = 2.0 * M_PI / n;
    scale = sqrt(scale/n); 
    for (size_t i = 0; i < (n >> 2); ++i) {
        tmp[2 * i + 0] = scale * cos(omiga * i + alpha);
        tmp[2 * i + 1] = scale * sin(omiga * i + alpha);
    }
    """
    tmp = []
    alpha = 2.0 * math.pi / (8.0 * n)
    omiga = 2.0 * math.pi / n
    scale = math.sqrt(scale / n)
    
    print(f"n={n}, alpha={alpha:.6f}, omiga={omiga:.6f}, scale={scale:.6f}")
    
    for i in range(n >> 2):  # n >> 2 = n/4
        cos_val = scale * math.cos(omiga * i + alpha)
        sin_val = scale * math.sin(omiga * i + alpha)
        tmp.append(cos_val)  # tmp[2*i + 0]
        tmp.append(sin_val)  # tmp[2*i + 1]
        
        print(f"i={i:2d}: cos={cos_val:9.6f}, sin={sin_val:9.6f}")
        
        # Check if any value is close to -0.372549
        target = -0.372549
        if abs(cos_val - target) < 1e-5:
            print(f"*** MATCH: cos({omiga * i + alpha:.6f}) = {cos_val:.6f} ≈ {target}")
        if abs(sin_val - target) < 1e-5:
            print(f"*** MATCH: sin({omiga * i + alpha:.6f}) = {sin_val:.6f} ≈ {target}")
        if abs(-cos_val - target) < 1e-5:
            print(f"*** MATCH: -cos({omiga * i + alpha:.6f}) = {-cos_val:.6f} ≈ {target}")
        if abs(-sin_val - target) < 1e-5:
            print(f"*** MATCH: -sin({omiga * i + alpha:.6f}) = {-sin_val:.6f} ≈ {target}")
    
    print()
    return tmp

def main():
    target_value = -0.372549
    print(f"Looking for matches to {target_value}")
    print("=" * 60)
    
    # Test common MDCT sizes used in ATRAC1
    # ATRAC1 uses different sizes: 64, 256, 512 for different bands
    mdct_sizes = [64, 256, 512, 128]  # Common sizes in audio codecs
    
    for size in mdct_sizes:
        print(f"\nMDCT size {size}:")
        print("-" * 40)
        sincos_values = calc_sincos_atracdenc(size, scale=1.0)
        
        # Also test with different scales
        print(f"\nMDCT size {size} with scale=0.5:")
        print("-" * 40)
        sincos_values_scaled = calc_sincos_atracdenc(size, scale=0.5)
    
    # Let's also check direct angle calculations
    print("\n" + "=" * 60)
    print("Direct angle analysis:")
    print("=" * 60)
    
    # What angle would give us -0.372549?
    angle_from_cos = math.acos(-target_value)
    angle_from_sin = math.asin(-target_value) if abs(target_value) <= 1.0 else None
    
    print(f"If -0.372549 = cos(θ), then θ = {angle_from_cos:.6f} rad = {math.degrees(angle_from_cos):.2f}°")
    if angle_from_sin is not None:
        print(f"If -0.372549 = sin(θ), then θ = {angle_from_sin:.6f} rad = {math.degrees(angle_from_sin):.2f}°")
    
    # Let's check common angles
    angles_deg = [22.5, 30, 45, 60, 67.5, 90, 112.5, 135, 157.5, 180]
    print("\nCommon angles:")
    for deg in angles_deg:
        rad = math.radians(deg)
        cos_val = math.cos(rad)
        sin_val = math.sin(rad)
        print(f"{deg:5.1f}°: cos={cos_val:8.6f}, sin={sin_val:8.6f}, -cos={-cos_val:8.6f}, -sin={-sin_val:8.6f}")
        
        if abs(cos_val - target_value) < 1e-5 or abs(-cos_val - target_value) < 1e-5:
            print(f"    *** MATCH at {deg}° (cos)")
        if abs(sin_val - target_value) < 1e-5 or abs(-sin_val - target_value) < 1e-5:
            print(f"    *** MATCH at {deg}° (sin)")

if __name__ == "__main__":
    main()