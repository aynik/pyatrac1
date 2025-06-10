#!/usr/bin/env python3

"""
Debug script to calculate the SineWindow values from atracdenc ATRAC1 implementation
and check if -0.372549 matches any of these windowing values.
"""

import math

def calc_sine_window_atracdenc():
    """
    Replicate the SineWindow calculation from atracdenc ATRAC1 implementation.
    
    From atrac1.h line 132-135:
    for (uint32_t i = 0; i < 32; i++) {
        SineWindow[i] = sin((i + 0.5) * (M_PI / (2.0 * 32.0)));
    }
    """
    sine_window = []
    target_value = -0.372549
    
    print("SineWindow calculation from atracdenc:")
    print("sin((i + 0.5) * (M_PI / (2.0 * 32.0)))")
    print("=" * 50)
    
    for i in range(32):
        angle = (i + 0.5) * (math.pi / (2.0 * 32.0))
        sin_val = math.sin(angle)
        sine_window.append(sin_val)
        
        angle_deg = math.degrees(angle)
        print(f"i={i:2d}: angle={angle:.6f} rad ({angle_deg:5.2f}°), sin={sin_val:9.6f}")
        
        # Check matches with our target value
        if abs(sin_val - abs(target_value)) < 1e-5:
            print(f"    *** MATCH: sin_val = {sin_val:.6f} ≈ |{target_value}| = {abs(target_value):.6f}")
        if abs(-sin_val - target_value) < 1e-5:
            print(f"    *** MATCH: -sin_val = {-sin_val:.6f} ≈ {target_value}")
    
    print("\n" + "=" * 50)
    print("Looking for matches with -0.372549:")
    
    for i, val in enumerate(sine_window):
        if abs(-val - target_value) < 1e-4:  # More lenient threshold
            print(f"Close match at i={i}: -{val:.6f} vs {target_value} (diff: {abs(-val - target_value):.6f})")
        if abs(val - abs(target_value)) < 1e-4:  # Check positive version
            print(f"Close match at i={i}: {val:.6f} vs |{target_value}| = {abs(target_value):.6f} (diff: {abs(val - abs(target_value)):.6f})")
    
    return sine_window

def main():
    print("Analyzing atracdenc SineWindow initialization")
    print("Target value: -0.372549")
    print("=" * 60)
    
    sine_window = calc_sine_window_atracdenc()
    
    # Also check if this could be used in overlap-add operations
    # where values might be negated or scaled
    print("\n" + "=" * 60)
    print("Checking potential overlap-add operations:")
    print("=" * 60)
    
    target = -0.372549
    for i, val in enumerate(sine_window):
        # Check various transformations that might occur in overlap-add
        transforms = [
            (val, f"SineWindow[{i}]"),
            (-val, f"-SineWindow[{i}]"),
            (val * 0.5, f"SineWindow[{i}] * 0.5"),
            (-val * 0.5, f"-SineWindow[{i}] * 0.5"),
            (val * 2.0, f"SineWindow[{i}] * 2.0"),
            (-val * 2.0, f"-SineWindow[{i}] * 2.0"),
        ]
        
        for transform_val, description in transforms:
            if abs(transform_val - target) < 1e-4:
                print(f"*** MATCH: {description} = {transform_val:.6f} ≈ {target}")
    
    # Let's also check if it's an index into the SineWindow
    print(f"\nDirect index check:")
    for i in range(len(sine_window)):
        if abs(sine_window[i] - abs(target)) < 0.01:  # Very lenient
            print(f"SineWindow[{i}] = {sine_window[i]:.6f} is close to |{target}| = {abs(target):.6f}")

if __name__ == "__main__":
    main()