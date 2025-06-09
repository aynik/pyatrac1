#!/usr/bin/env python3
"""
Debug script to understand what's happening with MDCT sizes.
"""

import sys
sys.path.insert(0, '/Users/pablo/Projects/pytrac')

import numpy as np
from pyatrac1.core.encoder import Atrac1Encoder
from pyatrac1.core.mdct import BlockSizeMode

def debug_encoder_flow():
    """Debug the encoder flow to see where MDCT sizes go wrong."""
    
    # Create test signal - very smooth DC signal to avoid transients
    test_signal = np.full(512, 0.001, dtype=np.float32)
    
    print("🔍 Debugging encoder flow...")
    print(f"Input signal shape: {test_signal.shape}")
    
    # Create encoder and track the process
    encoder = Atrac1Encoder()
    
    # Get the QMF outputs manually
    pcm_input_list = test_signal.tolist()
    pcm_buf_low, pcm_buf_mid, pcm_buf_hi = encoder.qmf_filter_bank_ch0.analysis(pcm_input_list)
    
    print(f"QMF outputs:")
    print(f"  Low: {len(pcm_buf_low)} samples")
    print(f"  Mid: {len(pcm_buf_mid)} samples") 
    print(f"  High: {len(pcm_buf_hi)} samples")
    
    # Check transient detection
    td_low = encoder.transient_detectors[0]["low"]
    td_mid = encoder.transient_detectors[0]["mid"]
    td_high = encoder.transient_detectors[0]["high"]
    
    transient_low = bool(td_low.detect(np.array(pcm_buf_low, dtype=np.float32)))
    transient_mid = bool(td_mid.detect(np.array(pcm_buf_mid, dtype=np.float32)))
    transient_high = bool(td_high.detect(np.array(pcm_buf_hi, dtype=np.float32)))
    
    print(f"Transient detection:")
    print(f"  Low: {transient_low}")
    print(f"  Mid: {transient_mid}")
    print(f"  High: {transient_high}")
    
    # Create BlockSizeMode
    block_size_mode = BlockSizeMode(transient_low, transient_mid, transient_high)
    
    print(f"Block size mode:")
    print(f"  Low MDCT size: {block_size_mode.low_mdct_size}")
    print(f"  Mid MDCT size: {block_size_mode.mid_mdct_size}")
    print(f"  High MDCT size: {block_size_mode.high_mdct_size}")
    
    # Also test manual long window configuration
    print(f"\\nTesting manual long window configuration...")
    manual_block_size_mode = BlockSizeMode(False, False, False)
    print(f"Manual block size mode:")
    print(f"  Low MDCT size: {manual_block_size_mode.low_mdct_size}")
    print(f"  Mid MDCT size: {manual_block_size_mode.mid_mdct_size}")
    print(f"  High MDCT size: {manual_block_size_mode.high_mdct_size}")
    
    # Try MDCT on each band
    try:
        print("Testing MDCT on each band...")
        
        specs_low = encoder.mdct_processor.mdct(pcm_buf_low, block_size_mode, "low")
        print(f"  Low MDCT output: {len(specs_low)} coefficients")
        
        specs_mid = encoder.mdct_processor.mdct(pcm_buf_mid, block_size_mode, "mid")
        print(f"  Mid MDCT output: {len(specs_mid)} coefficients")
        
        specs_high = encoder.mdct_processor.mdct(pcm_buf_hi, block_size_mode, "high")
        print(f"  High MDCT output: {len(specs_high)} coefficients")
        
        # Check if any are empty
        if len(specs_low) == 0:
            print("❌ Low band MDCT returned empty array!")
        if len(specs_mid) == 0:
            print("❌ Mid band MDCT returned empty array!")
        if len(specs_high) == 0:
            print("❌ High band MDCT returned empty array!")
            
        total_coeffs = len(specs_low) + len(specs_mid) + len(specs_high)
        print(f"Total coefficients: {total_coeffs} (expected: 512)")
        
        # Now test with manual long window mode
        print(f"\\nTesting manual long window MDCT...")
        specs_low_manual = encoder.mdct_processor.mdct(pcm_buf_low, manual_block_size_mode, "low")
        print(f"  Low MDCT output (manual): {len(specs_low_manual)} coefficients")
        
        specs_mid_manual = encoder.mdct_processor.mdct(pcm_buf_mid, manual_block_size_mode, "mid")
        print(f"  Mid MDCT output (manual): {len(specs_mid_manual)} coefficients")
        
        specs_high_manual = encoder.mdct_processor.mdct(pcm_buf_hi, manual_block_size_mode, "high")
        print(f"  High MDCT output (manual): {len(specs_high_manual)} coefficients")
        
        total_coeffs_manual = len(specs_low_manual) + len(specs_mid_manual) + len(specs_high_manual)
        print(f"Total coefficients (manual): {total_coeffs_manual} (expected: 512)")
        
    except Exception as e:
        print(f"❌ MDCT failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_encoder_flow()