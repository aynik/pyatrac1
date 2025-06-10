"""
Main ATRAC1 Decoder class, orchestrating all sub-components for playback.
"""

import numpy as np
from typing import List

from pyatrac1.core.codec_data import Atrac1CodecData
from pyatrac1.core.mdct import (
    Atrac1MDCT,
    BlockSizeMode,
)
from pyatrac1.core.qmf import Atrac1SynthesisFilterBank
from pyatrac1.core.bitstream import Atrac1BitstreamReader
from pyatrac1.tables.scale_table import ATRAC1_SCALE_TABLE


class Atrac1Decoder:
    """
    Orchestrates the ATRAC1 decoding process, from compressed frames
    to raw audio samples.
    """

    def __init__(self):
        """
        Initializes all necessary ATRAC1 codec components for decoding.
        """
        self.codec_data = Atrac1CodecData()
        self.mdct_processor = Atrac1MDCT()
        self.qmf_synthesis_filter_bank = Atrac1SynthesisFilterBank()
        self.bitstream_reader = Atrac1BitstreamReader(self.codec_data)


    def _dequantize_and_inverse_scale(
        self,
        quantized_values_per_block: List[List[int]],
        bits_per_each_block: List[int],
        scale_factor_indices: List[int],
    ) -> List[np.ndarray]:
        """
        Dequantizes and inverse scales the quantized MDCT coefficients.

        Args:
            quantized_values_per_block: 2D list of quantized integer values for each block.
            bits_per_each_block: List of allocated bits per block corresponding to quantized_values.
            scale_factor_indices: List of scale factor indices for each block.

        Returns:
            List of NumPy arrays, where each array contains the dequantized and inverse-scaled
            floating-point MDCT coefficients for a block.
        """
        mdct_coeffs_per_block: List[np.ndarray] = []

        for i, block_quant_values in enumerate(quantized_values_per_block):
            actual_bits_for_sample = bits_per_each_block[i]
            scale_factor_idx = scale_factor_indices[i]
            scale_value = ATRAC1_SCALE_TABLE[scale_factor_idx]

            dequantized_block_coeffs = np.zeros(
                len(block_quant_values), dtype=np.float32
            )

            if actual_bits_for_sample >= 2:
                # Calculate `multiple` used during quantization
                multiple = (1 << (actual_bits_for_sample - 1)) - 1

                for j, quantized_int_val in enumerate(block_quant_values):
                    # Dequantize: value / multiple
                    dequantized_val = quantized_int_val / multiple
                    # Apply scale factor (multiply by scale_value, matching atracdenc)
                    dequantized_block_coeffs[j] = dequantized_val * scale_value
            elif actual_bits_for_sample == 1:
                # If bits_per_sample was 1, no actual data was written to the bitstream.
                # The C++ encoder explicitly pushed 0.0f (or related default) to intermediate data.
                # So here, dequantized values are effectively 0.
                pass  # dequantized_block_coeffs remains zeros as initialized

            mdct_coeffs_per_block.append(dequantized_block_coeffs)

        return mdct_coeffs_per_block

    def decode_frame(self, compressed_frame_bytes: bytes) -> np.ndarray:
        """
        Processes a compressed ATRAC1 frame and decodes it into raw audio samples.

        Args:
            compressed_frame_bytes: A bytes object representing the compressed ATRAC1 frame.

        Returns:
            A NumPy array of raw audio samples for the decoded frame.
        """
        # 1. Frame Disassembly
        frame_data_obj = self.bitstream_reader.read_frame(compressed_frame_bytes)

        # Use simplified boolean flags from the bitstream reader
        block_size_mode = BlockSizeMode(
            frame_data_obj.low_band_short,
            frame_data_obj.mid_band_short,
            frame_data_obj.high_band_short
        )

        # bfu_amount_idx is available in frame_data_obj.bfu_amount_idx if needed.
        _ = frame_data_obj.bfu_amount_idx

        bits_per_each_block = frame_data_obj.word_lengths
        scale_factor_indices = frame_data_obj.scale_factor_indices
        quantized_values_per_block = frame_data_obj.quantized_mantissas

        # 2. Dequantization and Inverse Scaling
        mdct_coeffs_per_bfu = self._dequantize_and_inverse_scale(
            quantized_values_per_block, bits_per_each_block, scale_factor_indices
        )

        # 3. Reconstruct flat spectrum from BFU coefficients
        # The encoder creates a flat spectrum with sequential layout: [low, mid, high]
        # We need to reconstruct this from the BFU-organized coefficients
        from pyatrac1.tables.spectral_mapping import SPECS_START_LONG, SPECS_START_SHORT
        from pyatrac1.common import constants
        
        def bfu_to_band(bfu_idx: int) -> int:
            """Map BFU index to band (0=low, 1=mid, 2=high)"""
            if bfu_idx < 20:
                return 0  # low band
            elif bfu_idx < 36:
                return 1  # mid band
            else:
                return 2  # high band
        
        flat_spectrum_coeffs = np.zeros(constants.NUM_SAMPLES, dtype=np.float32)
        
        for bfu_idx, bfu_coeffs in enumerate(mdct_coeffs_per_bfu):
            # Determine which band this BFU belongs to
            band = bfu_to_band(bfu_idx)
            
            # Use the correct SPECS_START table based on block size mode for this band
            if band == 0 and block_size_mode.low_band_short:
                specs_start_table = SPECS_START_SHORT
            elif band == 1 and block_size_mode.mid_band_short:
                specs_start_table = SPECS_START_SHORT
            elif band == 2 and block_size_mode.high_band_short:
                specs_start_table = SPECS_START_SHORT
            else:
                specs_start_table = SPECS_START_LONG
            
            if bfu_idx < len(specs_start_table):
                start_idx = specs_start_table[bfu_idx]
                end_idx = start_idx + len(bfu_coeffs)
                if end_idx <= constants.NUM_SAMPLES:
                    flat_spectrum_coeffs[start_idx:end_idx] = bfu_coeffs

        # 4. Extract band coefficients from flat spectrum
        low_mdct_size = block_size_mode.low_mdct_size
        mid_mdct_size = block_size_mode.mid_mdct_size
        high_mdct_size = block_size_mode.high_mdct_size
        
        # Extract coefficients for each band
        low_coeffs = flat_spectrum_coeffs[:low_mdct_size]
        mid_coeffs = flat_spectrum_coeffs[low_mdct_size:low_mdct_size + mid_mdct_size]
        high_coeffs = flat_spectrum_coeffs[low_mdct_size + mid_mdct_size:low_mdct_size + mid_mdct_size + high_mdct_size]

        # 5. IMDCT using atracdenc-compatible interface  
        # Create properly sized output buffers with overlap regions
        low_buf = np.zeros(256 + 16, dtype=np.float64)
        mid_buf = np.zeros(256 + 16, dtype=np.float64)
        hi_buf = np.zeros(512 + 16, dtype=np.float64)
        
        # Initialize overlap regions from previous state if available
        if hasattr(self, 'prev_overlap_low'):
            low_buf[256-16:256] = self.prev_overlap_low
        if hasattr(self, 'prev_overlap_mid'):
            mid_buf[256-16:256] = self.prev_overlap_mid  
        if hasattr(self, 'prev_overlap_high'):
            hi_buf[512-16:512] = self.prev_overlap_high
        
        # Call atracdenc-compatible IMDCT 
        self.mdct_processor.imdct(flat_spectrum_coeffs, block_size_mode, low_buf, mid_buf, hi_buf)
        
        # Save overlap regions for next frame
        self.prev_overlap_low = low_buf[256-16:256].copy()
        self.prev_overlap_mid = mid_buf[256-16:256].copy()
        self.prev_overlap_high = hi_buf[512-16:512].copy()
        
        # Extract reconstructed samples for QMF synthesis
        reconstructed_low_samples_list = low_buf[:low_mdct_size].tolist()
        reconstructed_mid_samples_list = mid_buf[:mid_mdct_size].tolist()
        reconstructed_high_samples_list = hi_buf[:high_mdct_size].tolist()

        # 4. QMF Synthesis
        # Ensure inputs to QMF synthesis are correctly sized lists of floats
        reconstructed_audio_list = self.qmf_synthesis_filter_bank.synthesis(
            reconstructed_low_samples_list,
            reconstructed_mid_samples_list,
            reconstructed_high_samples_list,
        )

        return np.array(reconstructed_audio_list, dtype=np.float32)
