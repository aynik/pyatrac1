"""
Main ATRAC1 Encoder class, orchestrating all sub-components.
"""

import numpy as np
from typing import List

from pyatrac1.common.debug_logger import log_debug, log_debug_detailed, log_bitstream
from pyatrac1.core.codec_data import Atrac1CodecData
from pyatrac1.core.qmf import Atrac1AnalysisFilterBank
from pyatrac1.core.mdct import Atrac1MDCT, BlockSizeMode
from pyatrac1.core.psychoacoustic_model import PsychoacousticModel, ath_formula_frank
from pyatrac1.core.transient_detection import TransientDetector
from pyatrac1.core.bitstream import Atrac1BitstreamWriter, Atrac1FrameData
from pyatrac1.core.scaling_quantization import TScaler, quantize_mantissas, ScaledBlock
from pyatrac1.core.bit_allocation_logic import Atrac1SimpleBitAlloc, BitsBooster

from pyatrac1.common import constants
from pyatrac1.tables.spectral_mapping import SPECS_START_LONG, SPECS_START_SHORT


class Atrac1Encoder:
    """
    Orchestrates the ATRAC1 encoding process, from raw audio samples
    to compressed ATRAC1 frames.
    """

    def __init__(self):
        """
        Initializes all necessary ATRAC1 codec components.
        """
        self.codec_data = Atrac1CodecData()
        self.qmf_filter_bank_ch0 = Atrac1AnalysisFilterBank()
        self.qmf_filter_bank_ch1 = Atrac1AnalysisFilterBank()
        self.mdct_processor = Atrac1MDCT()
        self.psychoacoustic_model_ch0 = PsychoacousticModel()
        self.psychoacoustic_model_ch1 = PsychoacousticModel()

        self.transient_detectors = {
            0: {
                "low": TransientDetector(),
                "mid": TransientDetector(),
                "high": TransientDetector(),
            },
            1: {
                "low": TransientDetector(),
                "mid": TransientDetector(),
                "high": TransientDetector(),
            },
        }
        self.bitstream_writer = Atrac1BitstreamWriter(self.codec_data)
        self.scaler = TScaler(self.codec_data)
        self.bit_allocator = Atrac1SimpleBitAlloc(self.codec_data)
        self.bits_booster = BitsBooster(self.codec_data)
        
        # Frame counter for logging
        self.frame_counter = 0
        
        # Track windowing initialization state
        self.windowing_initialized = [False, False]  # Per channel
        
        # Persistent spectral coefficient buffer to match atracdenc's Specs array
        # This buffer persists across frames and MDCT blocks to match atracdenc behavior
        self.spectral_coeffs_buffer = [
            np.zeros(constants.NUM_SAMPLES, dtype=np.float32),  # Channel 0
            np.zeros(constants.NUM_SAMPLES, dtype=np.float32)   # Channel 1
        ]

    def _get_representative_freq_for_bfu(
        self, bfu_index: int, is_long_block: bool, block_size_mode: BlockSizeMode = None
    ) -> float:
        """Calculates a representative frequency for a BFU."""
        if block_size_mode is not None:
            # Helper function to determine band from BFU index (matches atracdenc BfuToBand)
            def bfu_to_band(bfu_idx: int) -> int:
                if bfu_idx < 20:
                    return 0  # low band
                elif bfu_idx < 36:
                    return 1  # mid band
                else:
                    return 2  # high band
            
            # Determine which band this BFU belongs to
            band = bfu_to_band(bfu_index)
            
            # Use correct SPECS_START table based on block size mode for this band
            if ((band == 0 and block_size_mode.low_band_short) or 
                (band == 1 and block_size_mode.mid_band_short) or 
                (band == 2 and block_size_mode.high_band_short)):
                spec_idx_start = SPECS_START_SHORT[bfu_index]
            else:
                spec_idx_start = SPECS_START_LONG[bfu_index]
        else:
            # Fallback for backwards compatibility
            spec_idx_start = SPECS_START_LONG[bfu_index]

        return spec_idx_start * (constants.SAMPLE_RATE / 2.0) / constants.NUM_SAMPLES

    def _encode_single_channel(
        self, channel_samples: np.ndarray, channel_idx: int, frame_idx: int = 0
    ) -> bytes:
        """Encodes a single channel of audio data."""
        
        # Initialize windowing state on first use (like atracdenc dummy frames)
        if not self.windowing_initialized[channel_idx]:
            self.mdct_processor.initialize_windowing_state(channel_idx)
            self.windowing_initialized[channel_idx] = True

        pcm_input_list = channel_samples.tolist()
        
        # Log input PCM samples
        log_debug("PCM_INPUT", "samples", pcm_input_list, 
                  channel=channel_idx, frame=frame_idx, 
                  algorithm="encode_frame", sample_rate=constants.SAMPLE_RATE)

        qmf_bank = (
            self.qmf_filter_bank_ch0 if channel_idx == 0 else self.qmf_filter_bank_ch1
        )
        pcm_buf_low, pcm_buf_mid, pcm_buf_hi = qmf_bank.analysis(pcm_input_list, frame_idx)
        
        # Log QMF analysis outputs to match atracdenc format
        log_debug("QMF_OUTPUT", "samples", pcm_buf_low, 
                  channel=channel_idx, frame=frame_idx, band="LOW",
                  algorithm="qmf_analysis", qmf_band='"low"')
        log_debug("QMF_OUTPUT", "samples", pcm_buf_mid, 
                  channel=channel_idx, frame=frame_idx, band="MID",
                  algorithm="qmf_analysis", qmf_band='"mid"')
        log_debug("QMF_OUTPUT", "samples", pcm_buf_hi, 
                  channel=channel_idx, frame=frame_idx, band="HIGH",
                  algorithm="qmf_analysis", qmf_band='"high"')

        td_low = self.transient_detectors[channel_idx]["low"]
        td_mid = self.transient_detectors[channel_idx]["mid"]
        td_high = self.transient_detectors[channel_idx]["high"]

        transient_low = bool(td_low.detect(np.array(pcm_buf_low, dtype=np.float32), frame_idx, "LOW"))
        transient_mid = bool(td_mid.detect(np.array(pcm_buf_mid, dtype=np.float32), frame_idx, "MID"))
        transient_high = bool(td_high.detect(np.array(pcm_buf_hi, dtype=np.float32), frame_idx, "HIGH"))

        # Log transient detection results
        log_debug("TRANSIENT_DETECT", "decision", [transient_low, transient_mid, transient_high], 
                  channel=channel_idx, frame=frame_idx,
                  algorithm="transient_detection", 
                  low_transient=transient_low, mid_transient=transient_mid, high_transient=transient_high)

        block_size_mode = BlockSizeMode(transient_low, transient_mid, transient_high)
        
        # Log BSM values (converted to atracdenc format)
        bsm_low_val = 0 if transient_low else 2
        bsm_mid_val = 0 if transient_mid else 2  
        bsm_high_val = 0 if transient_high else 3
        log_debug("BSM_VALUES", "bsm", [bsm_low_val, bsm_mid_val, bsm_high_val],
                  channel=channel_idx, frame=frame_idx,
                  algorithm="block_size_mode", 
                  low_mdct_size=block_size_mode.low_mdct_size,
                  mid_mdct_size=block_size_mode.mid_mdct_size, 
                  high_mdct_size=block_size_mode.high_mdct_size)

        # Use persistent buffers from MDCT processor (like atracdenc)
        # Copy QMF output into the main buffer area [0:band_size]
        self.mdct_processor.pcm_buf_low[channel_idx][:len(pcm_buf_low)] = pcm_buf_low
        self.mdct_processor.pcm_buf_mid[channel_idx][:len(pcm_buf_mid)] = pcm_buf_mid
        self.mdct_processor.pcm_buf_hi[channel_idx][:len(pcm_buf_hi)] = pcm_buf_hi
        
        # Get references to persistent buffers for MDCT processing
        low_buf = self.mdct_processor.pcm_buf_low[channel_idx]
        mid_buf = self.mdct_processor.pcm_buf_mid[channel_idx]
        hi_buf = self.mdct_processor.pcm_buf_hi[channel_idx]
        
        # Use persistent spectral coefficients buffer to match atracdenc's Specs array behavior
        # This maintains coefficients across MDCT blocks and frames like atracdenc
        flat_spectrum_coeffs = self.spectral_coeffs_buffer[channel_idx]
        
        # Call atracdenc-compatible MDCT
        self.mdct_processor.mdct(flat_spectrum_coeffs, low_buf, mid_buf, hi_buf, 
                                block_size_mode, channel_idx, frame_idx)
        
        # Log combined spectral coefficients to match atracdenc format
        log_debug("SPECTRAL_COMBINED", "coeffs", flat_spectrum_coeffs,
                  channel=channel_idx, frame=frame_idx,
                  algorithm="spectrum_combination")

        psy_model = (
            self.psychoacoustic_model_ch0
            if channel_idx == 0
            else self.psychoacoustic_model_ch1
        )

        loudness_curve_coeffs = psy_model.loudness_curve[: constants.NUM_SAMPLES]
        channel_loudness_val = np.sum(flat_spectrum_coeffs**2 * loudness_curve_coeffs)
        
        # Log psychoacoustic analysis
        log_debug("PSYCHO_LOUDNESS", "curve", loudness_curve_coeffs,
                  channel=channel_idx, frame=frame_idx,
                  algorithm="psychoacoustic_model", channel_loudness=channel_loudness_val)

        channel_window_mask_td = 0
        if block_size_mode.low_band_short:
            channel_window_mask_td |= 1
        if block_size_mode.mid_band_short:
            channel_window_mask_td |= 2
        if block_size_mode.high_band_short:
            channel_window_mask_td |= 4

        overall_loudness = psy_model.track_loudness(
            channel_loudness_val, window_masks_ch0=channel_window_mask_td
        )

        chosen_bfu_amount_idx = 7
        num_active_bfus = self.codec_data.bfu_amount_tab[chosen_bfu_amount_idx]

        ath_per_bfu_scaled: List[float] = []
        is_long_mode_for_ath = not (
            block_size_mode.low_band_short
            or block_size_mode.mid_band_short
            or block_size_mode.high_band_short
        )
        for i in range(num_active_bfus):
            freq_bfu = self._get_representative_freq_for_bfu(i, is_long_mode_for_ath, block_size_mode)
            ath_db = ath_formula_frank(freq_bfu)
            ath_linear_amplitude = 10 ** (ath_db / 20.0)
            effective_ath_threshold = ath_linear_amplitude
            if overall_loudness > 1e-9:
                effective_ath_threshold = ath_linear_amplitude / overall_loudness
            else:
                effective_ath_threshold = float("inf")

            ath_per_bfu_scaled.append(effective_ath_threshold)
        
        # Log ATH calculations
        log_debug("PSYCHO_ATH", "thresholds", ath_per_bfu_scaled,
                  channel=channel_idx, frame=frame_idx,
                  algorithm="ath_calculation", overall_loudness=overall_loudness,
                  num_active_bfus=num_active_bfus, is_long_mode=is_long_mode_for_ath)

        # Helper function to determine band from BFU index (matches atracdenc BfuToBand)
        def bfu_to_band(bfu_idx: int) -> int:
            if bfu_idx < 20:
                return 0  # low band
            elif bfu_idx < 36:
                return 1  # mid band
            else:
                return 2  # high band

        scaled_blocks_channel: List[ScaledBlock] = []
        for i in range(num_active_bfus):
            num_specs_in_bfu = self.codec_data.specs_per_block[i]
            
            # Determine which band this BFU belongs to
            band = bfu_to_band(i)
            
            # Use correct SPECS_START table based on block size mode for this band
            if ((band == 0 and block_size_mode.low_band_short) or 
                (band == 1 and block_size_mode.mid_band_short) or 
                (band == 2 and block_size_mode.high_band_short)):
                start_idx_in_flat_spectrum = SPECS_START_SHORT[i]
            else:
                start_idx_in_flat_spectrum = SPECS_START_LONG[i]

            bfu_spectral_data = flat_spectrum_coeffs[
                start_idx_in_flat_spectrum : start_idx_in_flat_spectrum
                + num_specs_in_bfu
            ].tolist()
            scaled_block = self.scaler.scale(bfu_spectral_data)
            scaled_blocks_channel.append(scaled_block)
            
            # Log scaling for each BFU
            log_debug("SCALING", "coeffs", scaled_block.values,
                      channel=channel_idx, frame=frame_idx,
                      algorithm="bfu_scaling", bfu_idx=i,
                      scale_factor=scaled_block.scale_factor_index,
                      max_energy=scaled_block.max_energy)

        spread_factor = psy_model.analyze_scale_factor_spread(scaled_blocks_channel)

        is_long_block_for_bit_alloc_table = not (
            block_size_mode.low_band_short
            or block_size_mode.mid_band_short
            or block_size_mode.high_band_short
        )

        header_control_bits = 2 + 2 + 2 + 2 + constants.BITS_PER_BFU_AMOUNT_TAB_IDX + 2 + 3
        wl_header_bits = num_active_bfus * constants.BITS_PER_IDWL
        sf_header_bits = num_active_bfus * constants.BITS_PER_IDSF

        total_frame_bits = constants.SOUND_UNIT_SIZE * 8
        bits_available_for_mantissas = (
            total_frame_bits - header_control_bits - wl_header_bits - sf_header_bits
        )

        # Log bit allocation parameters
        log_debug("BIT_ALLOC_PARAMS", "params", [bits_available_for_mantissas, spread_factor],
                  channel=channel_idx, frame=frame_idx,
                  algorithm="bit_allocation", is_long_block=is_long_block_for_bit_alloc_table,
                  bits_available=bits_available_for_mantissas, spread_factor=spread_factor)

        word_lengths_channel_full, mantissa_bits_used = (
            self.bit_allocator.perform_iterative_allocation(
                scaled_blocks_channel,
                is_long_block_for_bit_alloc_table,
                ath_per_bfu_scaled,
                spread_factor,
                num_active_bfus,
                bits_available_for_mantissas,
            )
        )
        
        # Log bit allocation results
        log_debug("BIT_ALLOC", "word_lengths", word_lengths_channel_full,
                  channel=channel_idx, frame=frame_idx,
                  algorithm="bit_allocation", mantissa_bits_used=mantissa_bits_used,
                  num_active_bfus=num_active_bfus)

        surplus_bits = bits_available_for_mantissas - mantissa_bits_used
        boosted_word_lengths_channel_full = list(word_lengths_channel_full)
        if surplus_bits > 0:
            boosted_word_lengths_channel_full, _ = self.bits_booster.apply_boost(
                boosted_word_lengths_channel_full, surplus_bits, num_active_bfus
            )

        final_word_lengths = boosted_word_lengths_channel_full[:num_active_bfus]

        quantized_mantissas_channel: List[List[int]] = []
        for i in range(num_active_bfus):
            if i < len(scaled_blocks_channel) and final_word_lengths[i] > 0:
                scaled_values = scaled_blocks_channel[i].values
                wl = final_word_lengths[i]
                mantissas, _, _ = quantize_mantissas(
                    scaled_values, wl, perform_energy_adjustment=True
                )
                quantized_mantissas_channel.append(mantissas)
                
                # Log quantization for each BFU
                log_debug("QUANTIZE", "mantissas", mantissas,
                          channel=channel_idx, frame=frame_idx,
                          algorithm="quantization", bfu_idx=i, word_length=wl,
                          input_count=len(scaled_values), output_count=len(mantissas))
            else:
                # For inactive BFUs or missing blocks, create zeros with correct size
                bfu_size = self.codec_data.specs_per_block[i] if i < len(self.codec_data.specs_per_block) else 0
                quantized_mantissas_channel.append([0] * bfu_size)

        frame_data_channel = Atrac1FrameData()
        frame_data_channel.bsm_low = 2 if transient_low else 0
        frame_data_channel.bsm_mid = 2 if transient_mid else 0
        frame_data_channel.bsm_high = 3 if transient_high else 0

        frame_data_channel.bfu_amount_idx = chosen_bfu_amount_idx
        frame_data_channel.num_active_bfus = num_active_bfus
        frame_data_channel.word_lengths = final_word_lengths
        frame_data_channel.scale_factor_indices = [
            sb.scale_factor_index for sb in scaled_blocks_channel if sb
        ]
        frame_data_channel.quantized_mantissas = quantized_mantissas_channel
        
        # Log final frame data before bitstream writing
        scale_factors = [sb.scale_factor_index for sb in scaled_blocks_channel if sb]
        frame_structure = [chosen_bfu_amount_idx, bsm_low_val, bsm_mid_val, bsm_high_val]
        log_debug("FRAME_DATA", "structure", frame_structure,
                  channel=channel_idx, frame=frame_idx,
                  algorithm="frame_assembly", num_active_bfus=num_active_bfus,
                  bfu_amount_idx=chosen_bfu_amount_idx, 
                  word_lengths_count=len(final_word_lengths),
                  scale_factors_count=len(scale_factors))

        encoded_bytes = self.bitstream_writer.write_frame(frame_data_channel)
        
        # Log final bitstream
        log_bitstream("BITSTREAM_OUTPUT", encoded_bytes,
                      channel=channel_idx, frame=frame_idx,
                      algorithm="bitstream_writer", frame_size_bytes=len(encoded_bytes))
        
        return encoded_bytes

    def encode_frame(self, input_audio_samples: np.ndarray) -> bytes:
        """
        Processes a block of input audio samples and encodes it into an
        ATRAC1 compressed frame.

        Args:
            input_audio_samples: A NumPy array of raw audio samples for the current frame.
                                 Shape (NUM_SAMPLES,) for mono or (NUM_SAMPLES, 2) for stereo.
                                 NUM_SAMPLES should be constants.NUM_SAMPLES (512).

        Returns:
            A bytes object representing the compressed ATRAC1 frame(s).
            If mono, 212 bytes. If stereo, 424 bytes (concatenated).
        """
        if not isinstance(input_audio_samples, np.ndarray):
            raise TypeError("input_audio_samples must be a NumPy array.")

        if input_audio_samples.ndim == 1:
            if len(input_audio_samples) != constants.NUM_SAMPLES:
                raise ValueError(
                    f"Mono input must have {constants.NUM_SAMPLES} samples."
                )
            result = self._encode_single_channel(input_audio_samples, 0, self.frame_counter)
            self.frame_counter += 1
            return result

        elif input_audio_samples.ndim == 2:
            if (
                input_audio_samples.shape[0] == constants.NUM_SAMPLES
                and input_audio_samples.shape[1] == 2
            ):
                pass
            elif (
                input_audio_samples.shape[1] == constants.NUM_SAMPLES
                and input_audio_samples.shape[0] == 2
            ):
                input_audio_samples = input_audio_samples.T
            else:
                raise ValueError(
                    f"Stereo input must have shape ({constants.NUM_SAMPLES}, 2) or (2, {constants.NUM_SAMPLES}). "
                    f"Got {input_audio_samples.shape}"
                )

            channel_left_samples = input_audio_samples[:, 0]
            channel_right_samples = input_audio_samples[:, 1]

            frame_bytes_left = self._encode_single_channel(channel_left_samples, 0, self.frame_counter)
            frame_bytes_right = self._encode_single_channel(channel_right_samples, 1, self.frame_counter)
            self.frame_counter += 1

            return frame_bytes_left + frame_bytes_right
        else:
            raise ValueError("Input audio samples must be 1D (mono) or 2D (stereo).")
