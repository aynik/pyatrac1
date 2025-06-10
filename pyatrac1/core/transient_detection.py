"""
Implements transient detection logic for ATRAC1 codec.
This component identifies sudden amplitude changes in the audio signal
to trigger adaptive windowing in MDCT.
"""

import numpy as np
from pyatrac1.tables.filter_coeffs import HPF_FIR_COEFFS, HPF_FIR_LEN, HPF_PREV_BUF_SZ
from pyatrac1.common.debug_logger import debug_logger


class TransientDetector:
    """
    Identifies transients in the audio signal by applying a High-Pass Filter (HPF)
    and performing RMS analysis.
    """

    def __init__(self):
        """
        Initializes the TransientDetector, including the HPF buffer and last energy state.
        HPFBuffer stores past samples required for FIR filtering across blocks.
        """
        self.hpf_buffer = np.zeros(HPF_FIR_LEN + HPF_PREV_BUF_SZ, dtype=np.float32)
        self.last_energy = -100.0  # Initialize to silence level in dB

    def _hpf_filter(self, input_data: np.ndarray) -> np.ndarray:
        """
        Applies a High-Pass Filter (HPF) to the input audio data.
        The filter is specified as a 21-tap symmetric FIR filter based on 10 coefficients.

        Args:
            input_data: A block of input audio samples.

        Returns:
            The high-pass filtered output, same size as input_data.
        """
        filtered_output = np.zeros_like(input_data, dtype=np.float32)

        for i, sample in enumerate(input_data):
            self.hpf_buffer[:-1] = self.hpf_buffer[1:]
            self.hpf_buffer[-1] = sample

            current_filter_input = self.hpf_buffer[
                HPF_PREV_BUF_SZ : HPF_PREV_BUF_SZ + HPF_FIR_LEN
            ]

            single_filtered_val = 0.0
            for j in range(10):
                single_filtered_val += HPF_FIR_COEFFS[j] * (
                    current_filter_input[j] + current_filter_input[HPF_FIR_LEN - 1 - j]
                )

            filtered_output[i] = single_filtered_val / 2.0

        return filtered_output

    def detect(self, band_samples: np.ndarray, frame_num: int = 0, band_name: str = "UNKNOWN") -> int:
        """
        Applies HPF and performs RMS analysis on the band samples to detect transients.
        Uses atracdenc-compatible algorithm that analyzes multiple short blocks within the frame.

        Args:
            band_samples: Raw audio samples for the current band (e.g., from QMF output).
            frame_num: Frame number for logging
            band_name: Band name (LOW/MID/HIGH) for logging

        Returns:
            An integer (0 or 1) indicating if a transient was detected in this band.
            0: No transient.
            1: Transient detected.
        """
        if not isinstance(band_samples, np.ndarray):
            band_samples = np.array(band_samples, dtype=np.float32)

        # Log input samples
        debug_logger.log_stage("TRANSIENT_INPUT", "SAMPLES", band_samples[:16], frame=frame_num, band=band_name)
        debug_logger.log_stage("TRANSIENT_INPUT_STATS", "RANGE", [np.min(band_samples), np.max(band_samples), np.mean(band_samples)], frame=frame_num, band=band_name)

        hpf_filtered_data = self._hpf_filter(band_samples)
        
        # Log HPF output to match atracdenc format
        debug_logger.log_stage("TRANSIENT_HPF_OUTPUT", "SAMPLES", hpf_filtered_data[:16], frame=frame_num, band=band_name, algorithm="transient_detection", operation="hpf_filter")
        debug_logger.log_stage("TRANSIENT_HPF_STATS", "RANGE", [np.min(hpf_filtered_data), np.max(hpf_filtered_data)], frame=frame_num, band=band_name, algorithm="transient_detection", operation="hpf_statistics")

        # Use atracdenc-compatible algorithm
        n_short_blocks = 4  # Number of short blocks to analyze
        short_sz = len(hpf_filtered_data) // n_short_blocks  # Size of each short block
        n_blocks_to_analyze = n_short_blocks + 1
        
        # Initialize RMS values array with previous energy
        rms_per_short_block = np.full(n_blocks_to_analyze, -100.0, dtype=np.float32)
        rms_per_short_block[0] = self.last_energy
        
        is_transient = False
        
        # Analyze each short block
        for i in range(1, n_blocks_to_analyze):
            # Calculate RMS energy for this block
            start_idx = (i - 1) * short_sz
            end_idx = min(start_idx + short_sz, len(hpf_filtered_data))
            
            if end_idx > start_idx:
                block_data = hpf_filtered_data[start_idx:end_idx]
                mean_sq = np.mean(block_data**2)
                
                if mean_sq < 1e-20:
                    raw_rms = 1e-10
                else:
                    raw_rms = np.sqrt(mean_sq)
                
                # Use atracdenc energy calculation: 19.0 * log10(rms)
                rms_per_short_block[i] = 19.0 * np.log10(raw_rms) if raw_rms > 1e-10 else -100.0
                
                # Log block energy for first few blocks
                if i <= 4:
                    debug_logger.log_stage("TRANSIENT_BLOCK_ENERGY", "VALUE", [rms_per_short_block[i]], 
                                         frame=frame_num, band=band_name, algorithm="transient_detection", operation="block_energy")
                
                # Check for energy increase (attack transient)
                energy_diff_up = rms_per_short_block[i] - rms_per_short_block[i - 1]
                if energy_diff_up > 16.0:
                    is_transient = True
                    debug_logger.log_stage("TRANSIENT_ATTACK", "DETECTED", [energy_diff_up, rms_per_short_block[i]], 
                                         frame=frame_num, band=band_name, algorithm="transient_detection", operation="attack_detection")
                
                # Check for energy decrease (decay transient) 
                energy_diff_down = rms_per_short_block[i - 1] - rms_per_short_block[i]
                if energy_diff_down > 20.0:
                    is_transient = True
                    debug_logger.log_stage("TRANSIENT_DECAY", "DETECTED", [energy_diff_down, rms_per_short_block[i]], 
                                         frame=frame_num, band=band_name, algorithm="transient_detection", operation="decay_detection")
        
        # Log energy values to match atracdenc format
        energy_values = [
            rms_per_short_block[1] if n_blocks_to_analyze > 1 else -100.0,
            rms_per_short_block[2] if n_blocks_to_analyze > 2 else -100.0, 
            -100.0  # Third energy value as in atracdenc
        ]
        debug_logger.log_stage("TRANSIENT_ENERGY", "VALUES", energy_values, frame=frame_num, band=band_name, algorithm="transient_detection", operation="energy_calculation")
        
        # Update last energy for next frame - use the maximum energy from the current frame
        # This ensures we capture the peak energy that could affect the next frame
        if len(rms_per_short_block) > 1:
            max_current_energy = max(rms_per_short_block[1:])  # Exclude the previous frame energy at index 0
            self.last_energy = max_current_energy
        
        # Calculate energy difference for decision logging
        db_change = rms_per_short_block[1] - rms_per_short_block[0] if n_blocks_to_analyze > 1 else 0.0
        
        # Log transient decision to match atracdenc format
        debug_logger.log_stage("TRANSIENT_DECISION", "RESULT", [db_change, int(is_transient)], frame=frame_num, band=band_name, algorithm="transient_detection", operation="final_decision")
        debug_logger.log_stage("TRANSIENT_THRESHOLDS", "VALUES", [16.0, -12.0, is_transient, 0.0], frame=frame_num, band=band_name, algorithm="transient_detection", operation="threshold_comparison")

        return 1 if is_transient else 0
