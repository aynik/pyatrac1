"""
Implements transient detection logic for ATRAC1 codec.
This component identifies sudden amplitude changes in the audio signal
to trigger adaptive windowing in MDCT.
"""

import numpy as np
from pyatrac1.tables.filter_coeffs import HPF_FIR_COEFFS, HPF_FIR_LEN, HPF_PREV_BUF_SZ


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

    def detect(self, band_samples: np.ndarray) -> int:
        """
        Applies HPF and performs RMS analysis on the band samples to detect transients.

        Args:
            band_samples: Raw audio samples for the current band (e.g., from QMF output).

        Returns:
            An integer (0 or 1) indicating if a transient was detected in this band.
            0: No transient.
            1: Transient detected.
        """
        if not isinstance(band_samples, np.ndarray):
            band_samples = np.array(band_samples, dtype=np.float32)

        hpf_filtered_data = self._hpf_filter(band_samples)

        if hpf_filtered_data.size == 0:
            current_energy = 0.0
        else:
            mean_sq = np.mean(hpf_filtered_data**2)
            if mean_sq < 1e-20:
                current_energy = 0.0
            else:
                current_energy = np.sqrt(mean_sq)

        if current_energy <= 1e-10:
            current_energy_db = -100.0
        else:
            current_energy_db = 20 * np.log10(current_energy)

        is_transient = False
        if self.last_energy != -100.0 or current_energy_db != -100.0:
            db_change = current_energy_db - self.last_energy
            # Debug print for testing
            # print(f"DEBUG: last_energy={self.last_energy:.2f}, current_energy_db={current_energy_db:.2f}, db_change={db_change:.2f}")
            if db_change >= 16.0:
                is_transient = True
            elif db_change <= -12.0:  # Adjusted threshold based on actual HPF behavior
                is_transient = True

        self.last_energy = current_energy_db

        return 1 if is_transient else 0
