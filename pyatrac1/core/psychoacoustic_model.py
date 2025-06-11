"""
Implements the psychoacoustic model components for ATRAC1 codec.
This model guides bit allocation by identifying inaudible components
and weighting perceptible ones, mimicking human hearing.
"""

import math
import numpy as np  # type: ignore
from typing import List, Optional
from pyatrac1.core.codec_data import ScaledBlock
# Import FLAT_ATH_TAB_FRANK instead of ATH_LOOKUP_TABLE
from pyatrac1.tables.psychoacoustic import FLAT_ATH_TAB_FRANK, generate_loudness_curve
from pyatrac1.common.constants import LOUD_FACTOR


def ath_formula_frank(frequency: float) -> float:
    """
    Calculates the Absolute Threshold of Hearing (ATH) at a given frequency
    using the ATHformula_Frank method. Aligned with C++ implementation.

    Args:
        frequency: The frequency in Hz.

    Returns:
        The ATH value in decibels (dB).
    """
    # Clamp frequency to the range used by the C++ formula
    clamped_freq = max(10.0, min(29853.0, frequency))

    # freq_log = 40. * log10 (0.1 * freq);
    freq_log = 40.0 * math.log10(0.1 * clamped_freq)

    # index = (unsigned) freq_log;
    index = int(freq_log)

    # Boundary checks for index to prevent accessing out of bounds of FLAT_ATH_TAB_FRANK
    # The table has 35 lines * 4 values/line = 140 entries. Max index is 139.
    # Interpolation needs index and index + 1. So, max index used in tab[index+1] is 139.
    # This means index itself must be at most 138.
    if index < 0: # Should not happen with clamped_freq >= 10.0
        index = 0
    if index >= len(FLAT_ATH_TAB_FRANK) - 1:
        index = len(FLAT_ATH_TAB_FRANK) - 2 # Ensure tab[index+1] is valid

    val_at_index = float(FLAT_ATH_TAB_FRANK[index])
    val_at_index_plus_1 = float(FLAT_ATH_TAB_FRANK[index + 1])

    # Linear interpolation: tab [index] * (1 + index - freq_log) + tab [index+1] * (freq_log - index)
    result_millibels = val_at_index * (1.0 + index - freq_log) + \
                       val_at_index_plus_1 * (freq_log - index)

    # Original C++ ATHformula_Frank returns millibels * 0.01 (i.e. centibels relative to 20uPa)
    # The problem asks for this function to return dB.
    # The C++ CalcATH uses this result and then subtracts 100.0.
    # If tab values are in millibels, result_millibels * 0.01 gives dB.
    # Let's assume C++ returns result_millibels*0.01 which is already dB.
    # The Python function was documented to return dB.
    return result_millibels * 0.01


def calc_ath_spectrum(length: int, sample_rate: int) -> List[float]:
    """
    Calculates the Absolute Threshold of Hearing (ATH) spectrum.
    Mirrors C++ CalcATH function.

    Args:
        length: The number of spectral lines for which to calculate ATH.
                Typically corresponds to MDCT output size (e.g., 512 for long blocks).
        sample_rate: The audio sample rate in Hz (e.g., 44100).

    Returns:
        A list of ATH values in dB for each spectral line.
    """
    ath_spectrum_db: List[float] = []
    # mf = (float)sample_rate / 2000.0; (mf in C++ is for kHz conversion)
    # freq_khz = (i + 1.0) * (sample_rate / 2000.0) / length
    # freq_hz = (i + 1.0) * (sample_rate / 2.0) / length

    for i in range(length):
        # Frequency for the i-th spectral line (center frequency)
        # (i+1) used in C++ seems to be 1-based indexing for spectral lines.
        # If Python uses 0-based lines, it might be i or i+0.5.
        # Let's stick to (i+1) to match C++ line logic directly.
        freq_hz = (i + 1.0) * (sample_rate / 2.0) / length

        # freq_khz for the specific C++ formula term
        freq_khz = freq_hz / 1000.0

        # Call the aligned ath_formula_frank (expects Hz, returns dB)
        ath_db_at_freq = ath_formula_frank(freq_hz)

        # C++: trh = ATHformula_Frank(1.e3 * f) - 100;
        # This means ath_db_at_freq is equivalent to ATHformula_Frank(1.e3*f) from C++.
        # So, subtract 100.0 to match C++ trh.
        ath_db_adjusted = ath_db_at_freq - 100.0

        # C++: res[i] = trh - (f * f * 0.015); (where f is freq_khz)
        final_ath_db = ath_db_adjusted - (freq_khz * freq_khz * 0.015)
        ath_spectrum_db.append(final_ath_db)

    return ath_spectrum_db


class PsychoacousticModel:
    """
    Manages psychoacoustic calculations for ATRAC1, including loudness curve,
    spectral spread analysis, and loudness tracking.
    """

    def __init__(self):
        self.loudness_curve = generate_loudness_curve()
        self.current_loudness = LOUD_FACTOR

    def analyze_scale_factor_spread(
        self,
        scaled_blocks: List[ScaledBlock],
    ) -> float:
        """
        Analyzes the standard deviation of ScaleFactorIndex values across scaled_blocks.
        This provides a metric for whether the signal is more tone-like (low spread)
        or noise-like (high spread), as per spec line 913.

        Args:
            scaled_blocks: A list of ScaledBlock objects.

        Returns:
            A float between 0.0 and 1.0 indicating the normalized spread.
        """
        if not scaled_blocks:
            return 0.0  # Return 0 if no blocks to analyze

        scale_factor_indices = [block.scale_factor_index for block in scaled_blocks]

        if not scale_factor_indices:  # Should be caught by `if not scaled_blocks`
            return 0.0

        sigma = float(np.std(scale_factor_indices))  # type: ignore
        normalized_spread = min(1.0, sigma / 14.0)
        return normalized_spread

    def track_loudness(
        self,
        l0: float,
        l1: Optional[float] = None
    ) -> float:
        """
        Performs smoothing of loudness estimate across frames.
        The caller is responsible for deciding whether to use this smoothed value
        or an alternative based on transients.
        Updates self.current_loudness and returns the new smoothed value.

        Args:
            l0: Loudness value for channel 0 (or mono).
            l1: Optional loudness value for channel 1 (for stereo).

        Returns:
            The new smoothed loudness estimate (self.current_loudness).
        """
        if l1 is not None:  # Stereo case
            # Formula from C++: 0.98 * prevLoud + 0.01 * (l0 + l1)
            smoothed_val = 0.98 * self.current_loudness + 0.01 * (l0 + l1)
        else:  # Mono case
            # Formula from C++: 0.98 * prevLoud + 0.02 * l
            smoothed_val = 0.98 * self.current_loudness + 0.02 * l0

        self.current_loudness = smoothed_val
        return self.current_loudness
