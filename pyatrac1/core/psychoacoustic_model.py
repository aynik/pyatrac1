"""
Implements the psychoacoustic model components for ATRAC1 codec.
This model guides bit allocation by identifying inaudible components
and weighting perceptible ones, mimicking human hearing.
"""

import math
import numpy as np  # type: ignore
from typing import List, Optional
from pyatrac1.core.codec_data import ScaledBlock
from pyatrac1.tables.psychoacoustic import ATH_MILLIBEL_TAB_INTERNAL, generate_loudness_curve
from pyatrac1.common.constants import LOUD_FACTOR


def ath_formula_frank(frequency: float) -> float:
    """
    Calculates the Absolute Threshold of Hearing (ATH) at a given frequency
    using a lookup table and linear interpolation, based on atracdenc's implementation.

    Args:
        frequency: The frequency in Hz.

    Returns:
        The ATH value in decibels.
    """
    freq = max(10.0, min(29853.0, frequency)) # Matches C++ limits more closely

    # freq_log calculation from C++
    # 4 steps per third, starting at 10 Hz. Max index will be < 128 for freq <= 29853 Hz
    freq_log = 40.0 * math.log10(0.1 * freq) if freq > 0 else 0.0
    index = math.floor(freq_log)

    # Ensure index is within bounds for the table (0 to 126 for interpolation tab[index+1])
    # Max C++ index is 127. So index can go up to 127.
    # If index is 127, index+1 is 128, which is out of bounds for a 128-element list.
    # The C++ code uses tab[index] and tab[index+1]. So index must be <= 126.
    index = max(0, min(int(index), len(ATH_MILLIBEL_TAB_INTERNAL) - 2))

    # Linear interpolation from C++
    # result = tab [index] * (1 + index - freq_log) + tab [index+1] * (freq_log - index)
    val1 = ATH_MILLIBEL_TAB_INTERNAL[index]
    val2 = ATH_MILLIBEL_TAB_INTERNAL[index+1]
    interpolated_millibel = val1 * (1.0 + index - freq_log) + val2 * (freq_log - index)

    # Convert millibels to dB
    ath_db = 0.01 * interpolated_millibel
    return ath_db


def calc_ath_spectrum_db(num_spectral_lines: int, sample_rate: float) -> List[float]:
    """
    Calculates the ATH for each spectral line, similar to C++ CalcATH.
    The resulting values are in dB.
    """
    ath_spectrum_db_values: List[float] = []
    for i in range(num_spectral_lines):
        # Calculate f_khz = (i + 1.0) * (sample_rate / 2000.0) / num_spectral_lines
        # Ensure floating point division
        f_khz = (float(i) + 1.0) * (sample_rate / 2000.0) / float(num_spectral_lines)

        # Call ath_formula_frank to get base ATH dB
        # The frequency input to ath_formula_frank is in Hz
        ath_frank_output = ath_formula_frank(f_khz * 1000.0)

        # Further adjustments as per C++ CalcATH
        ath_db = ath_frank_output - 100.0
        ath_db -= f_khz * f_khz * 0.015 # Equivalent to C++: trh -= freq * freq * 0.015f; (where freq is in kHz)

        ath_spectrum_db_values.append(ath_db)
    return ath_spectrum_db_values


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
        l1: Optional[float] = None,
        window_masks_ch0: int = 0,
        window_masks_ch1: int = 0,
    ) -> float:
        """
        Provides a mechanism to smooth the loudness estimate across frames.
        Conditionally bypasses smoothing during transient events.

        Args:
            l0: Loudness value for channel 0.
            l1: Optional loudness value for channel 1 (for stereo).
            window_masks_ch0: Window mask for channel 0, indicating transients.
            window_masks_ch1: Window mask for channel 1, indicating transients.

        Returns:
            The smoothed loudness estimate.
        """
        if l1 is not None:  # Stereo
            if window_masks_ch0 == 0 and window_masks_ch1 == 0:
                self.current_loudness = 0.98 * self.current_loudness + 0.01 * (l0 + l1)
            else:  # If transients, bypass smoothing
                self.current_loudness = (l0 + l1) / 2.0
        else:  # Mono
            if window_masks_ch0 == 0:
                self.current_loudness = 0.98 * self.current_loudness + 0.02 * l0
            else:  # If transients, bypass smoothing
                self.current_loudness = l0  # Use current loudness without smoothing

        return self.current_loudness
