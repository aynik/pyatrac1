"""
Implements the psychoacoustic model components for ATRAC1 codec.
This model guides bit allocation by identifying inaudible components
and weighting perceptible ones, mimicking human hearing.
"""

import math
import numpy as np  # type: ignore
from typing import List, Optional
from pyatrac1.core.codec_data import ScaledBlock
from pyatrac1.tables.psychoacoustic import ATH_LOOKUP_TABLE, generate_loudness_curve
from pyatrac1.common.constants import LOUD_FACTOR


def ath_formula_frank(frequency: float) -> float:
    """
    Calculates the Absolute Threshold of Hearing (ATH) at a given frequency
    using the ATHformula_Frank method from the ATRAC1 specification.

    Args:
        frequency: The frequency in Hz.

    Returns:
        The ATH value in decibels.
    """
    freq = max(10.0, min(29853.0, frequency))
    target_freq_log = 40.0 * math.log10(0.1 * freq) if freq > 0 else -float("inf")

    # Find two closest frequencies in the lookup table for interpolation
    lower_idx = 0
    upper_idx = len(ATH_LOOKUP_TABLE) - 1

    for i in range(len(ATH_LOOKUP_TABLE) - 1):
        if ATH_LOOKUP_TABLE[i][0] <= freq <= ATH_LOOKUP_TABLE[i + 1][0]:
            lower_idx = i
            upper_idx = i + 1
            break

    if freq < ATH_LOOKUP_TABLE[0][0]:
        lower_idx = 0
        upper_idx = 0
    elif freq > ATH_LOOKUP_TABLE[-1][0]:
        lower_idx = len(ATH_LOOKUP_TABLE) - 1
        upper_idx = len(ATH_LOOKUP_TABLE) - 1

    f0_hz = ATH_LOOKUP_TABLE[lower_idx][0]
    f1_hz = ATH_LOOKUP_TABLE[upper_idx][0]

    # Calculate freq_log for the table's bracketing frequencies
    freq_log_f0 = 40.0 * math.log10(0.1 * f0_hz) if f0_hz > 0 else -float("inf")
    freq_log_f1 = 40.0 * math.log10(0.1 * f1_hz) if f1_hz > 0 else -float("inf")

    interpolated_mb_values_at_target_freq_log: List[float] = []

    for k in range(4):
        mb_at_f0 = float(ATH_LOOKUP_TABLE[lower_idx][k + 1])
        mb_at_f1 = float(ATH_LOOKUP_TABLE[upper_idx][k + 1])

        if freq_log_f0 == freq_log_f1 or target_freq_log == -float("inf"):
            interpolated_k = mb_at_f0
        elif freq_log_f0 == -float("inf") and freq_log_f1 == -float("inf"):
            interpolated_k = mb_at_f0
        else:
            denominator = freq_log_f1 - freq_log_f0
            if denominator == 0:
                interpolated_k = mb_at_f0
            else:
                interpolation_factor = (target_freq_log - freq_log_f0) / denominator
                interpolation_factor = max(0.0, min(1.0, interpolation_factor))
                interpolated_k = mb_at_f0 + (mb_at_f1 - mb_at_f0) * interpolation_factor
        interpolated_mb_values_at_target_freq_log.append(interpolated_k)

    if not interpolated_mb_values_at_target_freq_log:
        final_interpolated_mb = 0.0
    else:
        final_interpolated_mb = sum(interpolated_mb_values_at_target_freq_log) / len(
            interpolated_mb_values_at_target_freq_log
        )

    trh = final_interpolated_mb * 0.01
    trh -= freq * freq * 0.015

    return trh


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
