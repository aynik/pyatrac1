"""
Codec Data Initialization for ATRAC1.
This module defines the Atrac1CodecData class, which is responsible for
initializing and holding globally required data such as the ScaleTable and SineWindow.
This serves as the equivalent of TAtrac1Data.
"""

import math
from pyatrac1.tables.scale_table import generate_scale_table
from pyatrac1.tables.filter_coeffs import generate_sine_window
from pyatrac1.tables.spectral_mapping import SPECS_PER_BLOCK, BFU_AMOUNT_TAB, SPECS_START_LONG
from pyatrac1.common.constants import MAX_BFUS, NUM_SAMPLES, SAMPLE_RATE
from pyatrac1.core.psychoacoustic_model import calc_ath_spectrum_db
from typing import List


class ScaledBlock:
    """
    Represents a block of scaled spectral coefficients and its associated scale factor.
    Output of the TScaler.scale method.
    """

    def __init__(
        self, scale_factor_index: int, scaled_values: List[float], max_energy: float
    ):
        self.scale_factor_index = scale_factor_index
        self.values = scaled_values
        self.max_energy = max_energy  # Max energy of the original block before scaling


class Atrac1CodecData:
    """
    Manages and provides access to essential, pre-computed codec data for ATRAC1.
    Initializes tables like ScaleTable, SineWindow, SpecsPerBlock, and BfuAmountTab
    upon instantiation.
    """

    def __init__(self):
        """
        Initializes the Atrac1CodecData instance by generating/loading and storing
        the ScaleTable, SineWindow, SpecsPerBlock, and BfuAmountTab.
        """
        self.scale_table = generate_scale_table()
        self.sine_window = generate_sine_window()
        self.specs_per_block = SPECS_PER_BLOCK
        self.bfu_amount_tab = BFU_AMOUNT_TAB

        # Calculate ATH spectrum (dB values for each spectral line)
        spectral_ath_db_list = calc_ath_spectrum_db(NUM_SAMPLES, float(SAMPLE_RATE))

        # Initialize and populate the ATH minimum energy table for long blocks
        self.ath_long_min_energy_table: List[float] = [0.0] * MAX_BFUS
        for j in range(MAX_BFUS):  # j is BFU index
            min_db_for_bfu = float('inf')
            start_spec_idx = SPECS_START_LONG[j]
            # self.specs_per_block is already initialized
            num_specs_in_bfu = self.specs_per_block[j]

            if num_specs_in_bfu == 0:
                self.ath_long_min_energy_table[j] = float('inf') # Represents extremely high energy for silence
            else:
                for k in range(num_specs_in_bfu):
                    spectral_line_index = start_spec_idx + k
                    if spectral_line_index < NUM_SAMPLES: # Ensure we don't go out of bounds
                        current_line_ath_db = spectral_ath_db_list[spectral_line_index]
                        min_db_for_bfu = min(min_db_for_bfu, current_line_ath_db)

                if min_db_for_bfu == float('inf'):
                    # This case should ideally not be hit if num_specs_in_bfu > 0
                    # and SPECS_START_LONG entries are valid.
                    # It implies no valid spectral lines were found for this BFU.
                    self.ath_long_min_energy_table[j] = float('inf')
                else:
                    # Convert min dB for BFU to energy: E = 10^(dB/10)
                    self.ath_long_min_energy_table[j] = math.pow(10.0, min_db_for_bfu / 10.0)
