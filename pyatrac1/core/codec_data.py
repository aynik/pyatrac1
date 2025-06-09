"""
Codec Data Initialization for ATRAC1.
This module defines the Atrac1CodecData class, which is responsible for
initializing and holding globally required data such as the ScaleTable and SineWindow.
This serves as the equivalent of TAtrac1Data.
"""

from pyatrac1.tables.scale_table import generate_scale_table
from pyatrac1.tables.filter_coeffs import generate_sine_window
from pyatrac1.tables.spectral_mapping import SPECS_PER_BLOCK, BFU_AMOUNT_TAB
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
